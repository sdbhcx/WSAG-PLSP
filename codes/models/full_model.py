import torch
import torch.nn as nn
import torch.nn.functional as F

from models.encoder_clip import VisionTransformer
from models.decoder_affordance import Affordance_Decoder
from models.SAM_decoder import SAM_Decoder_Simple


def masked_avg_pooling(features, mask):
    """
    Perform masked average pooling on features.
    
    Args:
        features: [B, N, D] where N is number of patches/tokens
        mask: [B, N] binary mask for pooling
    
    Returns:
        pooled_features: [B, D]
    """
    mask_sum = mask.sum(dim=1, keepdim=True).clamp(min=1e-6)
    pooled = (features * mask.unsqueeze(-1)).sum(dim=1) / mask_sum
    return pooled


def selective_prototype_contrast_loss(anchor, positives, negatives, temperature=0.07):
    """
    Compute selective prototype contrast loss.
    
    Args:
        anchor: [B, D] anchor prototypes
        positives: list of [B, D] positive prototypes
        negatives: list of [B, D] negative prototypes
        temperature: temperature parameter for softmax
    
    Returns:
        loss: scalar loss value
    """
    B, D = anchor.shape
    
    # Combine all prototypes
    all_prototypes = torch.cat([anchor.unsqueeze(1)] + positives + negatives, dim=1)  # [B, 1+P+N, D]
    
    # Compute similarity scores
    anchor_expanded = anchor.unsqueeze(1)  # [B, 1, D]
    similarities = torch.matmul(anchor_expanded, all_prototypes.transpose(1, 2))  # [B, 1, 1+P+N]
    similarities = similarities.squeeze(1) / temperature  # [B, 1+P+N]
    
    # Compute loss: -log(exp(sim_pos) / sum(exp(sim_all)))
    # Positive prototypes are at indices 1 to 1+len(positives)
    num_positives = len(positives)
    pos_similarities = similarities[:, 1:1+num_positives]  # [B, P]
    
    # Sum over positive prototypes
    pos_exp = torch.exp(pos_similarities).sum(dim=1)  # [B]
    all_exp = torch.exp(similarities).sum(dim=1)  # [B]
    
    loss = -torch.log(pos_exp / all_exp).mean()
    
    return loss


class ModelAGDsup(nn.Module):
    """ Masked Autoencoder with VisionTransformer backbone
    """
    def __init__(self, 
                 encoder_type, encoder_params,
                 img_size=224, patch_size=16, 
                 decoder_embed_dim=512, decoder_num_heads=16,
                 aff_decoder_depth=4,
                 drop_rate=0., attn_drop_rate=0., drop_path_rate=0.1,
                 mlp_ratio=4., norm_layer=nn.LayerNorm, 
                 decoder_layer_scale_init_value=0.1,
                 init_std=0.02, pred_model_type="SAM",
                 pred_decoder_args={"mlp_dim":1024, "depth":2, "use_up":2, "use_additional_token":True},
                 margin=0.5
                 ):
        super().__init__()

        self.margin = margin
        self.encoder = VisionTransformer(
            input_resolution=img_size, patch_size=patch_size, **encoder_params)
        
        self.verb_fuser = Affordance_Decoder(
            num_patches=self.encoder.num_patches,
            decoder_embed_dim=decoder_embed_dim, regresser_depth=aff_decoder_depth, 
            num_heads=decoder_num_heads, 
            mlp_ratio=mlp_ratio, qkv_bias=True, qk_scale=None, 
            drop_rate=drop_rate, attn_drop_rate=attn_drop_rate, drop_path_rate=drop_path_rate, 
            norm_layer=norm_layer, init_values=decoder_layer_scale_init_value, init_std=init_std
        )
        
        self.pred_decoder = SAM_Decoder_Simple(
            transformer_dim=decoder_embed_dim,
            activation=nn.GELU,
            **pred_decoder_args,
        )
          
        self.num_patches = self.encoder.num_patches
        self.patch_size = patch_size
        
        self.exo_cls = nn.Sequential(
            nn.Linear(512, 256),
            nn.GELU(),
            nn.Linear(256, 36)
        )
        
        self.noun_transform = nn.Sequential(
            nn.Linear(512, 512),
            nn.GELU(),
            nn.Linear(512, 512)
        )
        
        self.reason = nn.Sequential(
            nn.Linear(1024, 512),
            nn.GELU(),
            nn.Linear(512, 512)
        )
        
        self.proto_projector = nn.Sequential(
            nn.Linear(512, 512),
            nn.GELU(),
            nn.Linear(512, 512)
        )
        
        
    def forward(self, imgs, text_feat, exo=None, exo_obj_mask=None, num_exo=1, 
                ego_part_mask=None, ego_obj_mask=None, exo_obj_mask_full=None):
        # 1. 提取第一视角图像特征
        _, x = self.encoder(imgs)

        proj_x = self.proto_projector(x)
        # 2. 处理动作语义特征
        v = text_feat.float().unsqueeze(1)
        # 3. 预测物体特征
        pred_noun = self.noun_transform(x[:, 0:1, ].detach()) 
        # 4. 融合物体特征和动作语义特征
        pred_part = self.reason(torch.cat([pred_noun, v], dim=2))
        # 使用 verb_fuser 网络融合视觉特征和融合后的特征
        aff_token, _, _ = self.verb_fuser(x, pred_part+v)
        
        pred_heatmap = self.pred_decoder(x, aff_token)

        # 构建原型用于选择性原型对比损失
        proto_loss = None
        if exo is not None:
            # with torch.no_grad():
            _, exo = self.encoder(exo)
            proj_exo = self.proto_projector(exo)

            exo_token = (exo[:, 1:] * exo_obj_mask).sum(dim=1)
            D = aff_token.shape[-1]
            aff_token_expand = aff_token.expand(-1, num_exo, -1).reshape(-1, D)
            sim_loss = torch.max(
                1 - F.cosine_similarity(aff_token_expand, exo_token.detach(), dim=1) - self.margin, 
                torch.zeros(len(exo_token)).to(x.device))
            
            # 构建选择性原型对比损失
            if ego_part_mask is not None and exo_obj_mask_full is not None:
                proto_loss = self.compute_prototype_contrast_loss(
                    proj_x, proj_exo, ego_part_mask, ego_obj_mask, 
                    exo_obj_mask_full, num_exo
                )
            
            return pred_heatmap, sim_loss, self.exo_cls(exo_token), pred_noun, pred_part, proto_loss
        else:
            return pred_heatmap, pred_noun, pred_part
    
    def compute_prototype_contrast_loss(self, ego_features, exo_features, 
                                       ego_part_mask, ego_obj_mask, 
                                       exo_obj_mask_full, num_exo):
        """
        Compute selective prototype contrast loss using PLSP masks.
        
        Args:
            ego_features: [B, N, D] projected ego features
            exo_features: [B*num_exo, N, D] projected exo features
            ego_part_mask: [B, N] binary mask for affordance parts
            ego_obj_mask: [B, N] binary mask for objects
            exo_obj_mask_full: [B*num_exo, N] binary mask for exo objects
            num_exo: number of exo images per ego image
        
        Returns:
            proto_loss: scalar loss value
        """
        B, N, D = ego_features.shape
        B_exo = exo_features.shape[0]
        
        # 1. 构建自我中心图像的原型
        # 锚点原型：使用部件掩码进行掩码平均池化
        ego_part_mask_flat = ego_part_mask.reshape(B, -1)  # [B, N]
        anchor_prototype = masked_avg_pooling(ego_features[:, 1:], ego_part_mask_flat)  # [B, D]
        
        # 背景原型：使用背景区域 (1 - obj_mask)
        ego_obj_mask_flat = ego_obj_mask.reshape(B, -1)  # [B, N]
        ego_bg_mask = 1.0 - ego_obj_mask_flat
        ego_bg_prototype = masked_avg_pooling(ego_features[:, 1:], ego_bg_mask)  # [B, D]
        
        # 2. 构建第三人称图像的原型
        # 正原型：使用物体掩码进行掩码平均池化
        exo_obj_mask_flat = exo_obj_mask_full.reshape(B_exo, -1)  # [B*num_exo, N]
        exo_pos_prototype = masked_avg_pooling(exo_features[:, 1:], exo_obj_mask_flat)  # [B*num_exo, D]
        
        # 背景原型：使用背景区域
        exo_bg_mask = 1.0 - exo_obj_mask_flat
        exo_bg_prototype = masked_avg_pooling(exo_features[:, 1:], exo_bg_mask)  # [B*num_exo, D]
        
        # 3. 构建批次内的正负原型集合
        # 正原型集合：所有样本的exo正原型
        positives = [exo_pos_prototype]  # [B*num_exo, D]
        
        # 负原型集合：背景原型
        negatives = [
            ego_bg_prototype.unsqueeze(1).expand(-1, num_exo, -1).reshape(B*num_exo, D),  # [B*num_exo, D]
            exo_bg_prototype  # [B*num_exo, D]
        ]
        
        # 扩展锚点以匹配批次大小
        anchor_expanded = anchor_prototype.unsqueeze(1).expand(-1, num_exo, -1).reshape(B*num_exo, D)
        
        # 4. 计算选择性原型对比损失
        proto_loss = selective_prototype_contrast_loss(
            anchor_expanded, positives, negatives, temperature=0.07
        )
        
        return proto_loss
    