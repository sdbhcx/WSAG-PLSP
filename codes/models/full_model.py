import torch
import torch.nn as nn
import torch.nn.functional as F

from models.encoder_clip import VisionTransformer, DINOv2VisionTransformer
from models.decoder_affordance import Affordance_Decoder
from models.SAM_decoder import SAM_Decoder_Simple
from models.SAM_FPN import SAM_Decoder_FPN


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
    
    # Normalize features for stability and cosine similarity
    anchor = F.normalize(anchor, p=2, dim=1)
    positives = [F.normalize(p, p=2, dim=1) for p in positives]
    negatives = [F.normalize(n, p=2, dim=1) for n in negatives]
    
    # Combine all prototypes
    # positives and negatives are lists of [B, D] tensors
    # Convert them to [B, 1, D] for concatenation
    positives_expanded = [p.unsqueeze(1) for p in positives]
    negatives_expanded = [n.unsqueeze(1) for n in negatives]
    
    # All prototypes: Anchor (0) + Positives (1..P) + Negatives (P+1..end)
    all_prototypes = torch.cat([anchor.unsqueeze(1)] + positives_expanded + negatives_expanded, dim=1)  # [B, 1+P+N, D]
    
    # Compute similarity scores
    anchor_expanded = anchor.unsqueeze(1)  # [B, 1, D]
    
    # [B, 1, D] @ [B, D, 1+P+N] -> [B, 1, 1+P+N]
    logits = torch.matmul(anchor_expanded, all_prototypes.transpose(1, 2)).squeeze(1)  # [B, 1+P+N]
    
    # Scale by temperature
    logits = logits / temperature
    
    # Identify indices
    num_positives = len(positives)
    
    # Positives are at indices 1 to 1+num_positives (index 0 is anchor itself)
    pos_logits = logits[:, 1:1+num_positives]
    
    # Use LogSumExp for numerical stability
    # Loss = -log( sum(exp(pos)) / sum(exp(all)) )
    #      = -( logsumexp(pos) - logsumexp(all) )
    
    log_prob_pos = torch.logsumexp(pos_logits, dim=1)
    log_prob_all = torch.logsumexp(logits, dim=1)
    
    loss = -(log_prob_pos - log_prob_all).mean()
    
    return loss


class UnifiedCrossModalFusion(nn.Module):
    def __init__(self, dim, num_heads=8, dropout=0.1):
        super().__init__()
        self.text_to_image = nn.MultiheadAttention(
            embed_dim=dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.image_to_text = nn.MultiheadAttention(
            embed_dim=dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.norm_text_1 = nn.LayerNorm(dim)
        self.norm_img_1 = nn.LayerNorm(dim)
        self.norm_text_2 = nn.LayerNorm(dim)
        self.norm_img_2 = nn.LayerNorm(dim)
        self.text_mlp = nn.Sequential(
            nn.Linear(dim, dim * 2),
            nn.GELU(),
            nn.Linear(dim * 2, dim),
        )
        self.image_mlp = nn.Sequential(
            nn.Linear(dim, dim * 2),
            nn.GELU(),
            nn.Linear(dim * 2, dim),
        )

    def forward(self, image_tokens, text_token):
        # Text branch attends to image features (text <- image).
        text_delta, _ = self.text_to_image(query=text_token, key=image_tokens, value=image_tokens, need_weights=False)
        text_token = self.norm_text_1(text_token + text_delta)

        # Image branch attends to text token (image <- text).
        img_delta, _ = self.image_to_text(query=image_tokens, key=text_token, value=text_token, need_weights=False)
        image_tokens = self.norm_img_1(image_tokens + img_delta)

        text_token = self.norm_text_2(text_token + self.text_mlp(text_token))
        image_tokens = self.norm_img_2(image_tokens + self.image_mlp(image_tokens))
        return text_token, image_tokens


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
                 margin=0.5,
                 use_prompt_learning=False,
                 prompt_length=8,
                 prompt_dropout=0.1,
                 prompt_num_heads=8,
                   use_unified_fusion=True
                   ):
        super().__init__()

        self.margin = margin
        self.use_unified_fusion = use_unified_fusion

        encoder_params = dict(encoder_params)
        if str(encoder_type).lower() in ("dino", "dinov2"):
            dino_keys = {
                "model_name",
                "num_layers",
                "freeze_backbone",
                "pretrained",
                "checkpoint_path",
                "fuse",
            }
            encoder_params = {k: v for k, v in encoder_params.items() if k in dino_keys}
            self.encoder = DINOv2VisionTransformer(
                input_resolution=img_size,
                output_dim=decoder_embed_dim,
                **encoder_params,
            )
        else:
            self.encoder = VisionTransformer(
                input_resolution=img_size, patch_size=patch_size, **encoder_params)

        # Keep legacy fusion path for ablation and backward compatibility.
        self.verb_fuser = Affordance_Decoder(
            num_patches=self.encoder.num_patches,
            decoder_embed_dim=decoder_embed_dim, regresser_depth=aff_decoder_depth,
            num_heads=decoder_num_heads,
            mlp_ratio=mlp_ratio, qkv_bias=True, qk_scale=None,
            drop_rate=drop_rate, attn_drop_rate=attn_drop_rate, drop_path_rate=drop_path_rate,
            norm_layer=norm_layer, init_values=decoder_layer_scale_init_value, init_std=init_std
        )

        if self.use_unified_fusion:
            self.cross_modal_fuser = UnifiedCrossModalFusion(
                dim=decoder_embed_dim,
                num_heads=decoder_num_heads,
                dropout=attn_drop_rate,
            )

        if pred_model_type == "SAM":
            self.pred_decoder = SAM_Decoder_Simple(
                transformer_dim=decoder_embed_dim,
                activation=nn.GELU,
                **pred_decoder_args,
            )
        elif pred_model_type == "SAM_FPN":
            self.pred_decoder = SAM_Decoder_FPN(
                transformer_dim=decoder_embed_dim,
                activation=nn.GELU,
                **pred_decoder_args,
            )
        else:
            self.pred_decoder = SAM_Decoder_FPN(
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

        self.use_prompt_learning = use_prompt_learning
        if self.use_prompt_learning:
            # Learnable context tokens [V1]...[Vp] that are prepended before the affordance token.
            self.prompt_context = nn.Parameter(
                torch.randn(prompt_length, decoder_embed_dim) * init_std
            )
            self.prompt_attn = nn.MultiheadAttention(
                embed_dim=decoder_embed_dim,
                num_heads=prompt_num_heads,
                dropout=prompt_dropout,
                batch_first=True,
            )
            self.prompt_norm = nn.LayerNorm(decoder_embed_dim)
            self.prompt_dropout = nn.Dropout(prompt_dropout)
        
        # self.proto_projector = nn.Sequential(
        #     nn.Linear(512, 512),
        #     nn.GELU(),
        #     nn.Linear(512, 512)
        # )

    def build_prompted_text_feat(self, text_feat):
        if not self.use_prompt_learning:
            return text_feat

        B, D = text_feat.shape
        context = self.prompt_context.unsqueeze(0).expand(B, -1, -1)
        affordance_token = text_feat.unsqueeze(1)
        prompt_sequence = torch.cat([context, affordance_token], dim=1)
        prompt_update, _ = self.prompt_attn(prompt_sequence, prompt_sequence, prompt_sequence, need_weights=False)
        prompt_sequence = self.prompt_norm(prompt_sequence + self.prompt_dropout(prompt_update))
        return prompt_sequence[:, -1, :]
        
        
    def forward(self, imgs, text_feat, exo=None, exo_obj_mask=None, num_exo=1, 
                ego_part_mask=None, ego_obj_mask=None, exo_obj_mask_full=None):
        # 1. 提取第一视角图像特征
        _, x = self.encoder(imgs)

        # proj_x = self.proto_projector(x)
        # 2. 处理动作语义特征
        prompted_text_feat = self.build_prompted_text_feat(text_feat.float())
        v = prompted_text_feat.unsqueeze(1)
        # 3. 预测物体特征
        pred_noun = self.noun_transform(x[:, 0:1, ].detach()) 
        # 4. 融合物体特征和动作语义特征
        pred_part = self.reason(torch.cat([pred_noun, v], dim=2))
        if self.use_unified_fusion:
            # 在统一双向跨模态模块中同时执行 text<->image 融合
            aff_token, fused_tokens = self.cross_modal_fuser(x, pred_part + v)

            # 将融合后的 token 与视觉特征直接送入 SAM 头
            pred_heatmap = self.pred_decoder(fused_tokens, aff_token, skip_transformer=True)
        else:
            # 兼容旧路径：先用 verb_fuser 融合，再走 SAM 内部 TwoWay 交互
            aff_token, _, _ = self.verb_fuser(x, pred_part + v)
            pred_heatmap = self.pred_decoder(x, aff_token)

        # 构建原型用于选择性原型对比损失
        proto_loss = None
        if exo is not None:
            # with torch.no_grad():
            _, exo = self.encoder(exo)
            # proj_exo = self.proto_projector(exo)

            # Ensure exo_obj_mask matches the spatial token count of exo
            mask = exo_obj_mask
            if mask is None:
                mask = torch.ones(exo.shape[0], exo.shape[1] - 1, device=exo.device, dtype=exo.dtype)
            else:
                # squeeze trailing singleton if present
                if mask.dim() == 3 and mask.shape[-1] == 1:
                    mask = mask.squeeze(-1)

                token_N = exo.shape[1] - 1
                mask_N = mask.shape[1]
                if mask_N != token_N:
                    # try to interpret masks as square grids and resize to token grid
                    s_mask = int(round(mask_N ** 0.5))
                    s_feat = int(round(token_N ** 0.5))
                    if s_mask * s_mask == mask_N and s_feat * s_feat == token_N:
                        mask = mask.reshape(mask.shape[0], 1, s_mask, s_mask)
                        mask = F.interpolate(mask, size=(s_feat, s_feat), mode='nearest')
                        mask = mask.reshape(mask.shape[0], -1)
                    else:
                        # fallback: trim or repeat to fit
                        if mask_N < token_N:
                            repeat = int(token_N // mask_N)
                            mask = mask.repeat_interleave(repeat, dim=1)[:, :token_N]
                        else:
                            mask = mask[:, :token_N]

            exo_obj_mask = mask.to(exo.dtype)
            exo_token = (exo[:, 1:] * exo_obj_mask.unsqueeze(-1)).sum(dim=1)
            D = aff_token.shape[-1]
            aff_token_expand = aff_token.expand(-1, num_exo, -1).reshape(-1, D)
            sim_loss = torch.max(
                1 - F.cosine_similarity(aff_token_expand, exo_token.detach(), dim=1) - self.margin, 
                torch.zeros(len(exo_token)).to(x.device))
            
            # 构建选择性原型对比损失
            proto_loss = None
            if ego_part_mask is not None and exo_obj_mask_full is not None:
                proto_loss = self.compute_prototype_contrast_loss(
                    x, exo, ego_part_mask, ego_obj_mask, 
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
        # 这里的 ego_obj_mask 实际上应该是 whole_object_mask
        ego_whole_obj_mask_flat = ego_obj_mask.reshape(B, -1)  # [B, N]
        ego_bg_mask = 1.0 - ego_whole_obj_mask_flat
        ego_bg_prototype = masked_avg_pooling(ego_features[:, 1:], ego_bg_mask)  # [B, D]
        
        # 难负样本原型：物体其余部分 (Whole Object - Part)
        # 注意：需要确保相减后不为负数，且有足够的像素
        ego_body_mask = (ego_whole_obj_mask_flat - ego_part_mask_flat).clamp(min=0)
        # 为防止 ego_body_mask 全为0（即只有部件没有其他部分），可以加一个极小的数值或者只在mask非空时使用
        # 这里 masked_avg_pooling 内部有 clamp(min=1e-6) 防止除零
        ego_body_prototype = masked_avg_pooling(ego_features[:, 1:], ego_body_mask)
        
        # 2. 构建第三人称图像的原型
        # 正原型：使用物体掩码进行掩码平均池化
        exo_obj_mask_flat = exo_obj_mask_full.reshape(B_exo, -1)  # [B*num_exo, N]
        exo_pos_prototype = masked_avg_pooling(exo_features[:, 1:], exo_obj_mask_flat)  # [B*num_exo, D]
        
        # 背景原型：使用背景区域
        # exo_bg_mask = 1.0 - exo_obj_mask_flat
        # exo_bg_prototype = masked_avg_pooling(exo_features[:, 1:], exo_bg_mask)  # [B*num_exo, D]
        
        # 3. 构建批次内的正负原型集合
        # 正原型集合：所有样本的exo正原型
        positives = [exo_pos_prototype]  # [B*num_exo, D]
        
        # 负原型集合：背景原型 + 难负样本原型
        negatives = [
            ego_bg_prototype.unsqueeze(1).expand(-1, num_exo, -1).reshape(B*num_exo, D),  # [B*num_exo, D] (Easy Negative: Background)
            # exo_bg_prototype,  # [B*num_exo, D] (第三人称背景也可以作为负样本，可选)
            ego_body_prototype.unsqueeze(1).expand(-1, num_exo, -1).reshape(B*num_exo, D) # [B*num_exo, D] (Hard Negative: Object Body)
        ]
        
        # 扩展锚点以匹配批次大小
        anchor_expanded = anchor_prototype.unsqueeze(1).expand(-1, num_exo, -1).reshape(B*num_exo, D)
        
        # 4. 计算选择性原型对比损失
        proto_loss = selective_prototype_contrast_loss(
            anchor_expanded, positives, negatives, temperature=0.07
        )
        
        return proto_loss
    