# from CLIP https://github.com/openai/CLIP/blob/main/clip/model.py
from collections import OrderedDict
import torch
import torch.nn.functional as F
from torch import nn


class LayerNorm(nn.LayerNorm):
    """Subclass torch's LayerNorm to handle fp16."""

    def forward(self, x: torch.Tensor):
        orig_type = x.dtype
        ret = super().forward(x.type(torch.float32))
        return ret.type(orig_type)


class QuickGELU(nn.Module):
    def forward(self, x: torch.Tensor):
        return x * torch.sigmoid(1.702 * x)


class ResidualAttentionBlock(nn.Module):
    def __init__(self, d_model: int, n_head: int, attn_mask: torch.Tensor = None):
        super().__init__()

        self.attn = nn.MultiheadAttention(d_model, n_head)
        self.ln_1 = LayerNorm(d_model)
        self.mlp = nn.Sequential(OrderedDict([
            ("c_fc", nn.Linear(d_model, d_model * 4)),
            ("gelu", QuickGELU()),
            ("c_proj", nn.Linear(d_model * 4, d_model))
        ]))
        self.ln_2 = LayerNorm(d_model)
        self.attn_mask = attn_mask

    def attention(self, x: torch.Tensor):
        self.attn_mask = self.attn_mask.to(dtype=x.dtype, device=x.device) if self.attn_mask is not None else None
        return self.attn(x, x, x, need_weights=False, attn_mask=self.attn_mask)[0]

    def forward(self, x: torch.Tensor):
        x = x + self.attention(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x


class Transformer(nn.Module):
    def __init__(self, width: int, layers: int, heads: int, attn_mask: torch.Tensor = None):
        super().__init__()
        self.width = width
        self.layers = layers
        self.resblocks = nn.Sequential(*[ResidualAttentionBlock(width, heads, attn_mask) for _ in range(layers)])

    def forward(self, x: torch.Tensor):
        return self.resblocks(x)


class VisionTransformer(nn.Module):
    def __init__(self, input_resolution: int=224, patch_size: int=16, width: int=768, layers: int=12, heads: int=12, output_dim: int=512):
        super().__init__()
        self.input_resolution = input_resolution
        self.output_dim = output_dim
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=width, kernel_size=patch_size, stride=patch_size, bias=False)

        scale = width ** -0.5
        self.class_embedding = nn.Parameter(scale * torch.randn(width))
        self.positional_embedding = nn.Parameter(scale * torch.randn((input_resolution // patch_size) ** 2 + 1, width))
        self.ln_pre = LayerNorm(width)

        self.transformer = Transformer(width, layers, heads)

        self.ln_post = LayerNorm(width)
        self.proj = nn.Parameter(scale * torch.randn(width, output_dim))
        
        self.num_patches = int((input_resolution // patch_size)**2)

    def forward(self, x: torch.Tensor):
        x = self.conv1(x)  # shape = [*, width, grid, grid]
        x = x.reshape(x.shape[0], x.shape[1], -1)  # shape = [*, width, grid ** 2]
        x = x.permute(0, 2, 1)  # shape = [*, grid ** 2, width]
        
        x = torch.cat([self.class_embedding.to(x.dtype) + torch.zeros(x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device), x], dim=1)  # shape = [*, grid ** 2 + 1, width]
        x = x + self.positional_embedding.to(x.dtype)
        x = self.ln_pre(x)

        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD

        x = self.ln_post(x)
        xx = x @ self.proj

        return x, xx


class DINOv2VisionTransformer(nn.Module):
    """DINOv2 encoder wrapper that exposes multi-layer token features.

    The module keeps the same forward interface as the CLIP VisionTransformer:
    it returns a token sequence with CLS + patch tokens, and a projected version
    of the same sequence. The internal backbone is loaded from torch hub by
    default so the code stays dependency-light.
    """

    def __init__(self,
                 input_resolution: int = 224,
                 output_dim: int = 512,
                 model_name: str = "dinov2_vits14",
                 num_layers: int = 4,
                 freeze_backbone: bool = True,
                 pretrained: bool = True,
                 checkpoint_path: str = None,
                 fuse: str = "mean"):
        super().__init__()
        self.input_resolution = input_resolution
        self.output_dim = output_dim
        self.model_name = model_name
        self.num_layers = num_layers
        self.freeze_backbone = freeze_backbone
        self.fuse = fuse

        self.backbone = self._build_backbone(model_name, pretrained, checkpoint_path)
        self.embed_dim = getattr(self.backbone, "embed_dim", None)
        if self.embed_dim is None:
            self.embed_dim = getattr(self.backbone, "num_features", None)
        if self.embed_dim is None:
            raise ValueError(f"Unable to infer DINOv2 embed dim from backbone: {model_name}")

        patch_embed = getattr(self.backbone, "patch_embed", None)
        patch_size = getattr(patch_embed, "patch_size", 14) if patch_embed is not None else 14
        if isinstance(patch_size, tuple):
            patch_size = patch_size[0]
        self.patch_size = patch_size
        self.num_patches = int((input_resolution // self.patch_size) ** 2)

        self.ln_post = LayerNorm(self.embed_dim)
        self.proj = nn.Linear(self.embed_dim, output_dim)

        if self.fuse == "weighted":
            self.layer_weights = nn.Parameter(torch.ones(num_layers) / num_layers)
        else:
            self.layer_weights = None

        self.last_multilayer_tokens = None

        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False
            self.backbone.eval()

    def _build_backbone(self, model_name: str, pretrained: bool, checkpoint_path: str = None):
        try:
            backbone = torch.hub.load("facebookresearch/dinov2", model_name, pretrained=pretrained)
        except Exception as exc:
            raise RuntimeError(
                "Failed to load DINOv2 from torch.hub. "
                "Install the dinov2 package or ensure torch hub can access the model repo."
            ) from exc

        if checkpoint_path:
            state_dict = torch.load(checkpoint_path, map_location="cpu")
            if isinstance(state_dict, dict) and "state_dict" in state_dict:
                state_dict = state_dict["state_dict"]
            backbone.load_state_dict(state_dict, strict=False)

        return backbone

    def _fuse_tokens(self, tokens):
        if self.fuse == "weighted" and self.layer_weights is not None:
            weights = torch.softmax(self.layer_weights, dim=0)
            cls_token = torch.zeros_like(tokens[0][:, :1])
            patch_tokens = torch.zeros_like(tokens[0][:, 1:])
            for weight, token in zip(weights, tokens):
                cls_token = cls_token + weight * token[:, :1]
                patch_tokens = patch_tokens + weight * token[:, 1:]
            return cls_token, patch_tokens

        cls_token = torch.stack([token[:, :1] for token in tokens], dim=0).mean(dim=0)
        patch_tokens = torch.stack([token[:, 1:] for token in tokens], dim=0).mean(dim=0)
        return cls_token, patch_tokens

    def forward(self, x: torch.Tensor):
        if x.shape[-2:] != (self.input_resolution, self.input_resolution):
            x = F.interpolate(
                x,
                size=(self.input_resolution, self.input_resolution),
                mode="bilinear",
                align_corners=False,
            )

        if not hasattr(self.backbone, "get_intermediate_layers"):
            raise RuntimeError(
                f"Backbone {self.model_name} does not expose get_intermediate_layers, "
                "so multi-layer features cannot be extracted."
            )

        tokens = self.backbone.get_intermediate_layers(x, n=self.num_layers)
        if not isinstance(tokens, (list, tuple)):
            tokens = [tokens]

        normalized_tokens = []
        for token in tokens:
            if isinstance(token, tuple):
                token = token[0]
            if token.dim() != 3:
                raise RuntimeError(f"Unexpected DINOv2 token shape: {token.shape}")
            if token.shape[1] == self.num_patches:
                cls_token = token.mean(dim=1, keepdim=True)
                patch_tokens = token
                token = torch.cat([cls_token, patch_tokens], dim=1)
            normalized_tokens.append(token)

        cls_token, patch_tokens = self._fuse_tokens(normalized_tokens)
        x = torch.cat([cls_token, patch_tokens], dim=1)
        x = self.ln_post(x)
        xx = self.proj(x)

        self.last_multilayer_tokens = torch.stack(normalized_tokens, dim=0)
        return x, xx
