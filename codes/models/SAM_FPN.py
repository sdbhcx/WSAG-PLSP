import torch
import torch.nn as nn
import torch.nn.functional as F
from models.SAM_decoder import SAM_Decoder_Simple, LayerNorm2d, MLP

class FPNUpscaler(nn.Module):
    def __init__(self, in_channels, out_channels, activation=nn.GELU):
        super().__init__()
        # Generate multi-scale features from 1/16 scale (e.g., 14x14 for 224x224 input)
        # scale 4: 1/4 (e.g., 56x56)
        self.up4 = nn.Sequential(
            nn.ConvTranspose2d(in_channels, in_channels//2, kernel_size=2, stride=2),
            LayerNorm2d(in_channels//2),
            activation(),
            nn.ConvTranspose2d(in_channels//2, in_channels//4, kernel_size=2, stride=2),
            LayerNorm2d(in_channels//4),
            activation()
        )
        
        # scale 3: 1/8 (e.g., 28x28)
        self.up3 = nn.Sequential(
            nn.ConvTranspose2d(in_channels, in_channels//4, kernel_size=2, stride=2),
            LayerNorm2d(in_channels//4),
            activation()
        )
        
        # scale 2: 1/16 (e.g., 14x14)
        self.up2 = nn.Sequential(
            nn.Conv2d(in_channels, in_channels//4, kernel_size=1),
            LayerNorm2d(in_channels//4),
            activation()
        )
        
        # scale 1: 1/32 (e.g., 7x7)
        self.down1 = nn.Sequential(
            nn.Conv2d(in_channels, in_channels//4, kernel_size=2, stride=2),
            LayerNorm2d(in_channels//4),
            activation()
        )
        
        self.lateral4 = nn.Conv2d(in_channels//4, out_channels, kernel_size=1)
        self.lateral3 = nn.Conv2d(in_channels//4, out_channels, kernel_size=1)
        self.lateral2 = nn.Conv2d(in_channels//4, out_channels, kernel_size=1)
        self.lateral1 = nn.Conv2d(in_channels//4, out_channels, kernel_size=1)
        
        self.smooth4 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.smooth3 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.smooth2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.smooth1 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)

    def forward(self, x):
        f4 = self.up4(x)
        f3 = self.up3(x)
        f2 = self.up2(x)
        f1 = self.down1(x)
        
        p1 = self.lateral1(f1)
        p2 = self.lateral2(f2) + F.interpolate(p1, size=f2.shape[-2:], mode="bilinear", align_corners=False)
        p3 = self.lateral3(f3) + F.interpolate(p2, size=f3.shape[-2:], mode="bilinear", align_corners=False)
        p4 = self.lateral4(f4) + F.interpolate(p3, size=f4.shape[-2:], mode="bilinear", align_corners=False)
        
        p4 = self.smooth4(p4)
        return p4


class SAM_Decoder_FPN(SAM_Decoder_Simple):
    def __init__(self, transformer_dim: int, mlp_dim: int, depth: int, activation = nn.GELU, num_heads: int = 8, use_up: int = 2, use_additional_token: bool = False, conv_first: bool = True) -> None:
        super().__init__(transformer_dim, mlp_dim, depth, activation, num_heads, use_up, use_additional_token, conv_first)
        
        # Override the upscaling with FPN
        fpn_dim = transformer_dim // 8
        self.output_upscaling = FPNUpscaler(transformer_dim, fpn_dim, activation=activation)
        self.output_hypernetworks_mlp = MLP(transformer_dim, transformer_dim, fpn_dim, 3)

