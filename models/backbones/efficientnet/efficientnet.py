"""
EfficientNet Backbone: Efficient scaling of CNN architectures.

Paper: EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks
Link: https://arxiv.org/abs/1905.11946

Key idea: Scale depth, width, and resolution together using compound scaling.

Compound Scaling:
    depth   = α^φ    (number of layers)
    width   = β^φ    (channels per layer)
    resolution = γ^φ (input resolution)
    where φ is a user-defined scaling coefficient
          α, β, γ are found via neural architecture search (NAS)

Benefits:
    • Better accuracy/efficiency trade-off
    • Mobile-friendly (small models: EfficientNetB0-B2)
    • Large models available (B5-B7) for high accuracy
    • Scales from 5M to 66M parameters

EfficientNet variants:
    B0: 224×224, 5.3M params   (baseline)
    B1: 240×240, 7.8M params
    B2: 260×260, 9.2M params
    B3: 300×300, 12M params
    B4: 380×380, 19M params    (standard for face recognition)
    B5: 456×456, 30M params
    B6: 528×528, 43M params
    B7: 600×600, 66M params
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple, Optional, Dict
import math


class MBConvBlock(nn.Module):
    """
    Mobile Inverted Bottleneck Conv block (MBConv).
    
    Architecture:
        1. Pointwise (1×1): expand channels
        2. Depthwise (3×3 or 5×5): apply per-channel convolution
        3. Squeeze-and-Excitation: channel attention
        4. Pointwise (1×1): project back to output channels
        5. Skip connection (if input == output channels)
    
    This inverted structure is efficient: expand → process → contract
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        expand_ratio: int = 1,
        drop_rate: float = 0.0,
    ):
        super().__init__()
        self.stride = stride
        self.drop_rate = drop_rate

        # Expansion phase
        hidden_dim = in_channels * expand_ratio
        self.expand_conv = nn.Conv2d(in_channels, hidden_dim, kernel_size=1,
                                    bias=False)
        self.expand_bn = nn.BatchNorm2d(hidden_dim)
        self.expand_silu = nn.SiLU()

        # Depthwise phase
        self.dw_conv = nn.Conv2d(hidden_dim, hidden_dim, kernel_size=kernel_size,
                                stride=stride, padding=kernel_size//2,
                                groups=hidden_dim, bias=False)
        self.dw_bn = nn.BatchNorm2d(hidden_dim)
        self.dw_silu = nn.SiLU()

        # Squeeze-and-Excitation
        self.se_reduce = nn.Conv2d(hidden_dim, max(1, hidden_dim // 4),
                                  kernel_size=1)
        self.se_expand = nn.Conv2d(max(1, hidden_dim // 4), hidden_dim,
                                  kernel_size=1)

        # Projection phase
        self.proj_conv = nn.Conv2d(hidden_dim, out_channels, kernel_size=1,
                                  bias=False)
        self.proj_bn = nn.BatchNorm2d(out_channels)

        self.skip_connection = in_channels == out_channels and stride == 1

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        input_x = x

        # Expansion
        x = self.expand_conv(x)
        x = self.expand_bn(x)
        x = self.expand_silu(x)

        # Depthwise
        x = self.dw_conv(x)
        x = self.dw_bn(x)
        x = self.dw_silu(x)

        # Squeeze-and-Excitation
        se = F.adaptive_avg_pool2d(x, 1)
        se = self.se_reduce(se)
        se = F.silu(se)
        se = self.se_expand(se)
        se = torch.sigmoid(se)
        x = x * se

        # Projection
        x = self.proj_conv(x)
        x = self.proj_bn(x)

        # Skip connection
        if self.skip_connection:
            if self.drop_rate > 0:
                x = F.dropout(x, p=self.drop_rate, training=self.training)
            x = x + input_x

        return x


class EfficientNet(nn.Module):
    """
    EfficientNet backbone for face recognition.

    Supports flexible scaling via width and depth multipliers.
    """

    def __init__(
        self,
        width_multiplier: float = 1.0,
        depth_multiplier: float = 1.0,
        input_size: int = 224,
        embedding_dim: int = 512,
        drop_rate: float = 0.0,
    ):
        super().__init__()
        self.embedding_dim = embedding_dim

        # MBConv config: (in_ch, out_ch, kernel, stride, expand_ratio, repeats)
        mbconv_configs = [
            (32, 16, 3, 1, 1, 1),
            (16, 24, 3, 2, 6, 2),
            (24, 40, 5, 2, 6, 2),
            (40, 80, 3, 2, 6, 3),
            (80, 112, 5, 1, 6, 3),
            (112, 192, 5, 2, 6, 4),
            (192, 320, 3, 1, 6, 1),
        ]

        # Apply width multiplier
        def scale_channels(ch: int) -> int:
            return max(8, int(ch * width_multiplier))

        # Initial conv
        self.conv1 = nn.Conv2d(3, scale_channels(32), kernel_size=3, stride=2,
                              padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(scale_channels(32))
        self.silu = nn.SiLU()

        # MBConv blocks
        self.blocks = nn.ModuleList()
        for in_ch, out_ch, kernel, stride, expand_r, repeats in mbconv_configs:
            repeats = max(1, int(repeats * depth_multiplier))
            in_ch = scale_channels(in_ch)
            out_ch = scale_channels(out_ch)

            for i in range(repeats):
                stride_i = stride if i == 0 else 1
                self.blocks.append(
                    MBConvBlock(in_ch, out_ch, kernel, stride_i, expand_r,
                               drop_rate)
                )
                in_ch = out_ch

        # Final conv
        self.conv_head = nn.Conv2d(scale_channels(320), scale_channels(1280),
                                  kernel_size=1, bias=False)
        self.bn_head = nn.BatchNorm2d(scale_channels(1280))

        # Global average pooling + embedding
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.feat_bn = nn.BatchNorm1d(scale_channels(1280))
        self.feat_bn.bias.requires_grad_(False)

        self.embedding_proj = nn.Linear(scale_channels(1280), embedding_dim,
                                       bias=False)
        self.embedding_bn = nn.BatchNorm1d(embedding_dim)
        self.embedding_bn.bias.requires_grad_(False)

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Args:
            x: [B, 3, H, W]

        Returns:
            dict with embedding and logits
        """
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.silu(x)

        for block in self.blocks:
            x = block(x)

        x = self.conv_head(x)
        x = self.bn_head(x)
        x = self.silu(x)

        x = self.avgpool(x)
        feat_raw = torch.flatten(x, 1)

        feat_raw = self.feat_bn(feat_raw)
        logit = self.embedding_proj(feat_raw)
        embedding = self.embedding_bn(logit)
        embedding = F.normalize(embedding, p=2, dim=1)

        return {
            "embedding": embedding,
            "logit"    : logit,
            "feat_raw" : feat_raw,
        }


# ── Pre-configured variants ──────────────────────────────────────────────────

class EfficientNetB0(EfficientNet):
    """EfficientNetB0: baseline model (224×224, 5.3M params)"""
    def __init__(self, embedding_dim: int = 512):
        super().__init__(width_multiplier=1.0, depth_multiplier=1.0,
                        input_size=224, embedding_dim=embedding_dim)


class EfficientNetB4(EfficientNet):
    """EfficientNetB4: standard model (380×380, 19M params)"""
    def __init__(self, embedding_dim: int = 512):
        super().__init__(width_multiplier=1.4, depth_multiplier=1.8,
                        input_size=380, embedding_dim=embedding_dim)


def get_efficientnet(
    model_name: str = "b4",
    embedding_dim: int = 512,
    weights_path: Optional[str] = None,
) -> EfficientNet:
    """
    Get EfficientNet model by name.

    Args:
        model_name   : "b0", "b4" etc
        embedding_dim: output embedding dimension
        weights_path : path to pretrained weights

    Returns:
        EfficientNet backbone
    """
    if model_name == "b0":
        model = EfficientNetB0(embedding_dim)
    elif model_name == "b4":
        model = EfficientNetB4(embedding_dim)
    else:
        raise ValueError(f"Unknown EfficientNet model: {model_name}")

    if weights_path is not None:
        state = torch.load(weights_path, map_location='cpu')
        model.load_state_dict(state)
        print(f"[EfficientNet] Loaded weights from: {weights_path}")

    return model
