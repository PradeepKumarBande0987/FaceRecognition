"""
GhostFaceNet: Lightweight face recognition backbone using Ghost modules.

Paper: GhostFaceNets: Lightweight Face Recognition Model via Ghost Module
Link: https://arxiv.org/abs/2106.08254

Key idea: Ghost Module generates feature maps with minimal computation.

Ghost Module:
    Instead of generating all feature maps with expensive convolution:
    1. Use cheap operations (linear transforms) on a small set of features
    2. Generate "ghost" features from these intrinsic features
    3. Concatenate intrinsic + ghost features

    Result: Similar output quality with 50% fewer FLOPs

Benefits:
    • 50% fewer FLOPs than standard convolutions
    • Ultra-lightweight: ~2M parameters
    • Mobile deployment-friendly
    • Only ~2% accuracy drop vs ResNet50 on LFW

Use case: Ultra-lightweight face recognition for edge devices, mobile phones
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional, Dict


class GhostModule(nn.Module):
    """
    Ghost module: Generate feature maps with minimal computation.

    Architecture:
        Input → Conv (primary) → ReLU
               ↓
               → Linear transform (ghost) → ReLU
               ↓
        Concat (primary + ghost)

    This produces more features than the primary convolution alone,
    with much lower computational cost.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 1,
        ratio: int = 2,  # ghost features = primary * ratio
        dw_size: int = 3,
        stride: int = 1,
        relu: bool = True,
    ):
        super().__init__()
        self.out_channels = out_channels
        init_channels = out_channels // ratio  # primary channels

        # Primary convolution
        self.primary_conv = nn.Sequential(
            nn.Conv2d(in_channels, init_channels, kernel_size, stride,
                     kernel_size // 2, bias=False),
            nn.BatchNorm2d(init_channels),
            nn.ReLU(inplace=True) if relu else nn.Sequential(),
        )

        # Ghost convolution (depthwise)
        if init_channels * (ratio - 1) > 0:
            self.ghost_conv = nn.Sequential(
                nn.Conv2d(init_channels, init_channels * (ratio - 1),
                         dw_size, 1, dw_size // 2, groups=init_channels,
                         bias=False),
                nn.BatchNorm2d(init_channels * (ratio - 1)),
                nn.ReLU(inplace=True) if relu else nn.Sequential(),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        primary = self.primary_conv(x)
        ghost = self.ghost_conv(primary)
        return torch.cat([primary, ghost], dim=1)


class GhostBottleneck(nn.Module):
    """
    Ghost Bottleneck: Inverted residual block with ghost modules.

    Structure:
        GhostModule (expand) → DW Conv (depthwise)
        → GhostModule (contract) → + (skip connection)
    """

    def __init__(
        self,
        in_channels: int,
        hidden_dim: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
    ):
        super().__init__()
        self.stride = stride

        # Expansion
        self.ghost1 = GhostModule(in_channels, hidden_dim, 1, 2)

        # Depthwise
        if stride > 1:
            self.dw = nn.Sequential(
                nn.Conv2d(hidden_dim, hidden_dim, kernel_size, stride,
                         (kernel_size - 1) // 2, groups=hidden_dim, bias=False),
                nn.BatchNorm2d(hidden_dim),
            )
        else:
            self.dw = nn.Sequential()

        # Contraction
        self.ghost2 = GhostModule(hidden_dim, out_channels, 1, 2, relu=False)

        # Skip connection
        self.skip = (stride == 1 and in_channels == out_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_skip = x

        x = self.ghost1(x)
        x = self.dw(x)
        x = self.ghost2(x)

        if self.skip:
            x = x + x_skip

        return x


class GhostFaceNet(nn.Module):
    """
    GhostFaceNet backbone: Ultra-lightweight face recognition.

    Architecture:
        Conv(3×3) → BatchNorm → ReLU
        → GhostBottleneck blocks (progressive downsampling)
        → AdaptiveAvgPool → Flatten
        → Embedding projection → BatchNorm → L2 normalize

    Characteristics:
        • ~2M parameters (vs 25M for ResNet50)
        • ~500M FLOPs (vs 4B for ResNet50)
        • LFW accuracy: ~98% (vs 99%+ for ResNet50)
        • Mobile-compatible
    """

    def __init__(
        self,
        num_classes: int = 10000,
        embedding_dim: int = 512,
    ):
        super().__init__()
        self.embedding_dim = embedding_dim

        # Initial convolution
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
        )

        # Ghost bottleneck blocks
        # (in_ch, hidden_ch, out_ch, kernel, stride)
        ghost_config = [
            (16, 16, 16, 3, 1),
            (16, 48, 24, 3, 2),
            (24, 72, 24, 3, 1),
            (24, 72, 40, 5, 2),
            (40, 120, 40, 5, 1),
            (40, 240, 80, 3, 2),
            (80, 200, 80, 3, 1),
            (80, 184, 80, 3, 1),
            (80, 184, 80, 3, 1),
            (80, 480, 112, 3, 1),
            (112, 672, 112, 3, 1),
            (112, 672, 160, 5, 2),
            (160, 960, 160, 5, 1),
            (160, 960, 160, 5, 1),
        ]

        self.blocks = nn.ModuleList()
        for in_ch, hidden_ch, out_ch, kernel, stride in ghost_config:
            self.blocks.append(
                GhostBottleneck(in_ch, hidden_ch, out_ch, kernel, stride)
            )

        # Final convolution
        self.conv2 = nn.Sequential(
            nn.Conv2d(160, 960, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(960),
            nn.ReLU(inplace=True),
        )

        # Global average pooling
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        # Embedding layers
        self.feat_bn = nn.BatchNorm1d(960)
        self.feat_bn.bias.requires_grad_(False)

        self.embedding_proj = nn.Linear(960, embedding_dim, bias=False)
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

        for block in self.blocks:
            x = block(x)

        x = self.conv2(x)
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


def get_ghostfacenet(
    embedding_dim: int = 512,
    weights_path: Optional[str] = None,
) -> GhostFaceNet:
    """
    Get GhostFaceNet backbone.

    Args:
        embedding_dim: output embedding dimension
        weights_path : path to pretrained weights

    Returns:
        GhostFaceNet backbone
    """
    model = GhostFaceNet(embedding_dim=embedding_dim)

    if weights_path is not None:
        state = torch.load(weights_path, map_location='cpu')
        model.load_state_dict(state)
        print(f"[GhostFaceNet] Loaded weights from: {weights_path}")

    return model
