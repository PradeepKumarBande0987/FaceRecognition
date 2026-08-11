"""
MobileFaceNet: Ultra-lightweight face recognition for mobile devices.

Key idea: Depthwise Separable Convolutions reduce parameters 8-9x.

Depthwise Separable Conv:
    Standard Conv (H×W×in_ch → out_ch): kernel²×in_ch×out_ch params
    Depthwise Conv (H×W×in_ch → in_ch): kernel²×in_ch params per-channel
    Pointwise Conv (in_ch → out_ch): in_ch×out_ch params 1×1
    
    Total: kernel²×in_ch + in_ch×out_ch ≈ 8-9x fewer params

Use case: Mobile devices, edge deployment, real-time face recognition

Performance:
    • Parameters: ~1M (vs 25M for ResNet50)
    • FLOPs: ~40M (vs 4B for ResNet50)
    • Speed: ~50ms on mobile CPU
    • Accuracy: ~96-97% on LFW (vs 99%+ for ResNet50)
    • Memory: ~4MB model size
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict


class DepthwiseSeparableConv(nn.Module):
    """
    Depthwise Separable Convolution: efficient alternative to standard Conv.
    
    Structure:
        Depthwise (per-channel) → Pointwise (channel mixing)
    
    Reduces parameters from kernel²×in_ch×out_ch to
    kernel²×in_ch + in_ch×out_ch ≈ 8-9x fewer
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        padding: int = 1,
    ):
        super().__init__()
        # Depthwise: grouped convolution with groups=in_channels
        self.depthwise = nn.Conv2d(in_channels, in_channels, kernel_size,
                                  stride=stride, padding=padding,
                                  groups=in_channels, bias=False)
        self.bn1 = nn.BatchNorm2d(in_channels)

        # Pointwise: 1×1 convolution
        self.pointwise = nn.Conv2d(in_channels, out_channels, kernel_size=1,
                                  stride=1, padding=0, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.depthwise(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.pointwise(x)
        x = self.bn2(x)
        x = self.relu(x)

        return x


class MobileFaceNet(nn.Module):
    """
    MobileFaceNet: Ultra-lightweight face recognition backbone.

    Architecture:
        Conv(3×3) → ReLU
        → Depthwise Separable Blocks with stride 2 (progressive downsampling)
        → AdaptiveAvgPool → Flatten
        → Embedding projection → BatchNorm → L2 normalize

    Specifications:
        • Input: 112×112×3
        • Output: embedding_dim (typically 128)
        • Parameters: ~1M
        • FLOPs: ~40M
        • Memory: ~4MB
    """

    def __init__(
        self,
        embedding_dim: int = 128,
    ):
        super().__init__()
        self.embedding_dim = embedding_dim

        # Initial convolution
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )

        # Depthwise separable blocks
        # Progressive: 112 → 56 → 28 → 14 → 7 → 7
        self.dw_conv1 = DepthwiseSeparableConv(64, 64, kernel_size=3, stride=1)
        self.dw_conv2 = DepthwiseSeparableConv(64, 128, kernel_size=3, stride=2)
        self.dw_conv3 = DepthwiseSeparableConv(128, 128, kernel_size=3, stride=1)
        self.dw_conv4 = DepthwiseSeparableConv(128, 128, kernel_size=3, stride=1)
        self.dw_conv5 = DepthwiseSeparableConv(128, 256, kernel_size=3, stride=2)
        self.dw_conv6 = DepthwiseSeparableConv(256, 256, kernel_size=3, stride=1)
        self.dw_conv7 = DepthwiseSeparableConv(256, 256, kernel_size=3, stride=1)

        # Downsample to 7×7
        self.dw_conv8 = DepthwiseSeparableConv(256, 512, kernel_size=3, stride=2)
        self.dw_conv9 = DepthwiseSeparableConv(512, 512, kernel_size=3, stride=1)
        self.dw_conv10 = DepthwiseSeparableConv(512, 512, kernel_size=3, stride=1)

        # Global average pooling
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        # Embedding
        self.feat_bn = nn.BatchNorm1d(512)
        self.feat_bn.bias.requires_grad_(False)

        self.embedding_proj = nn.Linear(512, embedding_dim, bias=False)
        self.embedding_bn = nn.BatchNorm1d(embedding_dim)
        self.embedding_bn.bias.requires_grad_(False)

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Args:
            x: [B, 3, 112, 112]

        Returns:
            dict with embedding and logits
        """
        x = self.conv1(x)

        x = self.dw_conv1(x)
        x = self.dw_conv2(x)
        x = self.dw_conv3(x)
        x = self.dw_conv4(x)
        x = self.dw_conv5(x)
        x = self.dw_conv6(x)
        x = self.dw_conv7(x)
        x = self.dw_conv8(x)
        x = self.dw_conv9(x)
        x = self.dw_conv10(x)

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


def get_mobilefacenet(
    embedding_dim: int = 128,
    weights_path: Optional[str] = None,
) -> MobileFaceNet:
    """
    Get MobileFaceNet backbone.

    Args:
        embedding_dim: output embedding dimension
        weights_path : path to pretrained weights

    Returns:
        MobileFaceNet backbone
    """
    model = MobileFaceNet(embedding_dim=embedding_dim)

    if weights_path is not None:
        state = torch.load(weights_path, map_location='cpu')
        model.load_state_dict(state)
        print(f"[MobileFaceNet] Loaded weights from: {weights_path}")

    return model
