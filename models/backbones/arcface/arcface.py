"""
ArcFace Backbone: ResNet-based architecture optimized for ArcFace loss.

Paper: ArcFace: Additive Angular Margin Loss for Deep Face Recognition
Link: https://arxiv.org/abs/1801.07698

Architecture:
    Conv → BatchNorm → ReLU → MaxPool
    → ResLayer1 (64 ch)   → ResLayer2 (128 ch)
    → ResLayer3 (256 ch)  → ResLayer4 (512 ch)
    → AdaptiveAvgPool     → FC (embedding_dim)
    → BatchNorm + Normalize (L2)

Key features:
    • Large margin cosine loss (ArcFace) for angular separation
    • Normalized embeddings (L2 normalization)
    • Compatible with ArcFace, CosFace, SphereFace losses
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict, List
import math


# ── Building Blocks ───────────────────────────────────────────────────────────

class BasicBlock(nn.Module):
    """
    Basic residual block (2 Conv layers).
    Used in: ResNet18, ResNet34
    """
    expansion = 1

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stride: int = 1,
        downsample: Optional[nn.Module] = None,
    ):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3,
                              stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3,
                              stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.downsample = downsample
        self.stride = stride

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)
        return out


class Bottleneck(nn.Module):
    """
    Bottleneck residual block (3 Conv layers: 1×1 → 3×3 → 1×1).
    Used in: ResNet50, ResNet101, ResNet152
    
    Efficiency: 1×1 reduces channels, 3×3 works on reduced dims
    """
    expansion = 4

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stride: int = 1,
        downsample: Optional[nn.Module] = None,
    ):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=1,
                              bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)

        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3,
                              stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.conv3 = nn.Conv2d(out_channels, out_channels * self.expansion,
                              kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(out_channels * self.expansion)

        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)
        return out


# ── ResNet Backbone ──────────────────────────────────────────────────────────

class ResNet(nn.Module):
    """
    Standard ResNet backbone for face recognition.
    
    Args:
        block: BasicBlock or Bottleneck
        layers: list of block counts per layer [2, 2, 2, 2] etc
        embedding_dim: output embedding dimension (default 512)
    """

    def __init__(
        self,
        block,
        layers: List[int],
        embedding_dim: int = 512,
        zero_init_residual: bool = True,
    ):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.in_channels = 64

        # Initial convolution
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
                              bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # Residual layers
        self.layer1 = self._make_layer(block, 64, layers[0], stride=1)
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)

        # Global average pooling
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        # Embedding layers
        final_channels = 512 * block.expansion
        self.feat_bn = nn.BatchNorm1d(final_channels)
        self.feat_bn.bias.requires_grad_(False)

        self.embedding_proj = nn.Linear(final_channels, embedding_dim,
                                       bias=False)
        self.embedding_bn = nn.BatchNorm1d(embedding_dim)
        self.embedding_bn.bias.requires_grad_(False)

        # Initialize weights
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out',
                                       nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)

    def _make_layer(
        self,
        block,
        out_channels: int,
        blocks: int,
        stride: int = 1,
    ) -> nn.Sequential:
        """Build residual layer."""
        downsample = None

        if stride != 1 or self.in_channels != out_channels * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.in_channels, out_channels * block.expansion,
                         kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels * block.expansion),
            )

        layers = []
        layers.append(block(self.in_channels, out_channels, stride, downsample))
        self.in_channels = out_channels * block.expansion

        for _ in range(1, blocks):
            layers.append(block(self.in_channels, out_channels))

        return nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Args:
            x: [B, 3, H, W] face image

        Returns:
            dict:
                'embedding': [B, embedding_dim] L2-normalized embedding
                'logit'    : [B, embedding_dim] before L2 norm
                'feat_raw' : [B, 512*expansion] before projection
        """
        # Initial layers
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        # Residual layers
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        # Global average pooling
        x = self.avgpool(x)
        feat_raw = torch.flatten(x, 1)

        # Feature projection
        feat_raw = self.feat_bn(feat_raw)
        logit = self.embedding_proj(feat_raw)
        embedding = self.embedding_bn(logit)

        # L2 normalization
        embedding = F.normalize(embedding, p=2, dim=1)

        return {
            "embedding": embedding,
            "logit"    : logit,
            "feat_raw" : feat_raw,
        }


class ResNetArcFace(ResNet):
    """
    ResNet backbone with ArcFace-specific configurations.
    
    Optimized for:
        • Large margin cosine loss
        • Angular separation of identity classes
        • Normalized embeddings
    """

    def __init__(
        self,
        block,
        layers: List[int],
        embedding_dim: int = 512,
    ):
        super().__init__(block, layers, embedding_dim, zero_init_residual=True)


# ── Factory Functions ────────────────────────────────────────────────────────

def get_resnet(
    depth: int = 50,
    embedding_dim: int = 512,
    pretrained: bool = False,
    weights_path: Optional[str] = None,
) -> ResNet:
    """
    Get ResNet backbone by depth.

    Args:
        depth      : 18, 34, 50, 101, 152
        embedding_dim: output embedding dimension
        pretrained : load ImageNet pretrained weights
        weights_path: path to custom weights

    Returns:
        ResNet backbone
    """
    if depth == 18:
        model = ResNetArcFace(BasicBlock, [2, 2, 2, 2], embedding_dim)
    elif depth == 34:
        model = ResNetArcFace(BasicBlock, [3, 4, 6, 3], embedding_dim)
    elif depth == 50:
        model = ResNetArcFace(Bottleneck, [3, 4, 6, 3], embedding_dim)
    elif depth == 101:
        model = ResNetArcFace(Bottleneck, [3, 4, 23, 3], embedding_dim)
    elif depth == 152:
        model = ResNetArcFace(Bottleneck, [3, 8, 36, 3], embedding_dim)
    else:
        raise ValueError(f"Unsupported ResNet depth: {depth}")

    if weights_path is not None:
        state = torch.load(weights_path, map_location='cpu')
        model.load_state_dict(state)
        print(f"[ResNet] Loaded weights from: {weights_path}")

    return model
