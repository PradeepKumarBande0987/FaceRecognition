"""
FaceNet: Original triplet loss-based face recognition backbone.

Paper: FaceNet: A Unified Embedding for Face Recognition and Clustering
Link: https://arxiv.org/abs/1503.03832

Key contributions:
    1. Triplet loss: learns embeddings such that:
       - Same identity faces are close (minimize distance)
       - Different identity faces are far (maximize distance)
    
    2. Large margin training: explicit margin between classes
    
    3. Inception-ResNet-v1: Hybrid of Inception and ResNet architectures

Architecture:
    Inception modules (parallel convolution paths) + residual connections
    Generate highly discriminative face embeddings

Loss function:
    L = max(d(a, p) - d(a, n) + margin, 0)
    where:
        d = Euclidean distance
        a = anchor (reference face)
        p = positive (same identity)
        n = negative (different identity)
        margin = minimum required separation

Use case: Original face recognition method, still competitive
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict


class Inception(nn.Module):
    """
    Inception module: parallel convolutional paths.
    
    Outputs:
        1×1 conv → 1×1 conv → 3×3 conv (dimension reduction)
               → 1×1 conv → 5×5 conv
               → MaxPool → 1×1 conv
        Concatenate all paths
    """

    def __init__(
        self,
        in_channels: int,
        out_1x1: int,
        red_3x3: int,
        out_3x3: int,
        red_5x5: int,
        out_5x5: int,
        out_pool: int,
    ):
        super().__init__()

        # 1×1 path
        self.branch1x1 = nn.Sequential(
            nn.Conv2d(in_channels, out_1x1, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_1x1),
            nn.ReLU(inplace=True),
        )

        # 3×3 path
        self.branch3x3 = nn.Sequential(
            nn.Conv2d(in_channels, red_3x3, kernel_size=1, bias=False),
            nn.BatchNorm2d(red_3x3),
            nn.ReLU(inplace=True),
            nn.Conv2d(red_3x3, out_3x3, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_3x3),
            nn.ReLU(inplace=True),
        )

        # 5×5 path
        self.branch5x5 = nn.Sequential(
            nn.Conv2d(in_channels, red_5x5, kernel_size=1, bias=False),
            nn.BatchNorm2d(red_5x5),
            nn.ReLU(inplace=True),
            nn.Conv2d(red_5x5, out_5x5, kernel_size=5, padding=2, bias=False),
            nn.BatchNorm2d(out_5x5),
            nn.ReLU(inplace=True),
        )

        # MaxPool path
        self.branch_pool = nn.Sequential(
            nn.MaxPool2d(kernel_size=3, stride=1, padding=1),
            nn.Conv2d(in_channels, out_pool, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_pool),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        branch1 = self.branch1x1(x)
        branch2 = self.branch3x3(x)
        branch3 = self.branch5x5(x)
        branch4 = self.branch_pool(x)
        return torch.cat([branch1, branch2, branch3, branch4], 1)


class InceptionResNetV1(nn.Module):
    """
    Inception-ResNet-v1: Hybrid Inception + ResNet for face recognition.
    
    Combines:
        • Inception modules: multi-scale feature extraction
        • Residual connections: easy gradient flow, train deeper networks
    """

    def __init__(
        self,
        pretrained: bool = False,
        embedding_dim: int = 128,
    ):
        super().__init__()
        self.embedding_dim = embedding_dim

        # Initial convolution
        self.conv2d_1a = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=2, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
        )

        self.conv2d_2a = nn.Sequential(
            nn.Conv2d(32, 32, kernel_size=3, stride=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
        )

        self.conv2d_2b = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )

        self.maxpool_3a = nn.MaxPool2d(kernel_size=3, stride=2)

        # Inception modules
        self.conv2d_3b = nn.Sequential(
            nn.Conv2d(64, 80, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(80),
            nn.ReLU(inplace=True),
        )

        self.conv2d_4a = nn.Sequential(
            nn.Conv2d(80, 192, kernel_size=3, stride=1, bias=False),
            nn.BatchNorm2d(192),
            nn.ReLU(inplace=True),
        )

        self.maxpool_5a = nn.MaxPool2d(kernel_size=3, stride=2)

        self.mixed_5b = Inception(192, 96, 48, 64, 8, 64, 32)

        # Global average pooling
        self.avgpool_1a = nn.AdaptiveAvgPool2d((1, 1))

        # Embedding
        self.feat_bn = nn.BatchNorm1d(256)
        self.feat_bn.bias.requires_grad_(False)

        self.embedding_proj = nn.Linear(256, embedding_dim, bias=False)
        self.embedding_bn = nn.BatchNorm1d(embedding_dim)
        self.embedding_bn.bias.requires_grad_(False)

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Args:
            x: [B, 3, H, W]

        Returns:
            dict with embedding and logits
        """
        x = self.conv2d_1a(x)
        x = self.conv2d_2a(x)
        x = self.conv2d_2b(x)
        x = self.maxpool_3a(x)

        x = self.conv2d_3b(x)
        x = self.conv2d_4a(x)
        x = self.maxpool_5a(x)

        x = self.mixed_5b(x)

        x = self.avgpool_1a(x)
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


class FaceNet(InceptionResNetV1):
    """FaceNet backbone (alias for InceptionResNetV1)."""
    pass
