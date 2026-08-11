"""
Vision Transformer (ViT) for Face Recognition.

Paper: An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale
Link: https://arxiv.org/abs/2010.11929

Architecture:
    1. Patch Embedding: Split image into patches, project to embedding dim
    2. Positional Encoding: Add learnable position embeddings
    3. Transformer Encoder: Stack of multi-head self-attention + MLP
    4. Classification: Use [CLS] token output for classification

Advantages:
    • Global receptive field from start (sees entire face)
    • Scales well with data (needs large training sets)
    • Captures long-range dependencies
    • Can handle variable input sizes

Disadvantages:
    • More parameters than CNNs
    • Slower inference on CPU
    • Requires more training data
    • Less data-efficient than CNNs (without pretraining)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional
import math


class PatchEmbedding(nn.Module):
    """
    Convert image to patch embeddings.

    Process:
        1. Divide image into non-overlapping patches
        2. Flatten each patch
        3. Project to embedding dimension

    Input: [B, 3, H, W]
    Output: [B, num_patches, embed_dim]
    """

    def __init__(
        self,
        patch_size: int = 16,
        in_channels: int = 3,
        embed_dim: int = 768,
    ):
        super().__init__()
        self.patch_size = patch_size
        self.proj = nn.Conv2d(in_channels, embed_dim, kernel_size=patch_size,
                             stride=patch_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, 3, H, W]

        Returns:
            patches: [B, num_patches, embed_dim]
        """
        x = self.proj(x)  # [B, embed_dim, H', W']
        x = x.flatten(2).transpose(1, 2)  # [B, H'*W', embed_dim]
        return x


class TransformerBlock(nn.Module):
    """
    Transformer encoder block: LayerNorm → Attention → MLP with residuals.

    Structure:
        x → LayerNorm → MultiHeadAttention → + (residual)
          → LayerNorm → MLP → + (residual)
    """

    def __init__(
        self,
        embed_dim: int = 768,
        num_heads: int = 12,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
    ):
        super().__init__()

        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn = nn.MultiheadAttention(embed_dim, num_heads,
                                         dropout=dropout, batch_first=True)

        self.norm2 = nn.LayerNorm(embed_dim)
        mlp_hidden_dim = int(embed_dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, mlp_hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden_dim, embed_dim),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: [B, num_patches, embed_dim]"""
        # Attention
        x_norm = self.norm1(x)
        attn_out, _ = self.attn(x_norm, x_norm, x_norm)
        x = x + attn_out

        # MLP
        x_norm = self.norm2(x)
        mlp_out = self.mlp(x_norm)
        x = x + mlp_out

        return x


class ViTFace(nn.Module):
    """
    Vision Transformer for face recognition.

    Configuration:
        • Patch size: 16 (splits 224×224 into 196 patches)
        • Embedding dim: 768
        • Depth: 12 blocks
        • Heads: 12
    """

    def __init__(
        self,
        patch_size: int = 16,
        embed_dim: int = 768,
        depth: int = 12,
        num_heads: int = 12,
        mlp_ratio: float = 4.0,
        embedding_dim: int = 512,
        dropout: float = 0.0,
        input_size: int = 224,
    ):
        super().__init__()
        self.patch_size = patch_size
        self.embed_dim = embed_dim
        self.embedding_dim = embedding_dim
        self.num_patches = (input_size // patch_size) ** 2

        # Patch embedding
        self.patch_embed = PatchEmbedding(patch_size, in_channels=3,
                                         embed_dim=embed_dim)

        # CLS token
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))

        # Positional embeddings
        self.pos_embed = nn.Parameter(
            torch.zeros(1, self.num_patches + 1, embed_dim)
        )

        # Transformer blocks
        self.blocks = nn.ModuleList([
            TransformerBlock(embed_dim, num_heads, mlp_ratio, dropout)
            for _ in range(depth)
        ])

        self.norm = nn.LayerNorm(embed_dim)

        # Embedding projection
        self.feat_bn = nn.BatchNorm1d(embed_dim)
        self.feat_bn.bias.requires_grad_(False)

        self.embedding_proj = nn.Linear(embed_dim, embedding_dim, bias=False)
        self.embedding_bn = nn.BatchNorm1d(embedding_dim)
        self.embedding_bn.bias.requires_grad_(False)

        # Initialize
        nn.init.normal_(self.cls_token, std=0.02)
        nn.init.normal_(self.pos_embed, std=0.02)

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Args:
            x: [B, 3, H, W]

        Returns:
            dict with embedding and logits
        """
        B = x.shape[0]

        # Patch embedding
        x = self.patch_embed(x)  # [B, num_patches, embed_dim]

        # Add CLS token
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat([cls_tokens, x], dim=1)  # [B, 1+num_patches, embed_dim]

        # Add positional embedding
        x = x + self.pos_embed

        # Transformer blocks
        for block in self.blocks:
            x = block(x)

        x = self.norm(x)

        # Use CLS token
        feat_raw = x[:, 0]  # [B, embed_dim]

        # Projection
        feat_raw = self.feat_bn(feat_raw)
        logit = self.embedding_proj(feat_raw)
        embedding = self.embedding_bn(logit)
        embedding = F.normalize(embedding, p=2, dim=1)

        return {
            "embedding": embedding,
            "logit"    : logit,
            "feat_raw" : feat_raw,
        }


def get_vit_face(
    embedding_dim: int = 512,
    weights_path: Optional[str] = None,
) -> ViTFace:
    """
    Get ViT-Face backbone.

    Args:
        embedding_dim: output embedding dimension
        weights_path : path to pretrained weights

    Returns:
        ViTFace backbone
    """
    model = ViTFace(embedding_dim=embedding_dim)

    if weights_path is not None:
        state = torch.load(weights_path, map_location='cpu')
        model.load_state_dict(state)
        print(f"[ViTFace] Loaded weights from: {weights_path}")

    return model
