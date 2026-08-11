"""
Multi-Modal Fusion Module
Fuses multiple biometric modalities for robust face recognition:
    • RGB appearance features
    • Depth/3D shape features
    • Infrared (IR) features
    • Audio-visual features (lip sync / voice)
    • Thermal features
Architecture: Attention-based adaptive fusion with modality dropout
              for robustness when some modalities are unavailable.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Dict, List
import math

# ── Building Blocks ───────────────────────────────────────────────────────────

class ConvBNReLU(nn.Module):
    """Standard Conv → BatchNorm → LeakyReLU block."""

    def __init__(
        self,
        in_ch   : int,
        out_ch  : int,
        kernel  : int  = 3,
        stride  : int  = 1,
        padding : int  = 1,
        dilation: int  = 1,
        use_bn  : bool = True,
    ):
        super().__init__()
        layers = [
            nn.Conv2d(in_ch, out_ch, kernel,
                      stride=stride, padding=padding,
                      dilation=dilation, bias=not use_bn),
        ]
        if use_bn:
            layers.append(nn.BatchNorm2d(out_ch))
        layers.append(nn.LeakyReLU(0.2, inplace=True))
        self.block = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class ResidualBlock(nn.Module):
    """
    Residual block with two ConvBNReLU layers.
    Used inside modality-specific encoders.
    """

    def __init__(self, ch: int):
        super().__init__()
        self.block = nn.Sequential(
            ConvBNReLU(ch, ch),
            nn.Conv2d(ch, ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(ch),
        )
        self.relu = nn.LeakyReLU(0.2, inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.relu(x + self.block(x))


class MLP(nn.Module):
    """
    Multi-Layer Perceptron block.
    Used in attention and fusion heads.
    """

    def __init__(
        self,
        in_dim  : int,
        hidden  : int,
        out_dim : int,
        dropout : float = 0.1,
    ):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# ── Modality Encoders ─────────────────────────────────────────────────────────

class RGBEncoder(nn.Module):
    """
    RGB appearance encoder.
    Extracts identity-discriminative features from
    standard color face images.
    Input : [B, 3, H, W]
    Output: [B, feat_dim]
    """

    def __init__(self, feat_dim: int = 256):
        super().__init__()
        self.encoder = nn.Sequential(
            ConvBNReLU(3,   32),
            ResidualBlock(32),
            nn.MaxPool2d(2),                    # H/2

            ConvBNReLU(32,  64),
            ResidualBlock(64),
            nn.MaxPool2d(2),                    # H/4

            ConvBNReLU(64,  128),
            ResidualBlock(128),
            nn.MaxPool2d(2),                    # H/8

            ConvBNReLU(128, 256),
            ResidualBlock(256),
            nn.AdaptiveAvgPool2d(4),            # [B, 256, 4, 4]
            nn.Flatten(),                       # [B, 4096]
        )
        self.proj = nn.Sequential(
            nn.Linear(256 * 4 * 4, feat_dim),
            nn.LayerNorm(feat_dim),
            nn.GELU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(self.encoder(x))       # [B, feat_dim]


class DepthEncoder(nn.Module):
    """
    Depth / 3D shape encoder.
    Processes single-channel depth maps capturing
    3D facial geometry (nose bridge, cheek contours).
    Real faces have smooth, continuous depth;
    masks/prints are flat or geometrically inconsistent.
    Input : [B, 1, H, W]
    Output: [B, feat_dim]
    """

    def __init__(self, feat_dim: int = 256):
        super().__init__()
        self.encoder = nn.Sequential(
            ConvBNReLU(1,   32),
            ResidualBlock(32),
            nn.MaxPool2d(2),

            ConvBNReLU(32,  64),
            ResidualBlock(64),
            nn.MaxPool2d(2),

            ConvBNReLU(64,  128),
            ResidualBlock(128),
            nn.AdaptiveAvgPool2d(4),
            nn.Flatten(),
        )
        self.proj = nn.Sequential(
            nn.Linear(128 * 4 * 4, feat_dim),
            nn.LayerNorm(feat_dim),
            nn.GELU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(self.encoder(x))       # [B, feat_dim]


class IREncoder(nn.Module):
    """
    Infrared (IR) / Near-Infrared (NIR) encoder.
    IR imaging is illumination-invariant, capturing
    facial vascular patterns invisible to RGB cameras.
    Robust to lighting changes, makeup, and partial occlusion.
    Input : [B, 1, H, W]
    Output: [B, feat_dim]
    """

    def __init__(self, feat_dim: int = 256):
        super().__init__()
        self.encoder = nn.Sequential(
            ConvBNReLU(1,   32),
            ResidualBlock(32),
            nn.MaxPool2d(2),

            ConvBNReLU(32,  64),
            ResidualBlock(64),
            nn.MaxPool2d(2),

            ConvBNReLU(64,  128),
            ResidualBlock(128),
            nn.MaxPool2d(2),

            ConvBNReLU(128, 256),
            nn.AdaptiveAvgPool2d(4),
            nn.Flatten(),
        )
        self.proj = nn.Sequential(
            nn.Linear(256 * 4 * 4, feat_dim),
            nn.LayerNorm(feat_dim),
            nn.GELU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(self.encoder(x))       # [B, feat_dim]


class ThermalEncoder(nn.Module):
    """
    Thermal imaging encoder.
    Thermal cameras capture facial heat signatures —
    unique vascular heat patterns tied to identity.
    Works in complete darkness, highly spoof-resistant.
    Input : [B, 1, H, W]
    Output: [B, feat_dim]
    """

    def __init__(self, feat_dim: int = 256):
        super().__init__()
        self.encoder = nn.Sequential(
            # Wider early layers: thermal has coarser spatial resolution
            ConvBNReLU(1,   16),
            ConvBNReLU(16,  32),
            nn.MaxPool2d(2),

            ConvBNReLU(32,  64),
            ResidualBlock(64),
            nn.MaxPool2d(2),

            ConvBNReLU(64,  128),
            ResidualBlock(128),
            nn.AdaptiveAvgPool2d(4),
            nn.Flatten(),
        )
        self.proj = nn.Sequential(
            nn.Linear(128 * 4 * 4, feat_dim),
            nn.LayerNorm(feat_dim),
            nn.GELU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(self.encoder(x))       # [B, feat_dim]


class AudioEncoder(nn.Module):
    """
    Audio feature encoder for audio-visual fusion.
    Processes mel-spectrogram of speech paired with
    lip movements for audio-visual matching.
    Detects lip-sync inconsistencies in deepfake videos.

    Input : mel-spectrogram [B, 1, F, T]
            F = mel frequency bins (e.g. 80)
            T = time frames
    Output: [B, feat_dim]
    """

    def __init__(
        self,
        feat_dim: int = 256,
        freq_bins: int = 80,
    ):
        super().__init__()
        self.encoder = nn.Sequential(
            # Frequency axis convolutions
            nn.Conv2d(1,  32, kernel_size=(3, 3), padding=1),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.2, inplace=True),
            nn.MaxPool2d((2, 1)),              # compress freq, keep time

            nn.Conv2d(32, 64, kernel_size=(3, 3), padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2, inplace=True),
            nn.MaxPool2d((2, 1)),

            nn.Conv2d(64, 128, kernel_size=(3, 3), padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.AdaptiveAvgPool2d((4, 4)),
            nn.Flatten(),
        )
        self.proj = nn.Sequential(
            nn.Linear(128 * 4 * 4, feat_dim),
            nn.LayerNorm(feat_dim),
            nn.GELU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(self.encoder(x))       # [B, feat_dim]


class LipEncoder(nn.Module):
    """
    Lip region encoder for audio-visual synchrony check.
    Encodes lip appearance features from a temporal window
    to match against audio features.
    Input : lip crops sequence [B, T, 3, 48, 96]
    Output: [B, feat_dim]
    """

    def __init__(
        self,
        feat_dim: int = 256,
        seq_len : int = 25,
    ):
        super().__init__()
        self.seq_len      = seq_len
        self.frame_encoder = nn.Sequential(
            ConvBNReLU(3,  32),
            nn.MaxPool2d(2),
            ConvBNReLU(32, 64),
            nn.MaxPool2d(2),
            ConvBNReLU(64, 128),
            nn.AdaptiveAvgPool2d(2),
            nn.Flatten(),                       # [B, 128*4]
            nn.Linear(128 * 4, 128),
            nn.GELU(),
        )
        # Temporal aggregation over lip sequence
        self.temporal = nn.Sequential(
            nn.Conv1d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
        )
        self.proj = nn.Sequential(
            nn.Linear(128, feat_dim),
            nn.LayerNorm(feat_dim),
            nn.GELU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: [B, T, 3, H, W]"""
        B, T = x.shape[:2]
        frame_feats = torch.stack(
            [self.frame_encoder(x[:, t]) for t in range(T)], dim=2
        )                                        # [B, 128, T]
        temporal    = self.temporal(frame_feats) # [B, 128]
        return self.proj(temporal)               # [B, feat_dim]


# ── Cross-Modal Attention ─────────────────────────────────────────────────────

class CrossModalAttention(nn.Module):
    """
    Cross-Modal Attention between two modality feature vectors.
    Allows one modality to query relevant information from another.

    Example: RGB queries depth features at semantically matching
             face regions (e.g. nose tip has high depth value AND
             distinctive RGB texture).

    Uses multi-head attention treating each feature dim as a token.
    """

    def __init__(
        self,
        feat_dim  : int = 256,
        num_heads : int = 8,
        dropout   : float = 0.1,
    ):
        super().__init__()
        self.attn = nn.MultiheadAttention(
            embed_dim=feat_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.norm1 = nn.LayerNorm(feat_dim)
        self.norm2 = nn.LayerNorm(feat_dim)
        self.ffn   = MLP(feat_dim, feat_dim * 2, feat_dim, dropout)

    def forward(
        self,
        query: torch.Tensor,    # [B, feat_dim]  query modality
        key  : torch.Tensor,    # [B, feat_dim]  key/value modality
    ) -> torch.Tensor:
        """
        query attends to key modality features.
        Returns enriched query: [B, feat_dim]
        """
        # Expand to sequence dim=1 for attention: [B, 1, feat_dim]
        q = query.unsqueeze(1)
        k = key.unsqueeze(1)

        attended, _ = self.attn(q, k, k)        # [B, 1, feat_dim]
        attended    = attended.squeeze(1)        # [B, feat_dim]

        # Residual + LayerNorm
        out = self.norm1(query + attended)
        out = self.norm2(out + self.ffn(out))
        return out                               # [B, feat_dim]


# ── Modality Quality Estimator ────────────────────────────────────────────────

class ModalityQualityEstimator(nn.Module):
    """
    Estimates the quality / reliability of each modality input.
    Low-quality modalities (blurry, occluded, missing)
    should receive lower fusion weights.

    Outputs a quality score ∈ [0, 1] per modality per sample.
    Used to compute adaptive fusion weights.
    """

    def __init__(self, feat_dim: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(feat_dim, 64),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(64, 32),
            nn.GELU(),
            nn.Linear(32, 1),
            nn.Sigmoid(),                        # quality ∈ [0, 1]
        )

    def forward(self, feat: torch.Tensor) -> torch.Tensor:
        """feat: [B, feat_dim] → quality: [B, 1]"""
        return self.net(feat)


# ── Adaptive Fusion Gate ──────────────────────────────────────────────────────

class AdaptiveFusionGate(nn.Module):
    """
    Adaptive modality fusion gate.

    Computes dynamic per-sample fusion weights for each modality
    based on:
        1. Quality scores from ModalityQualityEstimator
        2. Cross-modal compatibility (attention scores)
        3. Modality availability mask (0 if modality missing)

    This makes the system robust to missing modalities at inference
    (e.g. no IR camera → IR weight → 0, others compensate).
    """

    def __init__(
        self,
        num_modalities: int,
        feat_dim      : int = 256,
    ):
        super().__init__()
        self.num_modalities   = num_modalities
        self.quality_estimator = nn.ModuleList([
            ModalityQualityEstimator(feat_dim)
            for _ in range(num_modalities)
        ])

        # Gating network: takes all quality scores → softmax weights
        self.gate = nn.Sequential(
            nn.Linear(num_modalities, num_modalities * 4),
            nn.GELU(),
            nn.Linear(num_modalities * 4, num_modalities),
        )

    def forward(
        self,
        feats          : List[torch.Tensor],  # list of [B, feat_dim]
        availability   : Optional[torch.Tensor] = None,
                                               # [B, num_modalities] binary mask
    ) -> torch.Tensor:
        """
        Returns fusion weights: [B, num_modalities]
        """
        B = feats[0].size(0)

        # Quality score for each modality
        quality_scores = torch.cat(
            [self.quality_estimator[i](feats[i]) for i in range(self.num_modalities)],
            dim=1,
        )                                      # [B, num_modalities]

        # Gate over quality
        raw_weights = self.gate(quality_scores)  # [B, num_modalities]

        # Zero out unavailable modalities
        if availability is not None:
            raw_weights = raw_weights * availability

        # Softmax over available modalities
        # Add small epsilon to avoid div-by-zero when all masked
        weights = F.softmax(raw_weights, dim=1)  # [B, num_modalities]

        return weights


# ── Transformer Fusion Encoder ────────────────────────────────────────────────

class ModalityTokenTransformer(nn.Module):
    """
    Treats each modality feature vector as a token and
    runs a Transformer encoder over the modality sequence.

    This allows full pairwise cross-modal reasoning:
        • RGB ↔ Depth
        • RGB ↔ IR
        • Depth ↔ Thermal
        • All modalities jointly

    Architecture:
        [M modality tokens of dim feat_dim]
            │
            ├── Positional modality embedding
            │
            ├── Transformer Encoder (L layers)
            │       Multi-head Self-Attention
            │       FFN
            │
            └── [CLS] token → fused identity embedding
    """

    def __init__(
        self,
        num_modalities: int,
        feat_dim      : int   = 256,
        num_heads     : int   = 8,
        num_layers    : int   = 4,
        dropout       : float = 0.1,
    ):
        super().__init__()
        self.feat_dim = feat_dim

        # Learnable modality-type embeddings (like segment embeddings in BERT)
        self.modality_embed = nn.Embedding(num_modalities + 1, feat_dim)
                                                # +1 for [CLS] token

        # [CLS] token (learnable)
        self.cls_token = nn.Parameter(torch.randn(1, 1, feat_dim))

        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=feat_dim,
            nhead=num_heads,
            dim_feedforward=feat_dim * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,            # Pre-LayerNorm for stability
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers,
            norm=nn.LayerNorm(feat_dim),
        )

        # Project [CLS] output to identity embedding
        self.cls_proj = nn.Sequential(
            nn.Linear(feat_dim, feat_dim),
            nn.LayerNorm(feat_dim),
            nn.GELU(),
        )

        self.num_modalities = num_modalities

    def forward(
        self,
        feats        : List[torch.Tensor],     # list of [B, feat_dim]
        weights      : Optional[torch.Tensor], # [B, num_modalities]
        key_padding_mask: Optional[torch.Tensor] = None,
                                               # [B, num_modalities+1] bool mask
    ) -> Dict[str, torch.Tensor]:
        """
        Args:
            feats           : list of M modality feature tensors [B, feat_dim]
            weights         : adaptive fusion weights [B, M]
            key_padding_mask: True = ignore token (missing modality)

        Returns:
            dict:
                'embedding'    : [B, feat_dim]  fused identity embedding
                'token_outputs': [B, M+1, feat_dim] all transformer outputs
        """
        B = feats[0].size(0)

        # Stack modality tokens: [B, M, feat_dim]
        tokens = torch.stack(feats, dim=1)

        # Scale each modality by its fusion weight
        if weights is not None:
            tokens = tokens * weights.unsqueeze(-1)   # broadcast over feat_dim

        # Add modality-type embeddings
        mod_ids = torch.arange(1, self.num_modalities + 1,
                               device=tokens.device)   # 1..M (0 reserved for CLS)
        tokens  = tokens + self.modality_embed(mod_ids).unsqueeze(0)

        # Prepend [CLS] token
        cls   = self.cls_token.expand(B, -1, -1)       # [B, 1, feat_dim]
        cls   = cls + self.modality_embed(
            torch.zeros(1, dtype=torch.long, device=tokens.device)
        )
        seq   = torch.cat([cls, tokens], dim=1)         # [B, M+1, feat_dim]

        # Build key padding mask (True = mask out)
        if key_padding_mask is not None:
            # Prepend False for [CLS] (always attend to CLS)
            cls_mask = torch.zeros(B, 1, dtype=torch.bool,
                                   device=tokens.device)
            full_mask = torch.cat([cls_mask, key_padding_mask], dim=1)
        else:
            full_mask = None

        # Transformer forward
        out     = self.transformer(seq, src_key_padding_mask=full_mask)
                                                        # [B, M+1, feat_dim]

        # CLS token = fused embedding
        cls_out = out[:, 0, :]                          # [B, feat_dim]
        embed   = self.cls_proj(cls_out)                # [B, feat_dim]

        return {
            "embedding"    : embed,
            "token_outputs": out,
        }


# ── Audio-Visual Synchrony ────────────────────────────────────────────────────

class AudioVisualSynchronyModule(nn.Module):
    """
    Audio-Visual Synchrony detector.
    Checks if lip movements match the audio signal.
    Deepfakes often fail lip-sync, making this a powerful
    anti-spoofing cue in video scenarios.

    Computes cosine similarity between:
        - Audio embedding
        - Lip visual embedding
    High similarity → synchronized (real or high-quality deepfake)
    Low similarity  → out-of-sync (likely fake)
    """

    def __init__(self, feat_dim: int = 256):
        super().__init__()
        # Project both to same comparison space
        self.audio_proj = nn.Sequential(
            nn.Linear(feat_dim, feat_dim),
            nn.LayerNorm(feat_dim),
        )
        self.lip_proj   = nn.Sequential(
            nn.Linear(feat_dim, feat_dim),
            nn.LayerNorm(feat_dim),
        )
        # Sync classifier from similarity features
        self.sync_head  = nn.Sequential(
            nn.Linear(feat_dim * 2 + 1, 128),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(128, 1),
        )

    def forward(
        self,
        audio_feat: torch.Tensor,   # [B, feat_dim]
        lip_feat  : torch.Tensor,   # [B, feat_dim]
    ) -> Dict[str, torch.Tensor]:

        a = F.normalize(self.audio_proj(audio_feat), dim=-1)  # [B, feat_dim]
        v = F.normalize(self.lip_proj(lip_feat),     dim=-1)  # [B, feat_dim]

        # Cosine similarity
        sim   = (a * v).sum(dim=-1, keepdim=True)              # [B, 1]

        # Sync classifier
        joint = torch.cat([a, v, sim], dim=-1)                 # [B, 2*D+1]
        logit = self.sync_head(joint)                          # [B, 1]
        prob  = torch.sigmoid(logit)

        return {
            "sync_logit": logit,     # [B, 1]
            "sync_prob" : prob,      # [B, 1]
            "similarity": sim,       # [B, 1]  cosine sim
        }


# ── Identity Embedding Head ───────────────────────────────────────────────────

class IdentityEmbeddingHead(nn.Module):
    """
    Projects fused multi-modal features to a compact
    L2-normalized identity embedding space.

    The embedding is used for:
        • Face verification (cosine distance between embeddings)
        • Face identification (nearest neighbor in embedding DB)
        • ArcFace / CosFace classification during training
    """

    def __init__(
        self,
        in_dim  : int = 256,
        emb_dim : int = 512,
    ):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, in_dim),
            nn.LayerNorm(in_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(in_dim, emb_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        emb = self.net(x)
        return F.normalize(emb, dim=-1)     # L2-normalized [B, emb_dim]


# ── ArcFace Classification Head ───────────────────────────────────────────────

class ArcFaceHead(nn.Module):
    """
    ArcFace classification head.
    Adds angular margin to cosine similarity for
    highly discriminative identity embeddings.

    L = -log( e^(s·cos(θ_yi + m)) /
              (e^(s·cos(θ_yi + m)) + Σ e^(s·cos(θ_j))) )

    s = scale (default 64), m = margin (default 0.5)
    """

    def __init__(
        self,
        emb_dim    : int   = 512,
        num_classes: int   = 10000,
        scale      : float = 64.0,
        margin     : float = 0.5,
    ):
        super().__init__()
        self.scale   = scale
        self.margin  = margin
        self.weight  = nn.Parameter(
            torch.FloatTensor(num_classes, emb_dim)
        )
        nn.init.xavier_uniform_(self.weight)

        self.cos_m   = math.cos(margin)
        self.sin_m   = math.sin(margin)
        self.th      = math.cos(math.pi - margin)
        self.mm      = math.sin(math.pi - margin) * margin

    def forward(
        self,
        embedding: torch.Tensor,    # [B, emb_dim]  L2-normalized
        labels   : Optional[torch.Tensor] = None,  # [B] class indices
    ) -> Dict[str, torch.Tensor]:

        # Cosine similarity with class weights
        cosine = F.linear(
            embedding,
            F.normalize(self.weight, dim=1)
        )                                           # [B, num_classes]

        if labels is None:
            return {"logit": cosine * self.scale}

        # ArcFace margin
        sine       = torch.sqrt((1.0 - cosine ** 2).clamp(1e-6, 1.0))
        phi        = cosine * self.cos_m - sine * self.sin_m
        phi        = torch.where(cosine > self.th, phi,
                                 cosine - self.mm)  # stable fallback

        # One-hot mask
        one_hot    = torch.zeros_like(cosine)
        one_hot.scatter_(1, labels.view(-1, 1), 1.0)

        # Apply margin only to ground-truth class
        output     = one_hot * phi + (1.0 - one_hot) * cosine
        logit      = output * self.scale

        return {
            "logit" : logit,         # [B, num_classes]
            "cosine": cosine,        # [B, num_classes]  raw similarity
        }


# ── Full Multi-Modal Fusion Module ────────────────────────────────────────────

class MultiModalFusion(nn.Module):
    """
    Full Multi-Modal Fusion pipeline.

    Supported modalities (all optional except RGB):
        • RGB      — color face image         [B, 3, H, W]
        • Depth    — depth map                [B, 1, H, W]
        • IR       — infrared image           [B, 1, H, W]
        • Thermal  — thermal image            [B, 1, H, W]
        • Audio    — mel-spectrogram          [B, 1, F, T]
        • Lips     — lip crop sequence        [B, T, 3, H, W]

    Pipeline:
        Each modality
            │
            ├── Modality-specific encoder → [B, feat_dim]
            │
            ├── Cross-Modal Attention (RGB ↔ each modality)
            │
            ├── AdaptiveFusionGate → per-sample weights [B, M]
            │
            ├── ModalityTokenTransformer → fused embedding [B, feat_dim]
            │
            ├── IdentityEmbeddingHead → L2 embedding [B, emb_dim]
            │
            └── ArcFaceHead → identity logits [B, num_classes]
    """

    MODALITY_NAMES = ["rgb", "depth", "ir", "thermal", "audio", "lips"]

    def __init__(
        self,
        feat_dim      : int   = 256,
        emb_dim       : int   = 512,
        num_classes   : int   = 10000,
        num_heads     : int   = 8,
        num_tf_layers : int   = 4,
        scale         : float = 64.0,
        margin        : float = 0.5,
        dropout       : float = 0.1,
        modality_drop_p: float = 0.2,  # prob of dropping a modality during training
    ):
        super().__init__()
        self.feat_dim       = feat_dim
        self.modality_drop_p = modality_drop_p

        # ── Modality Encoders ────────────────────────────────────────────────
        self.rgb_encoder     = RGBEncoder(feat_dim)
        self.depth_encoder   = DepthEncoder(feat_dim)
        self.ir_encoder      = IREncoder(feat_dim)
        self.thermal_encoder = ThermalEncoder(feat_dim)
        self.audio_encoder   = AudioEncoder(feat_dim)
        self.lip_encoder     = LipEncoder(feat_dim)

        # ── Cross-Modal Attention (RGB as primary query) ─────────────────────
        self.rgb_depth_attn   = CrossModalAttention(feat_dim, num_heads, dropout)
        self.rgb_ir_attn      = CrossModalAttention(feat_dim, num_heads, dropout)
        self.rgb_thermal_attn = CrossModalAttention(feat_dim, num_heads, dropout)
        self.av_sync          = AudioVisualSynchronyModule(feat_dim)

        # ── Adaptive Fusion Gate ─────────────────────────────────────────────
        self.fusion_gate = AdaptiveFusionGate(
            num_modalities=len(self.MODALITY_NAMES),
            feat_dim=feat_dim,
        )

        # ── Transformer Fusion ───────────────────────────────────────────────
        self.transformer_fusion = ModalityTokenTransformer(
            num_modalities=len(self.MODALITY_NAMES),
            feat_dim=feat_dim,
            num_heads=num_heads,
            num_layers=num_tf_layers,
            dropout=dropout,
        )

        # ── Identity Heads ───────────────────────────────────────────────────
        self.identity_head = IdentityEmbeddingHead(feat_dim, emb_dim)
        self.arcface_head  = ArcFaceHead(emb_dim, num_classes, scale, margin)

    def _modality_dropout(
        self,
        availability: torch.Tensor,   # [B, M]
    ) -> torch.Tensor:
        """
        Randomly zero out modality slots during training
        to simulate missing sensors and improve robustness.
        RGB (index 0) is never dropped.
        """
        if not self.training:
            return availability

        mask = torch.bernoulli(
            torch.full_like(availability, 1.0 - self.modality_drop_p)
        )
        mask[:, 0] = 1.0   # always keep RGB
        return availability * mask

    def forward(
        self,
        rgb          : torch.Tensor,                    # [B, 3, H, W]  required
        depth        : Optional[torch.Tensor] = None,   # [B, 1, H, W]
        ir           : Optional[torch.Tensor] = None,   # [B, 1, H, W]
        thermal      : Optional[torch.Tensor] = None,   # [B, 1, H, W]
        audio        : Optional[torch.Tensor] = None,   # [B, 1, F, T]
        lips         : Optional[torch.Tensor] = None,   # [B, T, 3, H, W]
        labels       : Optional[torch.Tensor] = None,   # [B]  class ids
        return_all   : bool = False,
    ) -> Dict[str, torch.Tensor]:
        """
        Args:
            rgb     : RGB face image         (required)
            depth   : depth map              (optional)
            ir      : infrared image         (optional)
            thermal : thermal image          (optional)
            audio   : mel-spectrogram        (optional)
            lips    : lip crop sequence      (optional)
            labels  : identity class ids     (optional, for ArcFace training)
            return_all: return all intermediate outputs

        Returns:
            dict:
                'embedding'  : [B, emb_dim]       L2-normalized identity embedding
                'logit'      : [B, num_classes]   ArcFace logits (if labels given)
                'weights'    : [B, M]             modality fusion weights
                'sync_prob'  : [B, 1]             AV sync probability (if audio+lips)
                + modality features (if return_all)
        """
        B = rgb.size(0)
        inputs = [rgb, depth, ir, thermal, audio, lips]

        # ── Step 1: Encode each available modality ───────────────────────────
        # Availability mask: 1 if modality provided, else 0
        availability = torch.tensor(
            [[1.0 if inp is not None else 0.0 for inp in inputs]],
            device=rgb.device,
        ).expand(B, -1).clone()                  # [B, M]

        # Encode
        rgb_feat     = self.rgb_encoder(rgb)
        depth_feat   = self.depth_encoder(depth)   if depth   is not None \
                       else torch.zeros(B, self.feat_dim, device=rgb.device)
        ir_feat      = self.ir_encoder(ir)         if ir      is not None \
                       else torch.zeros(B, self.feat_dim, device=rgb.device)
        thermal_feat = self.thermal_encoder(thermal) if thermal is not None \
                       else torch.zeros(B, self.feat_dim, device=rgb.device)
        audio_feat   = self.audio_encoder(audio)   if audio   is not None \
                       else torch.zeros(B, self.feat_dim, device=rgb.device)
        lip_feat     = self.lip_encoder(lips)       if lips    is not None \
                       else torch.zeros(B, self.feat_dim, device=rgb.device)

        # ── Step 2: Cross-modal attention (RGB queries other modalities) ─────
        if depth is not None:
            rgb_feat = self.rgb_depth_attn(rgb_feat, depth_feat)
        if ir is not None:
            rgb_feat = self.rgb_ir_attn(rgb_feat, ir_feat)
        if thermal is not None:
            rgb_feat = self.rgb_thermal_attn(rgb_feat, thermal_feat)

        feats = [rgb_feat, depth_feat, ir_feat,
                 thermal_feat, audio_feat, lip_feat]

        # ── Step 3: Audio-Visual Synchrony ───────────────────────────────────
        av_sync_out = {}
        if audio is not None and lips is not None:
            av_sync_out = self.av_sync(audio_feat, lip_feat)

        # ── Step 4: Modality dropout (training robustness) ───────────────────
        availability = self._modality_dropout(availability)

        # ── Step 5: Adaptive fusion weights ──────────────────────────────────
        weights = self.fusion_gate(feats, availability)  # [B, M]

        # ── Step 6: Transformer fusion ────────────────────────────────────────
        # Build padding mask: True = ignore (unavailable modality)
        pad_mask = (availability < 0.5)                  # [B, M] bool
        tf_out   = self.transformer_fusion(
            feats, weights, key_padding_mask=pad_mask
        )
        fused_emb = tf_out["embedding"]                  # [B, feat_dim]

        # ── Step 7: Identity embedding ────────────────────────────────────────
        embedding = self.identity_head(fused_emb)        # [B, emb_dim]

        # ── Step 8: ArcFace classification head ───────────────────────────────
        arcface_out = self.arcface_head(embedding, labels)

        # ── Build output dict ─────────────────────────────────────────────────
        out = {
            "embedding": embedding,               # [B, emb_dim]
            "logit"    : arcface_out["logit"],    # [B, num_classes]
            "cosine"   : arcface_out["cosine"],   # [B, num_classes]
            "weights"  : weights,                 # [B, M]
        }

        if av_sync_out:
            out["sync_prob"]  = av_sync_out["sync_prob"]
            out["sync_logit"] = av_sync_out["sync_logit"]

        if return_all:
            out["rgb_feat"]      = rgb_feat
            out["depth_feat"]    = depth_feat
            out["ir_feat"]       = ir_feat
            out["thermal_feat"]  = thermal_feat
            out["audio_feat"]    = audio_feat
            out["lip_feat"]      = lip_feat
            out["token_outputs"] = tf_out["token_outputs"]

        return out


# ── Loss ──────────────────────────────────────────────────────────────────────

class MultiModalFusionLoss(nn.Module):
    """
    Combined multi-modal fusion loss:

        L = α · ArcFace(logit, labels)
          + β · AVSync(sync_logit, sync_labels)
          + γ · ModalityContrastive(embeddings per modality)
          + δ · FusionRegularization(weights entropy)

    ArcFace          → discriminative identity classification
    AVSync           → audio-visual consistency
    Contrastive      → pull same-identity embeddings together
    FusionReg        → prevent fusion gate collapsing to one modality
    """

    def __init__(
        self,
        alpha: float = 1.0,    # ArcFace
        beta : float = 0.3,    # AV Sync
        gamma: float = 0.2,    # Contrastive
        delta: float = 0.1,    # Fusion regularization
    ):
        super().__init__()
        self.alpha = alpha
        self.beta  = beta
        self.gamma = gamma
        self.delta = delta
        self.ce    = nn.CrossEntropyLoss()
        self.bce   = nn.BCEWithLogitsLoss()

    def _contrastive_loss(
        self,
        emb1  : torch.Tensor,   # [B, emb_dim]
        emb2  : torch.Tensor,   # [B, emb_dim]
        labels: torch.Tensor,   # [B]  same identity = 1, different = 0
        margin: float = 0.5,
    ) -> torch.Tensor:
        """Contrastive loss between two modality embeddings."""
        dist = 1.0 - F.cosine_similarity(emb1, emb2, dim=-1)  # [B]
        pos  = labels       * dist.pow(2)
        neg  = (1 - labels) * F.relu(margin - dist).pow(2)
        return (pos + neg).mean()

    def _fusion_entropy_reg(
        self,
        weights: torch.Tensor,   # [B, M]
    ) -> torch.Tensor:
        """
        Entropy regularization on fusion weights.
        Encourages balanced use of all modalities
        rather than collapsing to a single one.
        Maximize entropy → penalize low entropy (peaked weights).
        """
        entropy = -(weights * (weights + 1e-8).log()).sum(dim=1).mean()
        return -entropy   # minimize negative entropy = maximize entropy

    def forward(
        self,
        logit       : torch.Tensor,            # [B, num_classes]
        labels      : torch.Tensor,            # [B]
        weights     : torch.Tensor,            # [B, M]
        sync_logit  : Optional[torch.Tensor],  # [B, 1]
        sync_labels : Optional[torch.Tensor],  # [B]  1=in-sync
        rgb_emb     : Optional[torch.Tensor],  # [B, emb_dim]
        aux_emb     : Optional[torch.Tensor],  # [B, emb_dim]  e.g. IR embedding
        pair_labels : Optional[torch.Tensor],  # [B]  same identity?
    ) -> Tuple[torch.Tensor, Dict[str, float]]:

        # ArcFace cross-entropy
        ce_loss  = self.ce(logit, labels)
        total    = self.alpha * ce_loss
        metrics  = {"ce_loss": ce_loss.item()}

        # AV Sync loss
        if sync_logit is not None and sync_labels is not None:
            sync_loss = self.bce(
                sync_logit.squeeze(1),
                sync_labels.float(),
            )
            total   += self.beta * sync_loss
            metrics["sync_loss"] = sync_loss.item()

        # Cross-modality contrastive
        if rgb_emb is not None and aux_emb is not None \
                and pair_labels is not None:
            cont_loss = self._contrastive_loss(rgb_emb, aux_emb, pair_labels)
            total    += self.gamma * cont_loss
            metrics["contrastive_loss"] = cont_loss.item()

        # Fusion regularization
        fusion_reg = self._fusion_entropy_reg(weights)
        total     += self.delta * fusion_reg
        metrics["fusion_reg"] = fusion_reg.item()

        metrics["total_loss"] = total.item()
        return total, metrics


# ── Fusion Gate (pipeline integration) ───────────────────────────────────────

class MultiModalFusionGate(nn.Module):
    """
    Drop-in multi-modal fusion gate for the face recognition pipeline.

    Full pipeline slot order:
        DenoiserGate
            → SuperResolutionGate
            → LivenessGate
            → AntiSpoofingGate
            → MultiModalFusionGate    ← HERE
            → Identity DB lookup

    Accepts any combination of available modalities,
    gracefully handles missing ones via zero-padding + availability mask.

    Usage:
        gate   = MultiModalFusionGate(weights_path="fusion.pt")
        result = gate(rgb=face_crop, depth=depth_map, ir=ir_img)
        emb    = result["embedding"]   # use for DB lookup
    """

    def __init__(
        self,
        weights_path  : Optional[str] = None,
        feat_dim      : int   = 256,
        emb_dim       : int   = 512,
        num_classes   : int   = 10000,
        device        : str   = "cpu",
    ):
        super().__init__()
        self.device = torch.device(device)
        self.model  = MultiModalFusion(
            feat_dim=feat_dim,
            emb_dim=emb_dim,
            num_classes=num_classes,
        ).to(self.device)
        self.model.eval()

        if weights_path is not None:
            state = torch.load(weights_path, map_location=self.device)
            self.model.load_state_dict(state)
            print(f"[MultiModalFusionGate] Loaded weights from: {weights_path}")
        else:
            print("[MultiModalFusionGate] ⚠️  No weights loaded — random init.")

    def _to(self, t: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
        return t.to(self.device) if t is not None else None

    @torch.no_grad()
    def forward(
        self,
        rgb     : torch.Tensor,
        depth   : Optional[torch.Tensor] = None,
        ir      : Optional[torch.Tensor] = None,
        thermal : Optional[torch.Tensor] = None,
        audio   : Optional[torch.Tensor] = None,
        lips    : Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Returns:
            dict:
                'embedding' : [B, emb_dim]   identity embedding
                'weights'   : [B, M]         modality fusion weights
                'sync_prob' : [B, 1]         AV sync (if audio+lips)
        """
        return self.model(
            rgb     = self._to(rgb),
            depth   = self._to(depth),
            ir      = self._to(ir),
            thermal = self._to(thermal),
            audio   = self._to(audio),
            lips    = self._to(lips),
        )
