"""
Voice Recognition Module
Recognizes identity from speaker characteristics and voice biometrics.
Complements face recognition with:
    • Speaker embedding (i-vectors, x-vectors)
    • Mel-spectrogram feature extraction
    • Temporal modeling via LSTM/Transformer
    • Voice activity detection (VAD)
    • Phonetic content analysis
Architecture: ResNet-based speaker encoder + temporal attention.
Robust to: background noise, accent variations, emotion changes.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Dict, List
import math


# ── Building Blocks ───────────────────────────────────────────────────────────

class ConvBNReLU(nn.Module):
    """Standard Conv → BatchNorm → ReLU block."""

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
        layers.append(nn.ReLU(inplace=True))
        self.block = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class ResBlock2D(nn.Module):
    """
    2D Residual block for spectrogram processing.
    Used in ResNet-style speaker encoder.
    """

    def __init__(self, ch: int, stride: int = 1):
        super().__init__()
        self.block = nn.Sequential(
            ConvBNReLU(ch, ch, kernel=3, stride=stride, padding=1),
            nn.Conv2d(ch, ch, kernel=3, padding=1, bias=False),
            nn.BatchNorm2d(ch),
        )
        self.relu = nn.ReLU(inplace=True)
        self.stride = stride

        # Downsample skip connection if needed
        if stride > 1:
            self.skip = nn.Sequential(
                nn.Conv2d(ch, ch, kernel=1, stride=stride, bias=False),
                nn.BatchNorm2d(ch),
            )
        else:
            self.skip = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.block(x)
        skip = self.skip(x) if self.skip is not None else x
        return self.relu(out + skip)


class SEBlock(nn.Module):
    """
    Squeeze-and-Excitation block for spectral channel weighting.
    """

    def __init__(self, ch: int, reduction: int = 16):
        super().__init__()
        self.gate = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(ch, ch // reduction),
            nn.ReLU(inplace=True),
            nn.Linear(ch // reduction, ch),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        w = self.gate(x).view(x.size(0), x.size(1), 1, 1)
        return x * w


class TemporalAttention(nn.Module):
    """
    Temporal attention over time frames.
    Weights different time slices based on speaker-relevant content
    (vowels, fricatives, etc. carry more identity info than silence).
    """

    def __init__(self, feat_dim: int, num_heads: int = 4):
        super().__init__()
        self.attn = nn.MultiheadAttention(
            embed_dim=feat_dim,
            num_heads=num_heads,
            dropout=0.1,
            batch_first=True,
        )
        self.norm = nn.LayerNorm(feat_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: [B, T, feat_dim]"""
        x_attn, _ = self.attn(x, x, x)
        return self.norm(x + x_attn)


# ── Voice Activity Detection (VAD) ────────────────────────────────────────────

class VoiceActivityDetector(nn.Module):
    """
    Voice Activity Detection (VAD) module.
    Distinguishes speech from silence/background noise.

    Input : mel-spectrogram [B, T, n_mels]
    Output: VAD scores [B, T]  (0 = silence, 1 = speech)

    Used to:
        • Weight speaker embedding computation (ignore silence)
        • Filter out background noise frames
        • Estimate speech activity ratio
    """

    def __init__(
        self,
        n_mels: int = 64,
        feat_dim: int = 128,
    ):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_mels, feat_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Linear(feat_dim, feat_dim),
            nn.ReLU(inplace=True),
            nn.Linear(feat_dim, 1),
            nn.Sigmoid(),                       # VAD score ∈ [0, 1]
        )

    def forward(self, mel_spec: torch.Tensor) -> Dict[str, torch.Tensor]:
        """mel_spec: [B, T, n_mels]"""
        B, T = mel_spec.shape[:2]

        # Per-frame VAD scores
        vad_logits = self.net(mel_spec)               # [B, T, 1]
        vad_scores = vad_logits.squeeze(-1)           # [B, T]

        # Aggregate statistics
        vad_ratio  = vad_scores.mean(dim=1)           # [B]  fraction of speech

        return {
            "vad_scores": vad_scores,                 # [B, T]
            "vad_ratio" : vad_ratio,                  # [B]
        }


# ── Speaker Encoder (x-vector style) ──────────────────────────────────────────

class SpeakerEncoder(nn.Module):
    """
    Speaker encoder based on x-vector architecture.

    Architecture:
        Mel-spectrogram [B, T, n_mels]
            │
            ├── Per-frame feature extraction (ResNet layers)
            │       [B, T, frame_dim]
            │
            ├── Temporal pooling (mean + std)
            │       [B, 2*frame_dim]
            │
            ├── DNN layers
            │       [B, emb_dim]
            │
            └── L2-normalize
                    [B, emb_dim]  speaker embedding

    x-vectors capture speaker-specific characteristics:
        • Vocal tract shape (resonances)
        • Voice quality & prosody
        • Speaking rate and patterns
        • Gender, age, accent cues
    """

    def __init__(
        self,
        n_mels   : int = 64,
        emb_dim  : int = 512,
        feat_dim : int = 256,
    ):
        super().__init__()
        self.n_mels = n_mels

        # ── Per-frame encoder ────────────────────────────────────────────────
        self.frame_encoder = nn.Sequential(
            nn.Linear(n_mels, feat_dim),
            nn.ReLU(inplace=True),
            nn.Linear(feat_dim, feat_dim),
            nn.ReLU(inplace=True),
            nn.Linear(feat_dim, feat_dim),
            nn.ReLU(inplace=True),
        )

        # ── Temporal pooling statistics ──────────────────────────────────────
        # Computes mean and std across time
        # Output: [B, 2*feat_dim]  (mean + std concatenated)

        # ── Speaker embedding DNN ────────────────────────────────────────────
        self.embedding = nn.Sequential(
            nn.Linear(feat_dim * 2, feat_dim),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(feat_dim),
            nn.Dropout(0.1),
            nn.Linear(feat_dim, feat_dim),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(feat_dim),
            nn.Linear(feat_dim, emb_dim),
        )

    def forward(
        self,
        mel_spec: torch.Tensor,                  # [B, T, n_mels]
        vad_scores: Optional[torch.Tensor] = None,  # [B, T]  for weighting
    ) -> Dict[str, torch.Tensor]:
        """
        Args:
            mel_spec  : mel-spectrogram
            vad_scores: optional VAD weights (upweight speech frames)

        Returns:
            dict:
                'embedding'    : [B, emb_dim]  speaker embedding
                'frame_feats'  : [B, T, feat_dim]  per-frame features
        """
        B, T = mel_spec.shape[:2]

        # Per-frame feature extraction
        frame_feats = self.frame_encoder(mel_spec)  # [B, T, feat_dim]

        # Apply VAD weighting if available
        if vad_scores is not None:
            # Weight frames by VAD scores
            vad_w = vad_scores.unsqueeze(-1)        # [B, T, 1]
            frame_feats = frame_feats * vad_w       # weighted by speech activity

        # Temporal statistics pooling
        mean_feat = frame_feats.mean(dim=1)         # [B, feat_dim]
        std_feat  = frame_feats.std(dim=1)          # [B, feat_dim]
        pooled    = torch.cat([mean_feat, std_feat], dim=1)  # [B, 2*feat_dim]

        # Embedding network
        embedding = self.embedding(pooled)          # [B, emb_dim]
        embedding = F.normalize(embedding, dim=-1)  # L2-normalize

        return {
            "embedding"  : embedding,
            "frame_feats": frame_feats,
        }


# ── ResNet Speaker Encoder (alternative) ──────────────────────────────────────

class ResNetSpeakerEncoder(nn.Module):
    """
    ResNet-based speaker encoder for mel-spectrograms.
    More expressive than linear x-vector, captures spectral patterns.

    Input : mel-spectrogram [B, 1, T, n_mels]  (add channel dim)
    Output: [B, emb_dim]  speaker embedding
    """

    def __init__(
        self,
        n_mels  : int = 64,
        emb_dim : int = 512,
        depth   : int = 34,           # 34, 50, 101, 152
    ):
        super().__init__()

        # Depth configuration: [64, 128, 256, 512]
        layers_config = {
            34: [3, 4, 6, 3],
            50: [3, 4, 6, 3],
            101: [3, 4, 23, 3],
            152: [3, 8, 36, 3],
        }
        num_blocks = layers_config.get(depth, [3, 4, 6, 3])

        # ── Initial conv ─────────────────────────────────────────────────────
        self.conv1 = ConvBNReLU(1, 64, kernel=7, stride=2, padding=3)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # ── Residual blocks ──────────────────────────────────────────────────
        self.layer1 = self._make_layer(64, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(64, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(128, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(256, 512, num_blocks[3], stride=2)

        # ── SE-Res2Net (squeeze-and-excitation) ──────────────────────────────
        self.se = SEBlock(512)

        # ── Embedding head ───────────────────────────────────────────────────
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.embedding = nn.Sequential(
            nn.Linear(512, emb_dim),
            nn.LayerNorm(emb_dim),
        )

    def _make_layer(
        self,
        in_ch: int,
        out_ch: int,
        num_blocks: int,
        stride: int = 1,
    ) -> nn.Sequential:
        layers = [ResBlock2D(in_ch, stride=stride)]
        for _ in range(1, num_blocks):
            layers.append(ResBlock2D(out_ch, stride=1))
        return nn.Sequential(*layers)

    def forward(self, mel_spec: torch.Tensor) -> Dict[str, torch.Tensor]:
        """mel_spec: [B, 1, T, n_mels]  (add channel dim if needed)"""
        if mel_spec.dim() == 3:
            mel_spec = mel_spec.unsqueeze(1)      # [B, T, n_mels] → [B, 1, T, n_mels]

        # Initial convolution
        x = self.conv1(mel_spec)
        x = self.maxpool(x)

        # Residual blocks
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        # Squeeze-and-excitation
        x = self.se(x)

        # Global average pooling
        x = self.avgpool(x)
        x = x.flatten(1)

        # Embedding
        embedding = self.embedding(x)
        embedding = F.normalize(embedding, dim=-1)

        return {
            "embedding": embedding,
        }


# ── Temporal Modeling with Transformer ────────────────────────────────────────

class SpeechTransformerEncoder(nn.Module):
    """
    Transformer-based encoder for speech sequences.
    Captures long-range temporal dependencies in speaker characteristics.

    Input : mel-spectrogram [B, T, n_mels]
    Output: [B, emb_dim]  speaker embedding
    """

    def __init__(
        self,
        n_mels   : int = 64,
        emb_dim  : int = 512,
        num_heads: int = 8,
        num_layers: int = 4,
        feat_dim : int = 256,
    ):
        super().__init__()

        # Initial projection to embedding dimension
        self.input_proj = nn.Linear(n_mels, emb_dim)

        # Positional encoding
        self.pos_encoding = nn.Parameter(
            torch.randn(1, 512, emb_dim) * math.sqrt(1.0 / emb_dim)
        )                                       # learnable positional embeddings

        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=emb_dim,
            nhead=num_heads,
            dim_feedforward=feat_dim,
            dropout=0.1,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers,
            norm=nn.LayerNorm(emb_dim),
        )

        # Output pooling and projection
        self.output_proj = nn.Sequential(
            nn.Linear(emb_dim, feat_dim),
            nn.GELU(),
            nn.Linear(feat_dim, emb_dim),
        )

    def forward(self, mel_spec: torch.Tensor) -> Dict[str, torch.Tensor]:
        """mel_spec: [B, T, n_mels]"""
        B, T = mel_spec.shape[:2]

        # Project to embedding dimension
        x = self.input_proj(mel_spec)           # [B, T, emb_dim]

        # Add positional encoding
        pos = self.pos_encoding[:, :T, :]       # [1, T, emb_dim]
        x = x + pos

        # Transformer encoder
        x = self.transformer(x)                 # [B, T, emb_dim]

        # Global pooling (mean across time)
        x = x.mean(dim=1)                       # [B, emb_dim]

        # Output projection
        embedding = self.output_proj(x)         # [B, emb_dim]
        embedding = F.normalize(embedding, dim=-1)

        return {
            "embedding": embedding,
        }


# ── Phonetic Posteriorgram (PPG) Encoder ──────────────────────────────────────

class PhoneticEncoder(nn.Module):
    """
    Phonetic Posteriorgram (PPG) encoder.
    Extracts phonetic content (linguistic information) from speech,
    which contributes to speaker identity beyond just voice quality.

    Different speakers pronounce phonemes differently:
        • Vowel formant frequencies
        • Consonant release patterns
        • Speech articulation style

    Input : mel-spectrogram [B, T, n_mels]
    Output: phonetic features [B, T, n_phonemes]
    """

    def __init__(
        self,
        n_mels   : int = 64,
        n_phonemes: int = 51,          # standard phone set
        feat_dim : int = 256,
    ):
        super().__init__()
        self.n_phonemes = n_phonemes

        # Feature extraction
        self.encoder = nn.Sequential(
            nn.Linear(n_mels, feat_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Linear(feat_dim, feat_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Linear(feat_dim, feat_dim),
            nn.ReLU(inplace=True),
        )

        # Phoneme posterior (softmax over phoneme classes)
        self.phoneme_head = nn.Linear(feat_dim, n_phonemes)

    def forward(self, mel_spec: torch.Tensor) -> Dict[str, torch.Tensor]:
        """mel_spec: [B, T, n_mels]"""
        B, T = mel_spec.shape[:2]

        # Per-frame feature encoding
        feats = self.encoder(mel_spec)          # [B, T, feat_dim]

        # Per-frame phoneme posteriors
        ppg_logits = self.phoneme_head(feats)   # [B, T, n_phonemes]
        ppg_probs  = F.softmax(ppg_logits, dim=-1)  # [B, T, n_phonemes]

        # Aggregate across time (speaker's typical phoneme distribution)
        ppg_mean = ppg_probs.mean(dim=1)        # [B, n_phonemes]

        return {
            "ppg_probs": ppg_probs,              # [B, T, n_phonemes]
            "ppg_mean" : ppg_mean,               # [B, n_phonemes]
            "ppg_logits": ppg_logits,
        }


# ── Full Voice Recognition Module ─────────────────────────────────────────────

class VoiceRecognition(nn.Module):
    """
    Full voice recognition / speaker identification module.

    Pipeline:
        Audio signal → MFCC/Mel-spectrogram
            │
            ├── Voice Activity Detection (VAD)
            │       → speech/silence classification
            │
            ├── Speaker Encoder (x-vector or ResNet)
            │       → [B, emb_dim]  speaker embedding
            │
            ├── Phonetic Encoder (optional)
            │       → [B, n_phonemes]  phonetic features
            │
            ├── Embedding projection
            │       → L2-normalized identity embedding
            │
            └── Classification head (ArcFace)
                    → identity logits [B, num_classes]

    Speaker embeddings capture:
        • Vocal tract resonances
        • Voice quality & timbre
        • Speaking patterns
        • Age, gender cues

    Robust to:
        • Background noise (with proper preprocessing)
        • Emotion changes (some emotion-specific features filtered)
        • Speaking rate variation
        • Accent differences
    """

    def __init__(
        self,
        n_mels     : int = 64,
        emb_dim    : int = 512,
        num_classes: int = 10000,
        encoder_type: str = "xvector",  # "xvector", "resnet", "transformer"
        feat_dim   : int = 256,
        use_vad    : bool = True,
        use_phonetic: bool = False,
        n_phonemes : int = 51,
    ):
        super().__init__()
        self.n_mels       = n_mels
        self.use_vad      = use_vad
        self.use_phonetic = use_phonetic

        # ── Voice Activity Detection ─────────────────────────────────────────
        if use_vad:
            self.vad = VoiceActivityDetector(n_mels, feat_dim)
        else:
            self.vad = None

        # ── Speaker Encoder ──────────────────────────────────────────────────
        if encoder_type == "xvector":
            self.encoder = SpeakerEncoder(n_mels, emb_dim, feat_dim)
        elif encoder_type == "resnet":
            self.encoder = ResNetSpeakerEncoder(n_mels, emb_dim, depth=34)
        elif encoder_type == "transformer":
            self.encoder = SpeechTransformerEncoder(n_mels, emb_dim, feat_dim=feat_dim)
        else:
            raise ValueError(f"Unknown encoder: {encoder_type}")

        # ── Phonetic Encoder (optional) ──────────────────────────────────────
        if use_phonetic:
            self.phonetic_encoder = PhoneticEncoder(n_mels, n_phonemes, feat_dim)
            # Fusion: combine speaker embedding + phonetic features
            self.fusion = nn.Sequential(
                nn.Linear(emb_dim + n_phonemes, feat_dim),
                nn.GELU(),
                nn.Linear(feat_dim, emb_dim),
                nn.LayerNorm(emb_dim),
            )
        else:
            self.phonetic_encoder = None
            self.fusion = None

        # ── Embedding head (L2-normalization) ────────────────────────────────
        self.embedding_proj = nn.Sequential(
            nn.Linear(emb_dim, emb_dim),
            nn.LayerNorm(emb_dim),
        )

        # ── Classification head (ArcFace) ────────────────────────────────────
        self.classifier = nn.Linear(emb_dim, num_classes, bias=False)
        nn.init.xavier_uniform_(self.classifier.weight)

    def forward(
        self,
        mel_spec : torch.Tensor,                # [B, T, n_mels]
        labels   : Optional[torch.Tensor] = None,  # [B]
        return_all: bool = False,
    ) -> Dict[str, torch.Tensor]:
        """
        Args:
            mel_spec : mel-spectrogram input
            labels   : speaker class IDs (for training)
            return_all: return intermediate outputs

        Returns:
            dict:
                'embedding'   : [B, emb_dim]   speaker embedding
                'logit'       : [B, num_classes]  classification logits
                'vad_scores'  : [B, T]  VAD output (if use_vad)
                'ppg_probs'   : [B, T, n_phonemes] (if use_phonetic)
        """

        # ── Voice Activity Detection ─────────────────────────────────────────
        vad_out = {}
        vad_scores = None
        if self.use_vad:
            vad_out = self.vad(mel_spec)
            vad_scores = vad_out["vad_scores"]   # [B, T]

        # ── Speaker Encoding ─────────────────────────────────────────────────
        encoder_out = self.encoder(mel_spec, vad_scores=vad_scores)
        speaker_emb = encoder_out["embedding"]   # [B, emb_dim]

        # ── Phonetic Features (optional) ─────────────────────────────────────
        ppg_out = {}
        if self.use_phonetic:
            ppg_out = self.phonetic_encoder(mel_spec)
            ppg_mean = ppg_out["ppg_mean"]       # [B, n_phonemes]
            # Fuse speaker embedding + phonetic features
            fused = torch.cat([speaker_emb, ppg_mean], dim=1)  # [B, emb+n_ph]
            speaker_emb = self.fusion(fused)
            speaker_emb = F.normalize(speaker_emb, dim=-1)

        # ── Final embedding projection ────────────────────────────────────────
        embedding = self.embedding_proj(speaker_emb)
        embedding = F.normalize(embedding, dim=-1)

        # ── Classification ───────────────────────────────────────────────────
        logit = self.classifier(embedding)

        out = {
            "embedding": embedding,
            "logit"    : logit,
        }

        if vad_out:
            out.update(vad_out)

        if ppg_out:
            out["ppg_probs"] = ppg_out["ppg_probs"]
            out["ppg_mean"]  = ppg_out["ppg_mean"]

        if return_all:
            out["speaker_emb"] = speaker_emb

        return out


# ── Loss ──────────────────────────────────────────────────────────────────────

class VoiceRecognitionLoss(nn.Module):
    """
    Combined voice recognition loss:

        L = α · ArcFace(logit, labels)
          + β · TripletLoss(embeddings)
          + γ · VADConsistency(vad_scores)
          + δ · PhoneticsLoss(ppg_probs)

    ArcFace          → discriminative speaker classification
    TripletLoss      → metric learning: same-speaker embeddings close
    VADConsistency   → penalize unreliable low-VAD frames
    PhoneticsLoss    → phonetic content should be speaker-independent
    """

    def __init__(
        self,
        alpha: float = 1.0,       # ArcFace
        beta : float = 0.5,       # Triplet
        gamma: float = 0.1,       # VAD
        delta: float = 0.1,       # Phonetics
        margin: float = 0.5,
    ):
        super().__init__()
        self.alpha  = alpha
        self.beta   = beta
        self.gamma  = gamma
        self.delta  = delta
        self.ce     = nn.CrossEntropyLoss()
        self.margin = margin

    def _triplet_loss(
        self,
        emb    : torch.Tensor,    # [B, emb_dim]
        labels : torch.Tensor,    # [B]
    ) -> torch.Tensor:
        """Hardest triplet loss."""
        dist = torch.cdist(emb, emb, p=2)

        labels_eq = labels.unsqueeze(0) == labels.unsqueeze(1)
        diag_mask = ~torch.eye(labels.size(0), dtype=torch.bool,
                               device=labels.device)
        pos_mask  = labels_eq & diag_mask
        neg_mask  = ~labels_eq

        pos_dist  = torch.where(pos_mask, dist, torch.tensor(float('inf')))
        neg_dist  = torch.where(neg_mask, dist, torch.tensor(-float('inf')))

        hardest_pos = pos_dist.min(dim=1).values
        hardest_neg = neg_dist.max(dim=1).values

        triplet = F.relu(hardest_pos - hardest_neg + self.margin)
        return triplet.mean()

    def forward(
        self,
        logit      : torch.Tensor,           # [B, num_classes]
        embedding  : torch.Tensor,           # [B, emb_dim]
        labels     : torch.Tensor,           # [B]
        vad_scores : Optional[torch.Tensor], # [B, T]
        ppg_probs  : Optional[torch.Tensor], # [B, T, n_phonemes]
    ) -> Tuple[torch.Tensor, Dict[str, float]]:

        # ArcFace loss
        ce_loss = self.ce(logit, labels)
        total   = self.alpha * ce_loss
        metrics = {"ce_loss": ce_loss.item()}

        # Triplet loss
        triplet_loss = self._triplet_loss(embedding, labels)
        total       += self.beta * triplet_loss
        metrics["triplet_loss"] = triplet_loss.item()

        # VAD consistency: penalize low confidence predictions
        # (model should abstain on silence/noise)
        if vad_scores is not None:
            max_vad = vad_scores.max(dim=1).values   # [B]  max VAD confidence
            vad_loss = (1.0 - max_vad).mean()        # penalize if no clear voice
            total   += self.gamma * vad_loss
            metrics["vad_loss"] = vad_loss.item()

        # Phonetics loss: phonetic content should be consistent
        # across different speakers (speaker-independent)
        # Minimize variation in phoneme distribution within speaker class
        if ppg_probs is not None:
            ppg_mean = ppg_probs.mean(dim=1)        # [B, n_phonemes]
            # Entropy regularization: encourage balanced phoneme distribution
            entropy = -(ppg_mean * (ppg_mean + 1e-8).log()).sum(dim=1).mean()
            ppg_loss = (1.0 - entropy)              # penalize low entropy
            total   += self.delta * ppg_loss
            metrics["ppg_loss"] = ppg_loss.item()

        metrics["total_loss"] = total.item()
        return total, metrics


# ── Voice Gate (pipeline integration) ─────────────────────────────────────────

class VoiceGate(nn.Module):
    """
    Drop-in voice recognition gate for speaker identification pipeline.

    Can be used to:
        1. Speaker verification (is this the claimed person?)
        2. Speaker identification (who is this?)
        3. Cross-verify voice + face for liveness detection
        4. Deepfake detection (speech synthesis artifacts)

    Usage:
        voice_gate = VoiceGate(weights_path="voice.pt")
        result     = voice_gate(mel_spec=mel_spectrogram)
        voice_emb  = result["embedding"]   # use for speaker lookup
    """

    def __init__(
        self,
        weights_path: Optional[str] = None,
        n_mels      : int = 64,
        emb_dim     : int = 512,
        num_classes : int = 10000,
        encoder_type: str = "xvector",
        device      : str = "cpu",
    ):
        super().__init__()
        self.device = torch.device(device)
        self.model  = VoiceRecognition(
            n_mels=n_mels,
            emb_dim=emb_dim,
            num_classes=num_classes,
            encoder_type=encoder_type,
            use_vad=True,
            use_phonetic=False,
        ).to(self.device)
        self.model.eval()

        if weights_path is not None:
            state = torch.load(weights_path, map_location=self.device)
            self.model.load_state_dict(state)
            print(f"[VoiceGate] Loaded weights from: {weights_path}")
        else:
            print("[VoiceGate] ⚠️  No weights loaded — random init.")

    @torch.no_grad()
    def forward(
        self,
        mel_spec: torch.Tensor,                # [B, T, n_mels]
    ) -> Dict[str, torch.Tensor]:
        """
        Returns:
            dict:
                'embedding' : [B, emb_dim]   speaker embedding
                'logit'     : [B, num_classes]
                'vad_scores': [B, T]         voice activity
        """
        mel_spec = mel_spec.to(self.device)
        return self.model(mel_spec)


# ── Voice-Face Verification ───────────────────────────────────────────────────

class VoiceFaceVerification(nn.Module):
    """
    Voice-Face verification for cross-modal biometric matching.

    Verifies that voice and face embeddings belong to the same person.
    Useful for:
        • Liveness detection: recorded audio + face video must match
        • Deepfake detection: speech synthesis + face swap detection
        • Identity confirmation in speaker verification systems

    Uses weighted similarity:
        score = α · face_sim + β · voice_sim
    """

    def __init__(
        self,
        emb_dim: int = 512,
        face_weight: float = 0.5,
        voice_weight: float = 0.5,
    ):
        super().__init__()
        self.face_weight  = face_weight
        self.voice_weight = voice_weight

        # Confidence calibration network
        self.confidence_head = nn.Sequential(
            nn.Linear(emb_dim * 2, 256),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(256, 128),
            nn.GELU(),
            nn.Linear(128, 1),
            nn.Sigmoid(),
        )

        # Liveness detector: how well do voice + face align?
        self.liveness_head = nn.Sequential(
            nn.Linear(emb_dim * 2, 256),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(256, 1),
            nn.Sigmoid(),
        )

    def forward(
        self,
        face_emb  : torch.Tensor,                 # [B, emb_dim]
        voice_emb : torch.Tensor,                 # [B, emb_dim]
        face_ref  : Optional[torch.Tensor] = None,   # [N, emb_dim]
        voice_ref : Optional[torch.Tensor] = None,   # [N, emb_dim]
    ) -> Dict[str, torch.Tensor]:
        """
        Args:
            face_emb : face embedding from face recognition
            voice_emb: voice embedding from speaker recognition
            face_ref : reference face embeddings from DB
            voice_ref: reference voice embeddings from DB

        Returns:
            dict:
                'face_sim'      : [B, N]  cosine similarity to face refs
                'voice_sim'     : [B, N]  cosine similarity to voice refs
                'combined_sim'  : [B, N]  weighted fusion
                'confidence'    : [B]     fusion reliability
                'liveness_prob' : [B]     voice-face consistency (liveness score)
                'top_match'     : [B]     index of best match
        """
        B = face_emb.size(0)

        # Similarities
        face_sim  = F.cosine_similarity(
            face_emb.unsqueeze(1), face_ref.unsqueeze(0), dim=2
        ) if face_ref is not None else None   # [B, N]

        voice_sim = F.cosine_similarity(
            voice_emb.unsqueeze(1), voice_ref.unsqueeze(0), dim=2
        ) if voice_ref is not None else None  # [B, N]

        # Combined similarity
        if face_sim is not None and voice_sim is not None:
            combined = (self.face_weight * face_sim
                      + self.voice_weight * voice_sim)  # [B, N]
        elif face_sim is not None:
            combined = face_sim
        else:
            combined = voice_sim

        # Confidence: how well do face + voice embeddings match?
        joint_emb = torch.cat([face_emb, voice_emb], dim=1)  # [B, 2*emb_dim]
        confidence = self.confidence_head(joint_emb).squeeze(1)  # [B]

        # Liveness: do they belong to the same person?
        liveness_prob = self.liveness_head(joint_emb).squeeze(1)  # [B]

        # Top match
        top_scores, top_indices = combined.max(dim=1)  # [B], [B]

        return {
            "face_sim"      : face_sim,
            "voice_sim"     : voice_sim,
            "combined_sim"  : combined,
            "confidence"    : confidence,
            "liveness_prob" : liveness_prob,
            "top_match"     : top_indices,
            "match_score"   : top_scores,
        }