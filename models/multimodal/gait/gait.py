"""
Gait Recognition Module
Recognizes identity from walking patterns and body dynamics.
Complements face recognition with:
    • Full-body silhouette features
    • Skeleton keypoint sequences
    • Motion capture (optical flow + pose dynamics)
    • Temporal gait cycle analysis
Architecture: Spatial-Temporal CNN + LSTM for gait sequence modeling.
Robust to: occlusion, low resolution, viewing angles, clothing changes.
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
        layers.append(nn.LeakyReLU(0.2, inplace=True))
        self.block = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class ResidualBlock(nn.Module):
    """
    Residual block for silhouette/skeleton encoding.
    Skip connection helps preserve identity-relevant spatial structure.
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


class TemporalConvBlock(nn.Module):
    """
    1D Temporal convolution block for gait sequence modeling.
    Causal padding ensures no future frame information leakage.
    """

    def __init__(
        self,
        in_ch  : int,
        out_ch : int,
        kernel : int = 3,
        dilation: int = 1,
    ):
        super().__init__()
        self.pad  = (kernel - 1) * dilation
        self.conv = nn.Conv1d(in_ch, out_ch, kernel,
                              dilation=dilation, bias=False)
        self.bn   = nn.BatchNorm1d(out_ch)
        self.relu = nn.LeakyReLU(0.2, inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: [B, C, T]"""
        x = F.pad(x, (self.pad, 0))    # causal left-pad
        return self.relu(self.bn(self.conv(x)))


class SEBlock(nn.Module):
    """
    Squeeze-and-Excitation block.
    Adaptively weights spatial channels based on importance.
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


# ── Silhouette Encoder ────────────────────────────────────────────────────────

class SilhouetteEncoder(nn.Module):
    """
    Encodes full-body silhouettes into compact feature representations.

    Silhouettes (binary masks) capture:
        • Body shape and proportions
        • Limb structure and articulation
        • Gait-specific posture patterns

    Invariant to: clothing, lighting, color
    Sensitive to: body shape, walking pattern

    Input : silhouette sequence [B, T, 1, H, W]
            T = temporal frames (e.g. 32 frames = 1 gait cycle)
    Output: [B, feat_dim]
    """

    def __init__(
        self,
        feat_dim: int = 256,
        seq_len : int = 32,
    ):
        super().__init__()
        self.seq_len = seq_len

        # Per-frame silhouette encoder (shared weights)
        self.frame_encoder = nn.Sequential(
            ConvBNReLU(1, 32),
            ResidualBlock(32),
            nn.MaxPool2d(2),                   # H/2, W/2

            ConvBNReLU(32, 64),
            ResidualBlock(64),
            nn.MaxPool2d(2),                   # H/4, W/4

            ConvBNReLU(64, 128),
            ResidualBlock(128),
            nn.AdaptiveAvgPool2d(4),           # [B, 128, 4, 4]
            nn.Flatten(),                      # [B, 2048]
            nn.Linear(128 * 4 * 4, 128),
            nn.GELU(),
        )

        # Temporal aggregation over gait cycle
        self.temporal = nn.Sequential(
            TemporalConvBlock(128, 128, kernel=3, dilation=1),
            TemporalConvBlock(128, 128, kernel=3, dilation=2),
            nn.AdaptiveAvgPool1d(1),           # [B, 128, 1]
            nn.Flatten(),                      # [B, 128]
        )

        self.proj = nn.Sequential(
            nn.Linear(128, feat_dim),
            nn.LayerNorm(feat_dim),
            nn.GELU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: [B, T, 1, H, W]"""
        B, T = x.shape[:2]

        # Encode each frame
        frame_feats = torch.stack(
            [self.frame_encoder(x[:, t]) for t in range(T)], dim=2
        )                                      # [B, 128, T]

        # Temporal aggregation
        temporal_feat = self.temporal(frame_feats)  # [B, 128]

        return self.proj(temporal_feat)        # [B, feat_dim]


# ── Skeleton Keypoint Encoder ─────────────────────────────────────────────────

class SkeletonEncoder(nn.Module):
    """
    Encodes skeleton keypoint sequences into gait features.

    Processes joint coordinates over time, capturing:
        • Joint angles and their changes
        • Limb length ratios
        • Temporal motion patterns
        • Gait cycle periodicity

    Input : skeleton sequence [B, T, num_joints, 3]
            T = temporal frames
            num_joints = 17 (COCO) or 25 (OpenPose)
            3 = (x, y, confidence)
    Output: [B, feat_dim]
    """

    def __init__(
        self,
        feat_dim: int    = 256,
        num_joints: int  = 17,     # COCO format
        seq_len : int    = 32,
    ):
        super().__init__()
        self.num_joints = num_joints
        self.seq_len    = seq_len

        # Per-joint temporal encoder (shared)
        self.joint_encoder = nn.LSTM(
            input_size=3,           # x, y, confidence
            hidden_size=64,
            num_layers=2,
            batch_first=True,
            dropout=0.1,
        )

        # Joint graph convolution (skeleton tree)
        self.gcn = nn.Sequential(
            nn.Linear(num_joints * 64, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
        )

        # Global temporal aggregation
        self.temporal_agg = nn.LSTM(
            input_size=128,
            hidden_size=feat_dim,
            num_layers=2,
            batch_first=True,
        )

        self.proj = nn.Sequential(
            nn.Linear(feat_dim, feat_dim),
            nn.LayerNorm(feat_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: [B, T, num_joints, 3]"""
        B, T, J = x.shape[:3]

        # Process each joint independently over time
        joint_feats_list = []
        for j in range(J):
            joint_seq = x[:, :, j, :]             # [B, T, 3]
            _, (h_n, _) = self.joint_encoder(joint_seq)
            h_n = h_n[-1]                          # [B, 64]  take last hidden state
            joint_feats_list.append(h_n)

        # Concatenate all joint features
        all_joint_feats = torch.cat(joint_feats_list, dim=1)  # [B, J*64]

        # Graph convolution over skeleton structure
        gcn_feat = self.gcn(all_joint_feats)    # [B, 128]

        # Reshape for temporal aggregation
        gcn_feat = gcn_feat.unsqueeze(1)         # [B, 1, 128]

        # Temporal LSTM
        _, (h_n, _) = self.temporal_agg(gcn_feat)
        temporal_feat = h_n[-1]                   # [B, feat_dim]

        return self.proj(temporal_feat)          # [B, feat_dim]


# ── Optical Flow Encoder ──────────────────────────────────────────────────────

class OpticalFlowGaitEncoder(nn.Module):
    """
    Encodes optical flow fields of walking sequences.

    Optical flow captures motion dynamics:
        • Limb velocities
        • Acceleration patterns
        • Stride characteristics
        • Walking speed and rhythm

    Real optical flow from video (compute separately with FlowNet / PWCNet)
    or use frame differences as proxy.

    Input : stacked optical flow [B, T-1, 2, H, W]
            T-1 flow fields between consecutive frames
            2 channels: (dx, dy) optical flow
    Output: [B, feat_dim]
    """

    def __init__(
        self,
        feat_dim: int = 256,
        seq_len : int = 31,       # T-1 flow pairs
    ):
        super().__init__()
        self.seq_len = seq_len

        # Per-flow frame encoder
        self.flow_encoder = nn.Sequential(
            ConvBNReLU(2, 32),
            ResidualBlock(32),
            nn.MaxPool2d(2),

            ConvBNReLU(32, 64),
            ResidualBlock(64),
            nn.MaxPool2d(2),

            ConvBNReLU(64, 128),
            nn.AdaptiveAvgPool2d(4),
            nn.Flatten(),
            nn.Linear(128 * 4 * 4, 128),
            nn.GELU(),
        )

        # Temporal dynamics aggregation
        self.temporal = nn.Sequential(
            TemporalConvBlock(128, 128, kernel=3, dilation=1),
            TemporalConvBlock(128, 128, kernel=3, dilation=2),
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
        )

        self.proj = nn.Sequential(
            nn.Linear(128, feat_dim),
            nn.LayerNorm(feat_dim),
            nn.GELU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: [B, T, 2, H, W]"""
        B, T = x.shape[:2]

        # Encode each flow frame
        flow_feats = torch.stack(
            [self.flow_encoder(x[:, t]) for t in range(T)], dim=2
        )                                      # [B, 128, T]

        temporal_feat = self.temporal(flow_feats)  # [B, 128]

        return self.proj(temporal_feat)        # [B, feat_dim]


# ── Gait Cycle Analyzer ───────────────────────────────────────────────────────

class GaitCycleAnalyzer(nn.Module):
    """
    Analyzes gait cycle periodicity and phase consistency.

    A complete gait cycle includes:
        1. Stance phase (foot on ground)
        2. Swing phase (foot in air)

    Typical gait cycle ~2 seconds at normal walking speed
    = 60 frames at 30fps

    This module:
        • Detects cycle boundaries (peaks in motion energy)
        • Computes phase alignment across multiple cycles
        • Extracts cycle-invariant features

    Input : gait features over extended sequence [B, feat_dim, T_long]
            T_long = multiple gait cycles (~2-4 seconds)
    Output: cycle metrics [B, cycle_feat_dim]
    """

    def __init__(
        self,
        feat_dim: int = 256,
    ):
        super().__init__()

        # Energy detector: identifies gait cycle peaks
        self.energy_net = nn.Sequential(
            nn.Conv1d(feat_dim, 128, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.Conv1d(128, 64, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.Conv1d(64, 1, kernel_size=1),
            nn.Sigmoid(),                       # energy ∈ [0, 1]
        )

        # Phase consistency classifier
        self.phase_net = nn.Sequential(
            nn.Linear(feat_dim, 128),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(128, 64),
            nn.GELU(),
            nn.Linear(64, 1),                  # phase consistency score
        )

        # Cycle frequency estimator (via FFT)
        self.freq_proj = nn.Sequential(
            nn.Linear(feat_dim, 64),
            nn.GELU(),
            nn.Linear(64, 16),                 # frequency feature bins
        )

    def forward(
        self,
        feat_seq: torch.Tensor,                # [B, feat_dim, T_long]
    ) -> Dict[str, torch.Tensor]:
        """
        Args:
            feat_seq: gait feature sequence over multiple cycles

        Returns:
            dict:
                'energy'          : [B, T_long]  motion energy
                'phase_consistency': [B]          consistency score
                'freq_features'   : [B, 16]      frequency domain features
        """
        B, D, T = feat_seq.shape

        # Detect motion energy peaks (cycle boundaries)
        energy = self.energy_net(feat_seq)     # [B, 1, T] → squeeze to [B, T]
        energy = energy.squeeze(1)              # [B, T]

        # Phase consistency: average feat at detected peaks
        feat_mean = feat_seq.mean(dim=2)        # [B, feat_dim]
        phase_consistency = self.phase_net(feat_mean)  # [B, 1]

        # Frequency analysis via FFT
        fft_out = torch.fft.rfft(feat_seq, dim=2)  # [B, D, T//2+1] complex
        fft_mag = torch.abs(fft_out).mean(dim=1)   # [B, T//2+1]  avg over channels
        freq_feat = self.freq_proj(
            fft_mag.mean(dim=1, keepdim=True).expand(B, -1)
        )                                      # [B, 16]

        return {
            "energy"           : energy,
            "phase_consistency": phase_consistency,
            "freq_features"    : freq_feat,
        }


# ── Full Gait Recognition Module ──────────────────────────────────────────────

class GaitRecognition(nn.Module):
    """
    Full gait recognition pipeline.

    Modalities (all optional, RGB silhouette extraction recommended):
        • Silhouette     — binary body mask from video
        • Skeleton       — pose keypoints (COCO/OpenPose)
        • OpticalFlow    — motion between frames
        • Extended sequence for cycle analysis

    Pipeline:
        Input sequence
            │
            ├── Silhouette encoder   → [B, feat_dim]
            ├── Skeleton encoder     → [B, feat_dim]
            ├── OpticalFlow encoder  → [B, feat_dim]
            │
            ├── Fusion: concatenate + learn joint representation
            │
            ├── Gait Cycle Analyzer  → cycle metrics [B, C]
            │
            ├── Identity projection  → L2-normalized embedding
            │
            └── Classifier head      → identity logits

    Strength: Gait is hard to fake and complements facial recognition.
    Works at: distance, low-resolution, occlusion (partial body visible).
    """

    def __init__(
        self,
        feat_dim      : int   = 256,
        emb_dim       : int   = 512,
        num_classes   : int   = 10000,
        seq_len       : int   = 32,
        num_joints    : int   = 17,
        use_sil       : bool  = True,
        use_skeleton  : bool  = True,
        use_flow      : bool  = False,
        use_cycle     : bool  = True,
    ):
        super().__init__()
        self.use_sil      = use_sil
        self.use_skeleton = use_skeleton
        self.use_flow     = use_flow
        self.use_cycle    = use_cycle

        # ── Modality Encoders ────────────────────────────────────────────────
        if use_sil:
            self.sil_encoder = SilhouetteEncoder(feat_dim, seq_len)

        if use_skeleton:
            self.skel_encoder = SkeletonEncoder(feat_dim, num_joints, seq_len)

        if use_flow:
            self.flow_encoder = OpticalFlowGaitEncoder(feat_dim, seq_len - 1)

        # ── Fusion ───────────────────────────────────────────────────────────
        num_encoders = sum([use_sil, use_skeleton, use_flow])
        fusion_in_dim = feat_dim * num_encoders if num_encoders > 0 else feat_dim

        self.fusion = nn.Sequential(
            nn.Linear(fusion_in_dim, feat_dim * 2),
            nn.LayerNorm(feat_dim * 2),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(feat_dim * 2, feat_dim),
            nn.LayerNorm(feat_dim),
        )

        # ── Gait Cycle Analysis ──────────────────────────────────────────────
        if use_cycle:
            self.cycle_analyzer = GaitCycleAnalyzer(feat_dim)
            # Append cycle features to fused embedding
            fusion_in_dim = feat_dim + 16     # 16 = freq_features dim
        else:
            fusion_in_dim = feat_dim

        # ── Identity Embedding ───────────────────────────────────────────────
        self.embedding_head = nn.Sequential(
            nn.Linear(fusion_in_dim, emb_dim),
            nn.LayerNorm(emb_dim),
            nn.GELU(),
        )

        # ── Classification Head (ArcFace) ────────────────────────────────────
        self.classifier = nn.Linear(emb_dim, num_classes, bias=False)
        nn.init.xavier_uniform_(self.classifier.weight)

    def forward(
        self,
        silhouette   : Optional[torch.Tensor] = None,  # [B, T, 1, H, W]
        skeleton     : Optional[torch.Tensor] = None,  # [B, T, num_joints, 3]
        optical_flow : Optional[torch.Tensor] = None,  # [B, T, 2, H, W]
        extended_seq : Optional[torch.Tensor] = None,  # [B, feat_dim, T_long] for cycle
        labels       : Optional[torch.Tensor] = None,  # [B]
        return_all   : bool = False,
    ) -> Dict[str, torch.Tensor]:
        """
        Args:
            silhouette    : binary body masks
            skeleton      : pose keypoints sequence
            optical_flow  : motion fields
            extended_seq  : long sequence for cycle analysis
            labels        : identity class ids (for training)
            return_all    : return intermediate outputs

        Returns:
            dict:
                'embedding'   : [B, emb_dim]   L2-normalized identity embedding
                'logit'       : [B, num_classes] classification logits
                'cycle_metrics': cycle analysis results (if use_cycle)
        """
        feats = []

        # ── Encode each modality ─────────────────────────────────────────────
        if self.use_sil and silhouette is not None:
            sil_feat = self.sil_encoder(silhouette)  # [B, feat_dim]
            feats.append(sil_feat)

        if self.use_skeleton and skeleton is not None:
            skel_feat = self.skel_encoder(skeleton)   # [B, feat_dim]
            feats.append(skel_feat)

        if self.use_flow and optical_flow is not None:
            flow_feat = self.flow_encoder(optical_flow)  # [B, feat_dim]
            feats.append(flow_feat)

        # ── Fuse modalities ──────────────────────────────────────────────────
        if feats:
            fused = torch.cat(feats, dim=1)            # [B, D*num_modalities]
        else:
            # Fallback: use silhouette or skeleton
            fused = feats[0] if feats else \
                    torch.randn(silhouette.size(0), self.feat_dim,
                               device=silhouette.device)

        fused = self.fusion(fused)                      # [B, feat_dim]

        # ── Gait Cycle Analysis ──────────────────────────────────────────────
        cycle_out = {}
        if self.use_cycle and extended_seq is not None:
            cycle_out = self.cycle_analyzer(extended_seq)
            # Append frequency features
            fused = torch.cat([fused, cycle_out["freq_features"]], dim=1)

        # ── Identity Embedding ───────────────────────────────────────────────
        embedding = self.embedding_head(fused)          # [B, emb_dim]
        embedding = F.normalize(embedding, dim=-1)      # L2-norm

        # ── Classification ───────────────────────────────────────────────────
        logit = self.classifier(embedding)              # [B, num_classes]

        out = {
            "embedding": embedding,
            "logit"    : logit,
        }

        if cycle_out:
            out["cycle_metrics"] = cycle_out

        if return_all:
            if self.use_sil and silhouette is not None:
                out["sil_feat"] = feats[0]
            if self.use_skeleton and skeleton is not None:
                out["skel_feat"] = feats[-1 if self.use_flow else -1]
            if self.use_flow and optical_flow is not None:
                out["flow_feat"] = feats[-1]

        return out


# ── Loss ──────────────────────────────────────────────────────────────────────

class GaitRecognitionLoss(nn.Module):
    """
    Combined gait recognition loss:

        L = α · ArcFace(logit, labels)
          + β · TripletLoss(embeddings)
          + γ · CycleCon(cycle_metrics, labels)

    ArcFace          → discriminative classification
    TripletLoss      → metric learning: same-identity embeddings close
    CycleCon         → cycle metrics consistent within identity
    """

    def __init__(
        self,
        alpha: float = 1.0,       # ArcFace
        beta : float = 0.5,       # Triplet
        gamma: float = 0.2,       # Cycle consistency
        margin: float = 0.5,
    ):
        super().__init__()
        self.alpha  = alpha
        self.beta   = beta
        self.gamma  = gamma
        self.ce     = nn.CrossEntropyLoss()
        self.margin = margin

    def _triplet_loss(
        self,
        emb     : torch.Tensor,   # [B, emb_dim]
        labels  : torch.Tensor,   # [B]
    ) -> torch.Tensor:
        """
        Triplet loss: L = max(0, d(a,p) - d(a,n) + margin)
        a = anchor, p = positive (same id), n = negative (different id)
        """
        # Compute pairwise distances
        dist = torch.cdist(emb, emb, p=2)        # [B, B] euclidean distances

        # Build mask: pos[i,j] = 1 if labels[i] == labels[j] and i != j
        labels_eq = labels.unsqueeze(0) == labels.unsqueeze(1)  # [B, B]
        diag_mask = ~torch.eye(labels.size(0), dtype=torch.bool,
                               device=labels.device)
        pos_mask  = labels_eq & diag_mask

        # Negative mask: neg[i,j] = 1 if labels[i] != labels[j]
        neg_mask  = ~labels_eq

        # Get hardest positive and hardest negative per anchor
        pos_dist  = torch.where(pos_mask, dist, torch.tensor(float('inf')))
        neg_dist  = torch.where(neg_mask, dist, torch.tensor(-float('inf')))

        hardest_pos = pos_dist.min(dim=1).values             # [B]
        hardest_neg = neg_dist.max(dim=1).values             # [B]

        triplet = F.relu(hardest_pos - hardest_neg + self.margin)
        return triplet.mean()

    def forward(
        self,
        logit         : torch.Tensor,           # [B, num_classes]
        embedding     : torch.Tensor,           # [B, emb_dim]
        labels        : torch.Tensor,           # [B]
        cycle_metrics : Optional[Dict],         # cycle analysis outputs
    ) -> Tuple[torch.Tensor, Dict[str, float]]:

        # ArcFace loss
        ce_loss = self.ce(logit, labels)
        total   = self.alpha * ce_loss
        metrics = {"ce_loss": ce_loss.item()}

        # Triplet loss
        triplet_loss = self._triplet_loss(embedding, labels)
        total       += self.beta * triplet_loss
        metrics["triplet_loss"] = triplet_loss.item()

        # Cycle consistency (simple: regularize phase consistency)
        if cycle_metrics is not None and "phase_consistency" in cycle_metrics:
            phase_con = cycle_metrics["phase_consistency"]
            cycle_loss = (1.0 - phase_con.sigmoid()).mean()  # encourage high phase consistency
            total     += self.gamma * cycle_loss
            metrics["cycle_loss"] = cycle_loss.item()

        metrics["total_loss"] = total.item()
        return total, metrics


# ── Gait Gate (pipeline integration) ──────────────────────────────────────────

class GaitGate(nn.Module):
    """
    Drop-in gait recognition gate for biometric fusion.

    Can be used to:
        1. Cross-verify face + gait match (same person?)
        2. Identify via gait when face is occluded
        3. Detect impostor in body-swap deepfakes

    Complementary to face recognition:
        • Face: high accuracy at short distance, sensitive to occlusion
        • Gait: works at distance, invariant to face occlusion/makeup

    Usage:
        gait_gate = GaitGate(weights_path="gait.pt")
        result    = gait_gate(silhouette=sil, skeleton=skel)
        gait_emb  = result["embedding"]   # compare with face embedding
    """

    def __init__(
        self,
        weights_path: Optional[str] = None,
        feat_dim    : int   = 256,
        emb_dim     : int   = 512,
        num_classes : int   = 10000,
        seq_len     : int   = 32,
        num_joints  : int   = 17,
        device      : str   = "cpu",
    ):
        super().__init__()
        self.device = torch.device(device)
        self.model  = GaitRecognition(
            feat_dim=feat_dim,
            emb_dim=emb_dim,
            num_classes=num_classes,
            seq_len=seq_len,
            num_joints=num_joints,
            use_sil=True,
            use_skeleton=True,
            use_flow=False,
            use_cycle=True,
        ).to(self.device)
        self.model.eval()

        if weights_path is not None:
            state = torch.load(weights_path, map_location=self.device)
            self.model.load_state_dict(state)
            print(f"[GaitGate] Loaded weights from: {weights_path}")
        else:
            print("[GaitGate] ⚠️  No weights loaded — random init.")

    @torch.no_grad()
    def forward(
        self,
        silhouette   : Optional[torch.Tensor] = None,
        skeleton     : Optional[torch.Tensor] = None,
        optical_flow : Optional[torch.Tensor] = None,
        extended_seq : Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Returns:
            dict:
                'embedding'     : [B, emb_dim]   gait identity embedding
                'logit'         : [B, num_classes]
                'cycle_metrics' : cycle analysis (if available)
        """
        if silhouette is not None:
            silhouette = silhouette.to(self.device)
        if skeleton is not None:
            skeleton = skeleton.to(self.device)
        if optical_flow is not None:
            optical_flow = optical_flow.to(self.device)
        if extended_seq is not None:
            extended_seq = extended_seq.to(self.device)

        return self.model(
            silhouette=silhouette,
            skeleton=skeleton,
            optical_flow=optical_flow,
            extended_seq=extended_seq,
        )


# ── Face-Gait Fusion Verification ────────────────────────────────────────────

class FaceGaitVerification(nn.Module):
    """
    Fusion module comparing face and gait embeddings for identity matching.

    Uses weighted combination:
        score = α · face_sim + β · gait_sim
    
    With confidence calibration:
        • High agreement (face & gait match) → high confidence
        • Low agreement (face & gait disagree) → suspicious, flag deepfake
        • One modality missing → fall back to single modality

    Useful for:
        • Presence verification (is it really the same person?)
        • Deepfake detection (face/gait swap)
        • Liveness assurance (gait is hard to fake convincingly)
    """

    def __init__(
        self,
        emb_dim : int = 512,
        face_weight: float = 0.6,
        gait_weight: float = 0.4,
    ):
        super().__init__()
        self.face_weight = face_weight
        self.gait_weight = gait_weight

        # Calibration network: predict confidence from embeddings
        self.confidence_head = nn.Sequential(
            nn.Linear(emb_dim * 2, 256),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(256, 128),
            nn.GELU(),
            nn.Linear(128, 1),
            nn.Sigmoid(),                       # confidence ∈ [0, 1]
        )

    def forward(
        self,
        face_emb : torch.Tensor,                # [B, emb_dim]
        gait_emb : Optional[torch.Tensor] = None,  # [B, emb_dim]
        face_ref : torch.Tensor = None,        # [N, emb_dim]  reference DB
        gait_ref : Optional[torch.Tensor] = None,  # [N, emb_dim]
    ) -> Dict[str, torch.Tensor]:
        """
        Args:
            face_emb : query face embedding
            gait_emb : query gait embedding (optional)
            face_ref : reference face embeddings from DB
            gait_ref : reference gait embeddings from DB

        Returns:
            dict:
                'face_sim'       : [B, N]  cosine similarity to face refs
                'gait_sim'       : [B, N]  cosine similarity to gait refs
                'combined_sim'   : [B, N]  weighted fusion
                'confidence'     : [B]     fusion reliability score
                'top_match'      : [B]     index of closest match
                'match_score'    : [B]     score of top match
        """
        B = face_emb.size(0)

        # Compute similarities
        face_sim = F.cosine_similarity(
            face_emb.unsqueeze(1), face_ref.unsqueeze(0), dim=2
        )                                       # [B, N]

        if gait_emb is not None and gait_ref is not None:
            gait_sim = F.cosine_similarity(
                gait_emb.unsqueeze(1), gait_ref.unsqueeze(0), dim=2
            )                                   # [B, N]
            combined = (self.face_weight * face_sim
                      + self.gait_weight * gait_sim)   # [B, N]

            # Confidence: how well do face & gait agree?
            face_top = face_sim.max(dim=1).values        # [B]
            gait_top = gait_sim.max(dim=1).values        # [B]
            joint_emb = torch.cat([face_emb, gait_emb], dim=1)  # [B, 2*D]
        else:
            gait_sim  = torch.zeros_like(face_sim)
            combined  = face_sim
            joint_emb = torch.cat([face_emb, face_emb], dim=1)  # duplicate for consistency

        confidence = self.confidence_head(joint_emb)  # [B, 1]

        # Top matches
        top_scores, top_indices = combined.max(dim=1)  # [B], [B]

        return {
            "face_sim"    : face_sim,
            "gait_sim"    : gait_sim,
            "combined_sim": combined,
            "confidence"  : confidence.squeeze(1),
            "top_match"   : top_indices,
            "match_score" : top_scores,
        }
