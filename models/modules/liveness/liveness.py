"""
Liveness Detection Module
Detects whether a face is live (real person) or non-live (photo/video/mask).
Architecture: Multi-cue fusion — blink detection + micro-motion +
              physiological signal (rPPG) + challenge-response.
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
        kernel  : int = 3,
        stride  : int = 1,
        padding : int = 1,
        dilation: int = 1,
    ):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel,
                      stride=stride, padding=padding,
                      dilation=dilation, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class SEBlock(nn.Module):
    """
    Squeeze-and-Excitation block.
    Recalibrates channel-wise feature responses adaptively.
    """

    def __init__(self, ch: int, reduction: int = 8):
        super().__init__()
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(ch, ch // reduction),
            nn.ReLU(inplace=True),
            nn.Linear(ch // reduction, ch),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        w = self.se(x).view(x.size(0), x.size(1), 1, 1)
        return x * w


class TemporalConvBlock(nn.Module):
    """
    1D Temporal Convolution block for sequence modeling.
    Used in blink detection and rPPG signal processing.
    Causal padding ensures no future frame leakage.
    """

    def __init__(
        self,
        in_ch  : int,
        out_ch : int,
        kernel : int = 3,
        dilation: int = 1,
    ):
        super().__init__()
        # Causal padding: pad only left side
        self.pad  = (kernel - 1) * dilation
        self.conv = nn.Conv1d(in_ch, out_ch, kernel,
                              dilation=dilation, bias=False)
        self.bn   = nn.BatchNorm1d(out_ch)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: [B, C, T]"""
        x = F.pad(x, (self.pad, 0))    # causal left-pad
        return self.relu(self.bn(self.conv(x)))


# ── Eye Landmark Feature Extractor ────────────────────────────────────────────

class EyeRegionEncoder(nn.Module):
    """
    Encodes left/right eye region crops into compact feature vectors.
    Used by BlinkDetector to track eye aperture over time.
    Input: eye region crop  [B, 1, 24, 64]  (grayscale, H×W)
    """

    def __init__(self, out_dim: int = 64):
        super().__init__()
        self.encoder = nn.Sequential(
            ConvBNReLU(1,  16, kernel=3),    # [B, 16, 24, 64]
            nn.MaxPool2d(2),                  # [B, 16, 12, 32]
            ConvBNReLU(16, 32, kernel=3),    # [B, 32, 12, 32]
            SEBlock(32),
            nn.MaxPool2d(2),                  # [B, 32,  6, 16]
            ConvBNReLU(32, 64, kernel=3),    # [B, 64,  6, 16]
            nn.AdaptiveAvgPool2d(1),          # [B, 64,  1,  1]
            nn.Flatten(),                     # [B, 64]
            nn.Linear(64, out_dim),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)              # [B, out_dim]


class BlinkDetector(nn.Module):
    """
    Blink detection over a temporal window of frames.

    Pipeline:
        Eye crops (T frames)
            │
            ├── EyeRegionEncoder (per frame, shared weights)
            │       → eye feature sequence [B, T, 64]
            │
            ├── TemporalConvNet (multi-scale temporal)
            │       → temporal features [B, T, 64]
            │
            └── Classifier → blink logits [B, T]
                           + blink count  [B]
                           + blink rate   [B]

    A real face should produce natural blink patterns (0.1–0.4 Hz).
    A photo/screen shows either no blinks or unnatural periodic blinks.
    """

    def __init__(
        self,
        feat_dim : int = 64,
        seq_len  : int = 30,          # number of frames in temporal window
    ):
        super().__init__()
        self.seq_len       = seq_len
        self.eye_encoder   = EyeRegionEncoder(out_dim=feat_dim)

        # Multi-scale temporal convolutions (dilation = 1, 2, 4)
        self.temporal = nn.Sequential(
            TemporalConvBlock(feat_dim, feat_dim, kernel=3, dilation=1),
            TemporalConvBlock(feat_dim, feat_dim, kernel=3, dilation=2),
            TemporalConvBlock(feat_dim, feat_dim, kernel=3, dilation=4),
        )

        # Per-frame blink probability
        self.frame_classifier = nn.Sequential(
            nn.Conv1d(feat_dim, 32, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv1d(32, 1, kernel_size=1),    # [B, 1, T]
        )

    def forward(
        self,
        left_eyes : torch.Tensor,     # [B, T, 1, 24, 64]
        right_eyes: torch.Tensor,     # [B, T, 1, 24, 64]
    ) -> Dict[str, torch.Tensor]:

        B, T = left_eyes.shape[:2]

        # Encode each eye frame (shared encoder)
        left_feats  = torch.stack(
            [self.eye_encoder(left_eyes[:, t])  for t in range(T)], dim=1
        )   # [B, T, feat_dim]

        right_feats = torch.stack(
            [self.eye_encoder(right_eyes[:, t]) for t in range(T)], dim=1
        )   # [B, T, feat_dim]

        # Average both eyes
        eye_feats = (left_feats + right_feats) / 2.0   # [B, T, feat_dim]
        eye_feats = eye_feats.permute(0, 2, 1)          # [B, feat_dim, T]

        # Temporal modeling
        temporal_feat = self.temporal(eye_feats)         # [B, feat_dim, T]

        # Per-frame blink logits
        blink_logits  = self.frame_classifier(temporal_feat).squeeze(1)  # [B, T]
        blink_probs   = torch.sigmoid(blink_logits)      # [B, T]

        # Blink count and rate
        blink_count   = (blink_probs > 0.5).float().sum(dim=1)          # [B]
        blink_rate    = blink_count / T                                   # [B]

        return {
            "blink_logits": blink_logits,   # [B, T]
            "blink_probs" : blink_probs,    # [B, T]
            "blink_count" : blink_count,    # [B]
            "blink_rate"  : blink_rate,     # [B]  blinks/frame
        }


# ── Micro-Motion Detector ─────────────────────────────────────────────────────

class OpticalFlowEncoder(nn.Module):
    """
    Encodes optical flow fields into motion feature vectors.
    Real faces show natural micro-motions (breathing, subtle head sway).
    Fake faces (photos) show near-zero or camera-shake-only motion.

    Input: stacked flow fields [B, 2*num_pairs, H, W]
           (2 channels per flow pair: dx, dy)
    """

    def __init__(
        self,
        in_ch  : int = 4,      # 2 flow pairs × 2 channels
        out_dim: int = 128,
    ):
        super().__init__()
        self.encoder = nn.Sequential(
            ConvBNReLU(in_ch, 32),
            SEBlock(32),
            nn.MaxPool2d(2),
            ConvBNReLU(32, 64),
            SEBlock(64),
            nn.MaxPool2d(2),
            ConvBNReLU(64, 128),
            SEBlock(128),
            nn.AdaptiveAvgPool2d(4),
            nn.Flatten(),
            nn.Linear(128 * 4 * 4, out_dim),
            nn.ReLU(inplace=True),
        )

    def forward(self, flow: torch.Tensor) -> torch.Tensor:
        return self.encoder(flow)               # [B, out_dim]


class MicroMotionDetector(nn.Module):
    """
    Detects natural micro-motion patterns across a frame sequence.

    Computes frame-difference features (proxy for optical flow)
    and classifies whether motion is consistent with a live face.

    Features:
    - Motion magnitude distribution (real: small + variable)
    - Motion direction entropy      (real: random, fake: structured)
    - Temporal autocorrelation      (real: no periodicity / natural)
    """

    def __init__(
        self,
        feat_dim: int = 128,
        seq_len : int = 15,
    ):
        super().__init__()
        self.seq_len       = seq_len
        self.flow_encoder  = OpticalFlowEncoder(in_ch=6, out_dim=feat_dim)

        # Temporal aggregator over frame-difference sequence
        self.temporal_agg  = nn.Sequential(
            TemporalConvBlock(feat_dim, feat_dim, kernel=3, dilation=1),
            TemporalConvBlock(feat_dim, feat_dim, kernel=3, dilation=2),
            nn.AdaptiveAvgPool1d(1),    # [B, feat_dim, 1]
            nn.Flatten(),               # [B, feat_dim]
        )

        # Motion liveness classifier
        self.classifier = nn.Sequential(
            nn.Linear(feat_dim, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(64, 1),           # logit: live motion vs static
        )

    def _compute_frame_diff(
        self,
        frames: torch.Tensor,           # [B, T, 3, H, W]
    ) -> torch.Tensor:
        """
        Compute consecutive frame differences as proxy optical flow.
        Returns stacked diffs: [B, T-1, 6, H, W]  (3ch × 2 pairs)
        """
        diffs = []
        T = frames.shape[1]
        for t in range(T - 1):
            diff = torch.cat([
                frames[:, t+1] - frames[:, t],       # forward diff  [B,3,H,W]
                frames[:, t]   - frames[:, max(0,t-1)],  # backward diff [B,3,H,W]
            ], dim=1)                                 # [B, 6, H, W]
            diffs.append(diff)
        return torch.stack(diffs, dim=1)              # [B, T-1, 6, H, W]

    def forward(
        self,
        frames: torch.Tensor,           # [B, T, 3, H, W]
    ) -> Dict[str, torch.Tensor]:

        B, T, C, H, W = frames.shape

        # Frame differences → proxy flow
        diffs = self._compute_frame_diff(frames)      # [B, T-1, 6, H, W]
        T_diff = diffs.shape[1]

        # Encode each diff frame
        flow_feats = torch.stack(
            [self.flow_encoder(diffs[:, t]) for t in range(T_diff)], dim=1
        )   # [B, T-1, feat_dim]

        flow_feats = flow_feats.permute(0, 2, 1)      # [B, feat_dim, T-1]

        # Temporal aggregation
        motion_feat = self.temporal_agg(flow_feats)   # [B, feat_dim]

        # Classify motion liveness
        logit       = self.classifier(motion_feat)    # [B, 1]
        prob        = torch.sigmoid(logit)

        # Motion statistics (for analysis/explainability)
        motion_mag  = diffs.abs().mean(dim=[2, 3, 4]) # [B, T-1]  per-frame mag

        return {
            "logit"      : logit,           # [B, 1]
            "prob"       : prob,            # [B, 1]
            "motion_feat": motion_feat,     # [B, feat_dim]
            "motion_mag" : motion_mag,      # [B, T-1]
        }


# ── rPPG (Remote Photoplethysmography) ───────────────────────────────────────

class rPPGExtractor(nn.Module):
    """
    Remote Photoplethysmography signal extractor.

    Extracts subtle skin color variations caused by blood flow,
    visible as tiny periodic changes in the green channel of the face.
    A live face shows a heartbeat signal (~0.8–2.5 Hz / 48–150 BPM).
    A fake face (photo/screen) shows no such physiological signal.

    Pipeline:
        Face sequence [B, T, 3, H, W]
            │
            ├── Spatial ROI pooling (forehead + cheek regions)
            │       → color signal [B, T, 3]
            │
            ├── Temporal ConvNet
            │       → refined signal [B, T, 1]
            │
            └── FFT → frequency spectrum → HR + liveness score
    """

    def __init__(
        self,
        seq_len    : int = 150,   # ~5 seconds at 30fps
        feat_dim   : int = 64,
    ):
        super().__init__()
        self.seq_len  = seq_len

        # Spatial attention: learn which face regions carry strongest rPPG
        self.spatial_attn = nn.Sequential(
            ConvBNReLU(3, 16),
            ConvBNReLU(16, 1),
            nn.Sigmoid(),             # attention map [B, 1, H, W]
        )

        # Temporal signal refiner
        self.signal_refiner = nn.Sequential(
            TemporalConvBlock(3,        feat_dim, kernel=5, dilation=1),
            TemporalConvBlock(feat_dim, feat_dim, kernel=5, dilation=2),
            TemporalConvBlock(feat_dim, feat_dim, kernel=3, dilation=4),
            nn.Conv1d(feat_dim, 1, kernel_size=1),   # → [B, 1, T]
        )

        # Liveness head from spectral features
        self.liveness_head = nn.Sequential(
            nn.Linear(seq_len // 2 + 1, 128),    # FFT output size
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 1),
        )

    def _extract_color_signal(
        self,
        frames: torch.Tensor,        # [B, T, 3, H, W]
    ) -> torch.Tensor:
        """
        Extract spatially-attended mean RGB signal per frame.
        Returns: [B, 3, T]
        """
        B, T, C, H, W = frames.shape
        signals = []

        for t in range(T):
            frame      = frames[:, t]                  # [B, 3, H, W]
            attn_map   = self.spatial_attn(frame)      # [B, 1, H, W]
            # Weighted spatial average
            weighted   = (frame * attn_map).sum(dim=[2, 3]) / \
                         (attn_map.sum(dim=[2, 3]) + 1e-6)  # [B, 3]
            signals.append(weighted)

        return torch.stack(signals, dim=2)             # [B, 3, T]

    def forward(
        self,
        frames    : torch.Tensor,    # [B, T, 3, H, W]
        fps       : float = 30.0,
        return_signal: bool = False,
    ) -> Dict[str, torch.Tensor]:

        B, T = frames.shape[:2]

        # Extract color signal
        color_signal = self._extract_color_signal(frames)  # [B, 3, T]

        # Normalize signal (remove lighting trend)
        color_signal = color_signal - color_signal.mean(dim=2, keepdim=True)
        color_signal = color_signal / (color_signal.std(dim=2, keepdim=True) + 1e-6)

        # Refine signal with temporal convnet
        rppg_signal  = self.signal_refiner(color_signal)   # [B, 1, T]
        rppg_signal  = rppg_signal.squeeze(1)              # [B, T]

        # FFT → frequency spectrum
        fft_out  = torch.fft.rfft(rppg_signal, dim=1)      # [B, T//2+1] complex
        spectrum = torch.abs(fft_out)                       # [B, T//2+1] magnitude

        # Liveness from spectrum
        logit    = self.liveness_head(spectrum)             # [B, 1]
        prob     = torch.sigmoid(logit)

        # Estimate heart rate from dominant frequency
        freqs    = torch.fft.rfftfreq(T, d=1.0 / fps)      # frequency bins in Hz
        hr_idx   = spectrum.argmax(dim=1)                   # [B]
        hr_hz    = freqs[hr_idx]                            # [B]
        hr_bpm   = hr_hz * 60.0                             # [B]

        out = {
            "logit"   : logit,        # [B, 1]
            "prob"    : prob,         # [B, 1]
            "hr_bpm"  : hr_bpm,       # [B]   estimated heart rate
            "spectrum": spectrum,     # [B, T//2+1]
        }
        if return_signal:
            out["rppg_signal"] = rppg_signal   # [B, T]

        return out


# ── Challenge-Response Module ─────────────────────────────────────────────────

class HeadPoseEstimator(nn.Module):
    """
    Estimates head pose (yaw, pitch, roll) from a single face crop.
    Used in challenge-response: user is asked to nod / turn head.
    Verifies that the face actually follows the requested action.

    Output: (yaw, pitch, roll) in degrees  ∈ [-90, 90]
    """

    def __init__(self, in_ch: int = 3):
        super().__init__()
        self.backbone = nn.Sequential(
            ConvBNReLU(in_ch, 32),
            nn.MaxPool2d(2),
            ConvBNReLU(32,  64),
            SEBlock(64),
            nn.MaxPool2d(2),
            ConvBNReLU(64, 128),
            SEBlock(128),
            nn.MaxPool2d(2),
            ConvBNReLU(128, 256),
            nn.AdaptiveAvgPool2d(4),
            nn.Flatten(),
        )
        self.pose_head = nn.Sequential(
            nn.Linear(256 * 4 * 4, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(256, 3),           # yaw, pitch, roll
            nn.Tanh(),                   # ∈ [-1, 1] → scale to degrees
        )
        self.scale = 90.0                # degrees range

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feat = self.backbone(x)          # [B, 4096]
        pose = self.pose_head(feat)      # [B, 3]
        return pose * self.scale         # [B, 3]  yaw, pitch, roll in degrees


class ChallengeResponseModule(nn.Module):
    """
    Active liveness check via challenge-response.

    Supported challenges:
        - "nod"         → pitch change  > threshold
        - "shake"       → yaw change    > threshold
        - "tilt_left"   → roll change   > threshold
        - "tilt_right"  → roll change   > threshold (opposite direction)
        - "blink"       → blink detected in sequence

    Verifies that the user's face responds correctly to a given challenge
    within a time window, confirming physical presence.
    """

    CHALLENGES     = ["nod", "shake", "tilt_left", "tilt_right", "blink"]
    POSE_THRESHOLD = 15.0   # degrees of movement required

    def __init__(self):
        super().__init__()
        self.pose_estimator = HeadPoseEstimator()

    def forward(
        self,
        frames   : torch.Tensor,       # [B, T, 3, H, W]
        challenge: str = "nod",
    ) -> Dict[str, torch.Tensor]:
        """
        Args:
            frames   : face sequence  [B, T, 3, H, W]
            challenge: one of CHALLENGES

        Returns:
            dict:
                'passed'      : [B]  bool — challenge completed
                'pose_seq'    : [B, T, 3]  yaw/pitch/roll over time
                'delta'       : [B, 3]     max pose change over sequence
        """
        assert challenge in self.CHALLENGES, \
            f"Unknown challenge: {challenge}. Choose from {self.CHALLENGES}"

        B, T, C, H, W = frames.shape

        # Estimate pose for each frame
        pose_seq = torch.stack(
            [self.pose_estimator(frames[:, t]) for t in range(T)], dim=1
        )   # [B, T, 3]   columns: [yaw, pitch, roll]

        # Compute max pose change over time window
        pose_min  = pose_seq.min(dim=1).values    # [B, 3]
        pose_max  = pose_seq.max(dim=1).values    # [B, 3]
        delta     = pose_max - pose_min           # [B, 3]

        # Check challenge completion
        # yaw=0, pitch=1, roll=2
        if challenge == "nod":
            passed = delta[:, 1] > self.POSE_THRESHOLD
        elif challenge == "shake":
            passed = delta[:, 0] > self.POSE_THRESHOLD
        elif challenge == "tilt_left":
            passed = (delta[:, 2] > self.POSE_THRESHOLD) & \
                     (pose_seq[:, -1, 2] - pose_seq[:, 0, 2] > 0)
        elif challenge == "tilt_right":
            passed = (delta[:, 2] > self.POSE_THRESHOLD) & \
                     (pose_seq[:, -1, 2] - pose_seq[:, 0, 2] < 0)
        elif challenge == "blink":
            # Delegate to blink rate check (simplified: proxy via pixel variance)
            passed = delta[:, 1] > 5.0   # small pitch during blink

        return {
            "passed"  : passed,           # [B] bool
            "pose_seq": pose_seq,         # [B, T, 3]
            "delta"   : delta,            # [B, 3]
        }


# ── Fusion Classifier ─────────────────────────────────────────────────────────

class LivenessFusionClassifier(nn.Module):
    """
    Fuses all liveness cues into a single liveness score.

    Inputs (concatenated):
        - Blink rate feature         [B, 1]
        - Micro-motion prob          [B, 1]
        - rPPG liveness prob         [B, 1]
        - Challenge-response passed  [B, 1]

    Output: liveness logit [B, 1]
    """

    def __init__(self, in_dim: int = 4):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 32),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(32, 16),
            nn.ReLU(inplace=True),
            nn.Linear(16, 1),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """features: [B, in_dim]"""
        return self.net(features)         # [B, 1]


# ── Full Liveness Module ──────────────────────────────────────────────────────

class LivenessDetector(nn.Module):
    """
    Full multi-cue liveness detection pipeline.

    Cues combined:
        1. Blink Detection        — eye motion over time
        2. Micro-Motion Detection — subtle facial motion analysis
        3. rPPG Signal            — physiological blood-flow signal
        4. Challenge-Response     — active head-pose verification

    Usage modes:
        • Passive  : blink + micro-motion + rPPG  (no user interaction)
        • Active   : + challenge-response         (user follows instruction)
    """

    def __init__(
        self,
        seq_len  : int = 30,
        rppg_len : int = 150,
        fps      : float = 30.0,
    ):
        super().__init__()
        self.seq_len  = seq_len
        self.rppg_len = rppg_len
        self.fps      = fps

        self.blink_detector    = BlinkDetector(seq_len=seq_len)
        self.motion_detector   = MicroMotionDetector(seq_len=seq_len)
        self.rppg_extractor    = rPPGExtractor(seq_len=rppg_len)
        self.challenge_module  = ChallengeResponseModule()
        self.fusion_classifier = LivenessFusionClassifier(in_dim=4)

    def forward(
        self,
        frames       : torch.Tensor,           # [B, T, 3, H, W]
        left_eyes    : Optional[torch.Tensor], # [B, T, 1, 24, 64]
        right_eyes   : Optional[torch.Tensor], # [B, T, 1, 24, 64]
        rppg_frames  : Optional[torch.Tensor], # [B, T_rppg, 3, H, W]
        challenge    : Optional[str] = None,   # "nod","shake","tilt_left", etc.
        return_all   : bool = False,
    ) -> Dict[str, torch.Tensor]:
        """
        Args:
            frames      : main face sequence      [B, T, 3, H, W]
            left_eyes   : left eye crops          [B, T, 1, 24, 64]
            right_eyes  : right eye crops         [B, T, 1, 24, 64]
            rppg_frames : longer face sequence    [B, T_rppg, 3, H, W]
            challenge   : optional challenge str
            return_all  : return all cue outputs

        Returns:
            dict:
                'logit'    : [B, 1]   raw liveness logit
                'prob'     : [B, 1]   liveness probability
                'is_live'  : [B]      bool predictions
                'hr_bpm'   : [B]      estimated heart rate (if rPPG available)
                + individual cue outputs (if return_all=True)
        """

        B = frames.shape[0]
        cue_features = []

        # ── Cue 1: Blink ─────────────────────────────────────────────────────
        if left_eyes is not None and right_eyes is not None:
            blink_out  = self.blink_detector(left_eyes, right_eyes)
            blink_feat = blink_out["blink_rate"].unsqueeze(1)    # [B, 1]
        else:
            blink_feat = torch.zeros(B, 1, device=frames.device)
            blink_out  = {}
        cue_features.append(blink_feat)

        # ── Cue 2: Micro-Motion ───────────────────────────────────────────────
        motion_out  = self.motion_detector(frames)
        motion_feat = motion_out["prob"]                          # [B, 1]
        cue_features.append(motion_feat)

        # ── Cue 3: rPPG ──────────────────────────────────────────────────────
        if rppg_frames is not None:
            rppg_out  = self.rppg_extractor(rppg_frames, fps=self.fps)
            rppg_feat = rppg_out["prob"]                          # [B, 1]
            hr_bpm    = rppg_out["hr_bpm"]                        # [B]
        else:
            rppg_out  = {}
            rppg_feat = torch.zeros(B, 1, device=frames.device)
            hr_bpm    = torch.zeros(B,    device=frames.device)
        cue_features.append(rppg_feat)

        # ── Cue 4: Challenge-Response ─────────────────────────────────────────
        if challenge is not None:
            cr_out   = self.challenge_module(frames, challenge=challenge)
            cr_feat  = cr_out["passed"].float().unsqueeze(1)      # [B, 1]
        else:
            cr_out   = {}
            cr_feat  = torch.zeros(B, 1, device=frames.device)
        cue_features.append(cr_feat)

        # ── Fusion ────────────────────────────────────────────────────────────
        fused = torch.cat(cue_features, dim=1)                    # [B, 4]
        logit = self.fusion_classifier(fused)                     # [B, 1]
        prob  = torch.sigmoid(logit)                              # [B, 1]

        out = {
            "logit"  : logit,
            "prob"   : prob,
            "is_live": (prob.squeeze(1) >= 0.5),
            "hr_bpm" : hr_bpm,
        }

        if return_all:
            out["blink_out"]  = blink_out
            out["motion_out"] = motion_out
            out["rppg_out"]   = rppg_out
            out["cr_out"]     = cr_out

        return out


# ── Loss ──────────────────────────────────────────────────────────────────────

class LivenessLoss(nn.Module):
    """
    Combined liveness loss:

        L = α · BCE(logit, label)
          + β · BlinkConsistency(blink_rate, label)
          + γ · MotionConsistency(motion_prob, label)
          + δ · rPPGConsistency(rppg_prob, label)

    Each consistency term enforces that individual cues align
    with the ground-truth liveness label.
    """

    def __init__(
        self,
        alpha: float = 1.0,    # main BCE
        beta : float = 0.3,    # blink
        gamma: float = 0.3,    # motion
        delta: float = 0.4,    # rPPG
    ):
        super().__init__()
        self.alpha = alpha
        self.beta  = beta
        self.gamma = gamma
        self.delta = delta
        self.bce   = nn.BCEWithLogitsLoss()

    def forward(
        self,
        logit      : torch.Tensor,            # [B, 1]
        labels     : torch.Tensor,            # [B]  1=live, 0=spoof
        blink_rate : Optional[torch.Tensor],  # [B]
        motion_prob: Optional[torch.Tensor],  # [B, 1]
        rppg_prob  : Optional[torch.Tensor],  # [B, 1]
    ) -> Tuple[torch.Tensor, Dict[str, float]]:

        # Main BCE loss
        bce_loss = self.bce(logit.squeeze(1), labels.float())

        metrics   = {"bce_loss": bce_loss.item()}
        total     = self.alpha * bce_loss

        # Blink consistency: live faces should have non-zero blink rate
        if blink_rate is not None:
            blink_loss  = self.bce(
                blink_rate.unsqueeze(1) if blink_rate.dim() == 1
                else blink_rate,
                labels.float().unsqueeze(1),
            )
            total      += self.beta * blink_loss
            metrics["blink_loss"] = blink_loss.item()

        # Motion consistency
        if motion_prob is not None:
            motion_loss  = self.bce(
                torch.logit(motion_prob.clamp(1e-6, 1 - 1e-6)),
                labels.float().unsqueeze(1),
            )
            total       += self.gamma * motion_loss
            metrics["motion_loss"] = motion_loss.item()

        # rPPG consistency
        if rppg_prob is not None:
            rppg_loss    = self.bce(
                torch.logit(rppg_prob.clamp(1e-6, 1 - 1e-6)),
                labels.float().unsqueeze(1),
            )
            total       += self.delta * rppg_loss
            metrics["rppg_loss"] = rppg_loss.item()

        metrics["total_loss"] = total.item()
        return total, metrics


# ── Liveness Gate (pipeline integration) ──────────────────────────────────────

class LivenessGate(nn.Module):
    """
    Drop-in liveness gate for the full face recognition pipeline.

    Slot order in pipeline:
        DenoiserGate → LivenessGate → AntiSpoofingGate → FaceRecognizer

    Usage:
        gate   = LivenessGate(weights_path="liveness.pt", threshold=0.6)
        result = gate(frames, left_eyes, right_eyes, rppg_frames)
        if result["passed"].all():
            embedding = face_recognizer(frames[:, -1])
    """

    def __init__(
        self,
        weights_path: Optional[str] = None,
        threshold   : float = 0.6,
        device      : str   = "cpu",
        seq_len     : int   = 30,
        rppg_len    : int   = 150,
        fps         : float = 30.0,
    ):
        super().__init__()
        self.threshold = threshold
        self.device    = torch.device(device)
        self.model     = LivenessDetector(
            seq_len=seq_len, rppg_len=rppg_len, fps=fps
        ).to(self.device)
        self.model.eval()

        if weights_path is not None:
            state = torch.load(weights_path, map_location=self.device)
            self.model.load_state_dict(state)
            print(f"[LivenessGate] Loaded weights from: {weights_path}")
        else:
            print("[LivenessGate] ⚠️  No weights loaded — running with random init.")

    @torch.no_grad()
    def forward(
        self,
        frames      : torch.Tensor,
        left_eyes   : Optional[torch.Tensor] = None,
        right_eyes  : Optional[torch.Tensor] = None,
        rppg_frames : Optional[torch.Tensor] = None,
        challenge   : Optional[str]          = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Returns:
            dict:
                'passed'  : [B] bool  — True if face is live
                'prob'    : [B]       — liveness probability
                'hr_bpm'  : [B]       — estimated heart rate
                'blocked' : indices of blocked (non-live) samples
        """
        frames = frames.to(self.device)
        if left_eyes   is not None: left_eyes   = left_eyes.to(self.device)
        if right_eyes  is not None: right_eyes  = right_eyes.to(self.device)
        if rppg_frames is not None: rppg_frames = rppg_frames.to(self.device)

        out    = self.model(frames, left_eyes, right_eyes,
                            rppg_frames, challenge=challenge)

        prob    = out["prob"].squeeze(1)
        passed  = prob >= self.threshold
        blocked = (~passed).nonzero(as_tuple=False).squeeze(1)

        return {
            "passed" : passed,
            "prob"   : prob,
            "hr_bpm" : out["hr_bpm"],
            "blocked": blocked,
        }
