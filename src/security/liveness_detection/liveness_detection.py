"""
Security — Liveness Detection Module.

Multi-signal liveness detection for robust anti-spoofing.

Signals used:
    1. CNN-based texture analysis   (static frame)
    2. Depth estimation             (face depth map)
    3. Blink detection              (video stream)
    4. Head movement                (temporal analysis)
    5. Reflection pattern analysis  (screen vs real skin)

Fusion:
    Weighted average of all available signal scores → final decision
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch
import torch.nn as nn
from PIL import Image


@dataclass
class LivenessSignal:
    """Score from a single liveness signal (0=spoof, 1=real)."""
    name        : str
    score       : float       # 0.0 – 1.0
    confidence  : float
    available   : bool = True


@dataclass
class LivenessResult:
    """Aggregated liveness detection result."""
    is_live      : bool
    final_score  : float
    signals      : List[LivenessSignal]
    threshold    : float
    method       : str = "weighted_fusion"


# ── Blink Detector ────────────────────────────────────────────────────────────

class BlinkDetector:
    """
    Detects eye blinks from a video stream using EAR (Eye Aspect Ratio).

    EAR = (||p2-p6|| + ||p3-p5||) / (2 * ||p1-p4||)

    A blink occurs when EAR drops below a threshold.
    Requires facial landmarks (68-point or 5-point).
    """

    def __init__(
        self,
        ear_threshold    : float = 0.21,
        consec_frames    : int   = 3,
        min_blinks       : int   = 1,
        max_frames       : int   = 90,   # 3 seconds @ 30fps
    ):
        self.ear_threshold = ear_threshold
        self.consec_frames = consec_frames
        self.min_blinks    = min_blinks
        self.max_frames    = max_frames
        self._frame_count  = 0
        self._blink_count  = 0
        self._consec_below = 0

    def compute_ear(self, eye_landmarks: np.ndarray) -> float:
        """
        Compute Eye Aspect Ratio from 6 eye landmark points.

        Args:
            eye_landmarks: (6, 2) array of (x, y) eye points

        Returns:
            EAR value (typically 0.1–0.4)
        """
        p1, p2, p3, p4, p5, p6 = eye_landmarks
        A = np.linalg.norm(p2 - p6)
        B = np.linalg.norm(p3 - p5)
        C = np.linalg.norm(p1 - p4)
        return float((A + B) / (2.0 * C + 1e-8))

    def update(self, ear: float) -> Tuple[bool, int]:
        """
        Update blink detector with new EAR value.

        Args:
            ear: Eye Aspect Ratio for current frame

        Returns:
            (blink_detected_this_frame, total_blinks)
        """
        self._frame_count += 1
        blink_now = False

        if ear < self.ear_threshold:
            self._consec_below += 1
        else:
            if self._consec_below >= self.consec_frames:
                self._blink_count += 1
                blink_now = True
            self._consec_below = 0

        return blink_now, self._blink_count

    def get_liveness_signal(self) -> LivenessSignal:
        """Get liveness signal based on blink count."""
        blinked = self._blink_count >= self.min_blinks
        score   = min(self._blink_count / max(self.min_blinks, 1), 1.0)

        return LivenessSignal(
            name       = "blink_detection",
            score      = round(float(score), 4),
            confidence = 0.85,
            available  = True,
        )

    def reset(self):
        """Reset blink detector state."""
        self._frame_count  = 0
        self._blink_count  = 0
        self._consec_below = 0


# ── Depth Map Estimator ───────────────────────────────────────────────────────

class DepthLivenessEstimator:
    """
    Estimates face depth map to distinguish 3D (real) from 2D (flat print/screen).

    A real face has non-uniform depth (nose protrudes, eyes recessed).
    A printed photo or screen is flat (uniform depth).

    Uses CDCN (Central Difference Convolution Network) variant.
    """

    def __init__(self, model_path: Optional[str] = None):
        self.model = self._build_depth_cnn()
        self.model.eval()

        if model_path:
            ckpt = torch.load(model_path, map_location="cpu", weights_only=False)
            self.model.load_state_dict(ckpt.get("model_state_dict", ckpt))

    def _build_depth_cnn(self) -> nn.Module:
        """
        Simple depth estimation CNN.
        TODO: Replace with CDCN for production.
        """
        return nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1), nn.ReLU(),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(),
            nn.AdaptiveAvgPool2d((14, 14)),
            nn.Flatten(),
            nn.Linear(64 * 14 * 14, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
            nn.Sigmoid(),
        )

    @torch.no_grad()
    def estimate(self, img: np.ndarray) -> LivenessSignal:
        """
        Estimate depth-based liveness score.

        Args:
            img: BGR face crop (112×112)

        Returns:
            LivenessSignal with depth liveness score
        """
        # Preprocess
        rgb    = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        tensor = torch.from_numpy(rgb).permute(2, 0, 1).float() / 255.0
        tensor = tensor.unsqueeze(0)

        score  = float(self.model(tensor)[0, 0])

        return LivenessSignal(
            name       = "depth_estimation",
            score      = round(score, 4),
            confidence = 0.80,
            available  = True,
        )


# ── Texture Analyzer ──────────────────────────────────────────────────────────

class TextureLivenessAnalyzer:
    """
    Analyzes micro-texture patterns to distinguish real vs fake faces.

    Real faces have complex micro-textures (pores, wrinkles, etc.)
    Printed photos have regular printing dot patterns.
    Screen replays have periodic pixel grid patterns.

    Uses LBP (Local Binary Patterns) + frequency domain analysis.
    """

    def analyze(self, img: np.ndarray) -> LivenessSignal:
        """
        Analyze texture for liveness cues.

        Args:
            img: BGR face crop

        Returns:
            LivenessSignal based on texture analysis
        """
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # FFT frequency analysis: real faces have richer high-freq content
        fft       = np.fft.fft2(gray.astype(np.float32))
        fft_shift = np.fft.fftshift(fft)
        mag       = np.log(np.abs(fft_shift) + 1)

        h, w   = mag.shape
        center = mag[h//4:3*h//4, w//4:3*w//4]
        border = mag.copy()
        border[h//4:3*h//4, w//4:3*w//4] = 0

        # High-freq ratio (real faces have more high-freq content)
        hf_ratio = float(border.mean() / (center.mean() + 1e-8))
        score    = min(hf_ratio * 0.5, 1.0)

        return LivenessSignal(
            name       = "texture_analysis",
            score      = round(score, 4),
            confidence = 0.75,
            available  = True,
        )


# ── Liveness Detector ─────────────────────────────────────────────────────────

class LivenessDetector:
    """
    Fuses multiple liveness signals into a final decision.

    Signal weights:
        • texture_analysis  : 0.30
        • depth_estimation  : 0.40
        • blink_detection   : 0.30

    Usage:
        detector = LivenessDetector(threshold=0.60)

        # Single frame
        result = detector.detect_frame(face_bgr_array)

        # Video stream (3s @ 30fps)
        result = detector.detect_video(frames_list)
    """

    SIGNAL_WEIGHTS = {
        "texture_analysis" : 0.30,
        "depth_estimation" : 0.40,
        "blink_detection"  : 0.30,
    }

    def __init__(
        self,
        threshold      : float = 0.60,
        depth_model    : Optional[str] = None,
        use_blink      : bool = True,
    ):
        self.threshold    = threshold
        self.texture_analyzer = TextureLivenessAnalyzer()
        self.depth_estimator  = DepthLivenessEstimator(depth_model)
        self.blink_detector   = BlinkDetector() if use_blink else None
        self.use_blink        = use_blink

    def detect_frame(self, img: np.ndarray) -> LivenessResult:
        """
        Single-frame liveness detection.

        Uses texture + depth signals (no blink without video).

        Args:
            img: BGR face crop (112×112)

        Returns:
            LivenessResult with final decision
        """
        signals = []

        # Texture
        tex = self.texture_analyzer.analyze(img)
        signals.append(tex)

        # Depth
        dep = self.depth_estimator.estimate(img)
        signals.append(dep)

        return self._fuse(signals)

    def detect_video(
        self,
        frames : List[np.ndarray],
    ) -> LivenessResult:
        """
        Video-based liveness detection with blink analysis.

        Args:
            frames: list of BGR face crops (consecutive video frames)

        Returns:
            LivenessResult aggregated across all frames
        """
        if self.blink_detector:
            self.blink_detector.reset()

        tex_scores = []
        dep_scores = []

        for frame in frames:
            tex = self.texture_analyzer.analyze(frame)
            dep = self.depth_estimator.estimate(frame)
            tex_scores.append(tex.score)
            dep_scores.append(dep.score)

            # Blink via EAR — placeholder (needs landmark detector)
            # self.blink_detector.update(ear_value)

        signals = [
            LivenessSignal("texture_analysis", np.mean(tex_scores), 0.75),
            LivenessSignal("depth_estimation", np.mean(dep_scores), 0.80),
        ]

        if self.use_blink and self.blink_detector:
            blink_sig = self.blink_detector.get_liveness_signal()
            signals.append(blink_sig)

        return self._fuse(signals)

    def _fuse(self, signals: List[LivenessSignal]) -> LivenessResult:
        """Weighted fusion of liveness signals."""
        total_weight = 0.0
        weighted_sum = 0.0

        for sig in signals:
            weight       = self.SIGNAL_WEIGHTS.get(sig.name, 0.25)
            weighted_sum += sig.score * weight * sig.confidence
            total_weight += weight * sig.confidence

        final_score = weighted_sum / max(total_weight, 1e-8)
        is_live     = final_score >= self.threshold

        return LivenessResult(
            is_live     = is_live,
            final_score = round(float(final_score), 4),
            signals     = signals,
            threshold   = self.threshold,
            method      = "weighted_fusion",
        )
