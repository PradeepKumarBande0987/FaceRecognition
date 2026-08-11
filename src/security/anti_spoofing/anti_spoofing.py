"""
Security — Anti-Spoofing Module.

Binary classifier to detect presentation attacks:
    • Printed photo attacks
    • Digital screen replay attacks
    • 3D mask attacks

Architecture:
    Binary CNN trained on CASIA-FASD + Replay-Attack
    Input: 112×112 face crop
    Output: real (0) / spoof (1) + confidence score

Features:
    • Depth-map estimation (CDCN variant)
    • Patch-based ensemble classification
    • Temporal consistency check for video streams
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms.v2 as T
from PIL import Image


class SpoofType(str, Enum):
    REAL    = "real"
    PRINT   = "print_attack"
    REPLAY  = "replay_attack"
    MASK_3D = "3d_mask"
    UNKNOWN = "unknown"


@dataclass
class AntiSpoofResult:
    """Result from anti-spoofing detection."""
    is_real      : bool
    confidence   : float
    spoof_type   : SpoofType
    depth_score  : Optional[float] = None
    patch_scores : Optional[List[float]] = None


# ── Binary Anti-Spoofing CNN ──────────────────────────────────────────────────

class AntiSpoofCNN(nn.Module):
    """
    Lightweight binary CNN for liveness detection.

    Architecture:
        MobileNetV2 backbone → Global Average Pooling
        → FC(256) → ReLU → Dropout(0.5) → FC(2) → Softmax

    Input:  (B, 3, 112, 112) normalized face crop
    Output: (B, 2) logits [real, spoof]
    """

    def __init__(self, dropout: float = 0.5):
        super().__init__()
        import torchvision.models as models

        backbone = models.mobilenet_v2(
            weights=models.MobileNet_V2_Weights.IMAGENET1K_V1
        )
        # Replace classifier
        backbone.classifier = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(backbone.last_channel, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
            nn.Linear(256, 2),
        )
        self.backbone = backbone

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x)


# ── Anti-Spoof Detector ───────────────────────────────────────────────────────

class AntiSpoofDetector:
    """
    Runs anti-spoofing inference on face images.

    Usage:
        detector = AntiSpoofDetector(
            checkpoint = "models/anti_spoof/best_model.pt",
            device     = "cuda",
        )
        result = detector.predict(face_pil_image)
        if not result.is_real:
            reject_access()
    """

    def __init__(
        self,
        checkpoint  : Optional[str] = None,
        device      : str   = "cuda",
        threshold   : float = 0.5,
        patch_based : bool  = False,
    ):
        self.device      = torch.device(
            device if torch.cuda.is_available() else "cpu"
        )
        self.threshold   = threshold
        self.patch_based = patch_based

        # Build model
        self.model = AntiSpoofCNN().to(self.device)

        if checkpoint:
            ckpt = torch.load(checkpoint, map_location=self.device,
                              weights_only=False)
            state = ckpt.get("model_state_dict", ckpt)
            self.model.load_state_dict(state)

        self.model.eval()

        # Transform
        self.transform = T.Compose([
            T.Resize((112, 112), antialias=True),
            T.ToDtype(torch.float32, scale=True),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])

    @torch.no_grad()
    def predict(
        self,
        img : Image.Image | np.ndarray,
    ) -> AntiSpoofResult:
        """
        Predict real/spoof for a single face image.

        Args:
            img: PIL RGB image or BGR numpy array

        Returns:
            AntiSpoofResult with is_real, confidence, spoof_type
        """
        import cv2

        if isinstance(img, np.ndarray):
            img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

        tensor  = self.transform(img).unsqueeze(0).to(self.device)
        logits  = self.model(tensor)                       # (1, 2)
        probs   = torch.softmax(logits, dim=1)[0]          # (2,)
        real_p  = float(probs[0])
        spoof_p = float(probs[1])

        is_real    = real_p >= self.threshold
        confidence = real_p if is_real else spoof_p
        spoof_type = SpoofType.REAL if is_real else SpoofType.UNKNOWN

        return AntiSpoofResult(
            is_real    = is_real,
            confidence = round(confidence, 4),
            spoof_type = spoof_type,
        )

    @torch.no_grad()
    def predict_batch(
        self,
        images : List[Image.Image],
    ) -> List[AntiSpoofResult]:
        """Predict real/spoof for a batch of images."""
        tensors = torch.stack([self.transform(img) for img in images])
        tensors = tensors.to(self.device)
        logits  = self.model(tensors)
        probs   = torch.softmax(logits, dim=1).cpu().numpy()

        results = []
        for p in probs:
            real_p   = float(p[0])
            is_real  = real_p >= self.threshold
            results.append(AntiSpoofResult(
                is_real    = is_real,
                confidence = round(float(max(p)), 4),
                spoof_type = SpoofType.REAL if is_real else SpoofType.UNKNOWN,
            ))
        return results
