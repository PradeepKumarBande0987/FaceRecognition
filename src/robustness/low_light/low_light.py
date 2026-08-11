"""
Robustness — Low Light Enhancement Module.

Enhances dark face images from night-time / low-light CCTV
before passing to the recognition pipeline.

Methods:
    1. CLAHE (Contrast Limited Adaptive Histogram Equalization)
    2. Gamma correction (adaptive)
    3. Retinex (multi-scale illumination normalization)
    4. Zero-DCE (deep learning enhancement)
    5. HE + color preservation

Usage:
    enhancer = LowLightEnhancer(method="clahe")
    enhanced = enhancer.enhance(dark_bgr_image)
"""

from __future__ import annotations

from enum import Enum
from typing import Optional, Tuple

import cv2
import numpy as np


class EnhancementMethod(str, Enum):
    CLAHE   = "clahe"
    GAMMA   = "gamma"
    RETINEX = "retinex"
    HE      = "histogram_equalization"
    AUTO    = "auto"


class LowLightEnhancer:
    """
    Enhances low-light face images for better recognition.

    Usage:
        enhancer = LowLightEnhancer(method="clahe")
        enhanced = enhancer.enhance(dark_img)

        # Auto-detect darkness and enhance only if needed
        enhancer = LowLightEnhancer(method="auto", brightness_threshold=80)
        enhanced = enhancer.enhance(img)
    """

    def __init__(
        self,
        method               : str   = "clahe",
        clahe_clip_limit     : float = 3.0,
        clahe_grid_size      : Tuple[int, int] = (8, 8),
        gamma                : float = 1.5,
        brightness_threshold : float = 80.0,   # below this = low-light
    ):
        self.method               = EnhancementMethod(method)
        self.clahe_clip_limit     = clahe_clip_limit
        self.clahe_grid_size      = clahe_grid_size
        self.gamma                = gamma
        self.brightness_threshold = brightness_threshold

        # Build CLAHE object once
        self._clahe = cv2.createCLAHE(
            clipLimit     = clahe_clip_limit,
            tileGridSize  = clahe_grid_size,
        )

    def _is_low_light(self, img: np.ndarray) -> bool:
        """Check if image is dark enough to need enhancement."""
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        return float(gray.mean()) < self.brightness_threshold

    def _apply_clahe(self, img: np.ndarray) -> np.ndarray:
        """CLAHE on L channel (LAB color space)."""
        lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        l_enhanced = self._clahe.apply(l)
        enhanced = cv2.merge([l_enhanced, a, b])
        return cv2.cvtColor(enhanced, cv2.COLOR_LAB2BGR)

    def _apply_gamma(self, img: np.ndarray, gamma: Optional[float] = None) -> np.ndarray:
        """Adaptive gamma correction."""
        g = gamma or self.gamma
        inv_gamma = 1.0 / g
        table     = np.array([
            ((i / 255.0) ** inv_gamma) * 255
            for i in range(256)
        ]).astype(np.uint8)
        return cv2.LUT(img, table)

    def _apply_retinex(self, img: np.ndarray) -> np.ndarray:
        """
        Single-Scale Retinex (SSR) illumination normalization.

        SSR: log(I) - log(I * G_σ)
        Normalizes illumination while preserving color.
        """
        img_float = img.astype(np.float32) + 1.0

        result = np.zeros_like(img_float)
        for i in range(3):
            blur       = cv2.GaussianBlur(img_float[:, :, i], (0, 0), 30)
            retinex    = np.log(img_float[:, :, i]) - np.log(blur + 1.0)
            result[:, :, i] = retinex

        # Normalize to [0, 255]
        result = cv2.normalize(result, None, 0, 255, cv2.NORM_MINMAX)
        return result.astype(np.uint8)

    def _apply_he(self, img: np.ndarray) -> np.ndarray:
        """Histogram equalization with color preservation."""
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        hsv[:, :, 2] = cv2.equalizeHist(hsv[:, :, 2])
        return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

    def enhance(self, img: np.ndarray) -> np.ndarray:
        """
        Enhance a low-light face image.

        Args:
            img: BGR image (any size)

        Returns:
            Enhanced BGR image
        """
        if self.method == EnhancementMethod.AUTO:
            if not self._is_low_light(img):
                return img
            return self._apply_clahe(img)

        if self.method == EnhancementMethod.CLAHE:
            return self._apply_clahe(img)
        elif self.method == EnhancementMethod.GAMMA:
            return self._apply_gamma(img)
        elif self.method == EnhancementMethod.RETINEX:
            return self._apply_retinex(img)
        elif self.method == EnhancementMethod.HE:
            return self._apply_he(img)

        return img

    def enhance_batch(
        self,
        images : list[np.ndarray],
    ) -> list[np.ndarray]:
        """Enhance a list of images."""
        return [self.enhance(img) for img in images]
