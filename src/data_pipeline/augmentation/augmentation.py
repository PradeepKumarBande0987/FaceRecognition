"""
Data Pipeline — Augmentation Module.

Provides a unified augmentation interface for face recognition training.
Wraps torchvision v2 transforms + custom OpenCV augmentations.

Augmentation Levels:
    • light    : flip + slight color jitter
    • standard : flip + rotation + color jitter + blur
    • heavy    : standard + occlusion + LR degradation + noise
    • custom   : fully configurable via AugConfig dataclass
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional, Tuple

import cv2
import numpy as np
import torch
import torchvision.transforms.v2 as T
from PIL import Image


# ── Augmentation Level ────────────────────────────────────────────────────────

class AugLevel(str, Enum):
    LIGHT    = "light"
    STANDARD = "standard"
    HEAVY    = "heavy"
    CUSTOM   = "custom"


# ── Config ────────────────────────────────────────────────────────────────────

@dataclass
class AugConfig:
    """Full augmentation configuration."""

    image_size          : Tuple[int, int] = (112, 112)
    level               : AugLevel        = AugLevel.STANDARD

    # Geometric
    p_hflip             : float = 0.50
    p_rotation          : float = 0.60
    rotation_degrees    : int   = 15
    p_perspective       : float = 0.20

    # Color
    p_color_jitter      : float = 0.70
    brightness          : float = 0.30
    contrast            : float = 0.30
    saturation          : float = 0.20
    hue                 : float = 0.10
    p_grayscale         : float = 0.05

    # Noise & Blur
    p_gaussian_blur     : float = 0.30
    p_gaussian_noise    : float = 0.40
    noise_std_range     : Tuple[float, float] = (5.0, 25.0)

    # Compression
    p_jpeg_compression  : float = 0.30
    jpeg_quality_range  : Tuple[int, int] = (30, 80)

    # Occlusion (heavy only)
    p_occlusion         : float = 0.40
    occlusion_types     : list = field(default_factory=lambda: [
        "mask", "sunglasses", "hat", "random_patches"
    ])

    # LR Degradation (heavy only)
    p_lr_degradation    : float = 0.30
    lr_profiles         : list  = field(default_factory=lambda: [
        "cctv_standard", "cctv_extreme", "mobile_low"
    ])

    # Random Erasing
    p_random_erase      : float = 0.30
    erase_scale         : Tuple[float, float] = (0.02, 0.15)

    # Normalization
    mean                : Tuple[float, ...] = (0.485, 0.456, 0.406)
    std                 : Tuple[float, ...] = (0.229, 0.224, 0.225)


# ── Preset Configs ────────────────────────────────────────────────────────────

AUGMENTATION_PRESETS: dict[str, AugConfig] = {
    AugLevel.LIGHT: AugConfig(
        level           = AugLevel.LIGHT,
        p_hflip         = 0.5,
        p_rotation      = 0.0,
        p_color_jitter  = 0.3,
        brightness      = 0.1,
        contrast        = 0.1,
        saturation      = 0.1,
        hue             = 0.0,
        p_gaussian_blur = 0.0,
        p_gaussian_noise= 0.0,
        p_jpeg_compression=0.0,
        p_occlusion     = 0.0,
        p_lr_degradation= 0.0,
        p_random_erase  = 0.0,
    ),
    AugLevel.STANDARD: AugConfig(
        level           = AugLevel.STANDARD,
        p_occlusion     = 0.0,
        p_lr_degradation= 0.0,
    ),
    AugLevel.HEAVY: AugConfig(
        level           = AugLevel.HEAVY,
        p_occlusion     = 0.40,
        p_lr_degradation= 0.30,
    ),
}


# ── Core Augmentor ────────────────────────────────────────────────────────────

class FaceAugmentor:
    """
    Unified face augmentation engine.

    Combines torchvision v2 transforms with OpenCV-based
    augmentations in a configurable pipeline.

    Usage:
        aug = FaceAugmentor(level="heavy")
        tensor = aug(pil_image)          # returns (C, H, W) tensor

        aug = FaceAugmentor(config=custom_config)
    """

    def __init__(
        self,
        level  : str       = "standard",
        config : Optional[AugConfig] = None,
        seed   : int       = 42,
    ):
        self.config = config or AUGMENTATION_PRESETS.get(
            level, AUGMENTATION_PRESETS[AugLevel.STANDARD]
        )
        random.seed(seed)
        np.random.seed(seed)
        self._build_torchvision_pipeline()

    def _build_torchvision_pipeline(self):
        """Build torchvision v2 transform pipeline from config."""
        cfg = self.config
        H, W = cfg.image_size

        self._tv_pipeline = T.Compose([
            T.RandomHorizontalFlip(p=cfg.p_hflip),
            T.RandomRotation(degrees=cfg.rotation_degrees),
            T.ColorJitter(
                brightness = cfg.brightness,
                contrast   = cfg.contrast,
                saturation = cfg.saturation,
                hue        = cfg.hue,
            ),
            T.RandomPerspective(distortion_scale=0.3, p=cfg.p_perspective),
            T.RandomGrayscale(p=cfg.p_grayscale),
            T.GaussianBlur(kernel_size=5, sigma=(0.1, 2.0)),
            T.RandomErasing(
                p       = cfg.p_random_erase,
                scale   = cfg.erase_scale,
                ratio   = (0.3, 3.3),
                value   = 0,
            ),
            T.Resize((H, W), antialias=True),           # ✅ v2 required
            T.ToDtype(torch.float32, scale=True),        # ✅ replaces ToTensor
            T.Normalize(mean=list(cfg.mean), std=list(cfg.std)),
        ])

    # ── OpenCV Augmentations ──────────────────────────────────────────────────

    def _add_gaussian_noise(self, img: np.ndarray) -> np.ndarray:
        lo, hi = self.config.noise_std_range
        std    = random.uniform(lo, hi)
        noise  = np.random.normal(0, std, img.shape).astype(np.float32)
        return np.clip(img.astype(np.float32) + noise, 0, 255).astype(np.uint8)

    def _jpeg_compress(self, img: np.ndarray) -> np.ndarray:
        lo, hi   = self.config.jpeg_quality_range
        quality  = random.randint(lo, hi)
        params   = [int(cv2.IMWRITE_JPEG_QUALITY), quality]
        _, enc   = cv2.imencode(".jpg", img, params)
        return cv2.imdecode(enc, cv2.IMREAD_COLOR)

    def _lr_degrade(self, img: np.ndarray) -> np.ndarray:
        """Simulate low-resolution CCTV-like degradation."""
        h, w    = img.shape[:2]
        scale   = random.choice([2, 3, 4])
        small   = cv2.resize(img, (max(1, w // scale), max(1, h // scale)),
                             interpolation=cv2.INTER_AREA)
        noise   = np.random.normal(0, 10, small.shape).astype(np.float32)
        small   = np.clip(small.astype(np.float32) + noise, 0, 255).astype(np.uint8)
        return cv2.resize(small, (w, h), interpolation=cv2.INTER_CUBIC)

    def _apply_cv2_augs(self, img: np.ndarray) -> np.ndarray:
        cfg = self.config
        if random.random() < cfg.p_gaussian_noise:
            img = self._add_gaussian_noise(img)
        if random.random() < cfg.p_jpeg_compression:
            img = self._jpeg_compress(img)
        if random.random() < cfg.p_lr_degradation:
            img = self._lr_degrade(img)
        return img

    # ── Main __call__ ─────────────────────────────────────────────────────────

    def __call__(self, img: Image.Image) -> torch.Tensor:
        """
        Apply full augmentation pipeline.

        Args:
            img: PIL RGB image

        Returns:
            Augmented (C, H, W) float32 tensor, normalized
        """
        # Convert PIL → cv2 for OpenCV augmentations
        cv2_img = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
        cv2_img = self._apply_cv2_augs(cv2_img)

        # Convert back → PIL → torchvision v2 pipeline
        pil_img = Image.fromarray(cv2.cvtColor(cv2_img, cv2.COLOR_BGR2RGB))
        return self._tv_pipeline(pil_img)

    def get_val_transform(self) -> T.Compose:
        """Return deterministic validation transform (no augmentation)."""
        H, W = self.config.image_size
        return T.Compose([
            T.Resize((H, W), antialias=True),
            T.ToDtype(torch.float32, scale=True),
            T.Normalize(mean=list(self.config.mean), std=list(self.config.std)),
        ])
