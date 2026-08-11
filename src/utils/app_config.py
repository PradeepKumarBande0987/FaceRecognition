from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch


@dataclass
class Config:
    """Master runtime configuration for webcam + LFW flow."""

    mode: str = "lfw_webcam"

    device: str = "cuda"

    backbone: str = "resnet50"
    embedding_dim: int = 512
    pretrained: bool = True
    checkpoint: Optional[str] = None

    lfw_dir: str = "data/raw/lfw"
    no_download: bool = False
    max_identities: int = 100
    min_images: int = 3
    max_per_id: int = 5
    reset_db: bool = False
    run_lfw_eval: bool = True
    eval_pairs: int = 500

    webcam_id: int = 0
    frame_width: int = 1280
    frame_height: int = 720
    fps: int = 30

    threshold: float = 0.45
    top_k: int = 3

    enable_security: bool = True
    liveness_threshold: float = 0.30
    antispoof_threshold: float = 0.30
    adversarial_threshold: float = 0.50

    show_topk: bool = True
    show_fps: bool = True
    show_security: bool = True

    detector: str = "haar"

    seed: int = 42

    def resolve_device(self) -> torch.device:
        if self.device == "cuda" and not torch.cuda.is_available():
            self.device = "cpu"
        return torch.device(self.device)
