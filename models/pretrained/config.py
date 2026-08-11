"""
Configuration management for pre-training.
"""

from dataclasses import dataclass
from typing import Optional


@dataclass
class TrainingConfig:
    """Training configuration."""
    # Paths
    train_data: str = "./data/train"
    test_data: Optional[str] = "./data/test"
    checkpoint_dir: str = "./checkpoints"

    # Model
    backbone: str = "resnet50"  # resnet50, mobilefacenet, etc
    embedding_dim: int = 512
    loss_fn: str = "arcface"  # arcface, cosface, sphereface, triplet

    # Loss parameters
    arcface_margin: float = 0.5
    arcface_scale: float = 64.0
    cosface_margin: float = 0.35
    cosface_scale: float = 64.0
    sphereface_margin: int = 4
    triplet_margin: float = 0.2

    # Training
    batch_size: int = 128
    num_epochs: int = 100
    warmup_epochs: int = 5
    learning_rate: float = 0.1
    momentum: float = 0.9
    weight_decay: float = 5e-4

    # Optimization
    lr_scheduler: str = "cosine"  # cosine, step
    gradient_clip: float = 1.0
    mixed_precision: bool = True
    num_workers: int = 8
    pin_memory: bool = True

    # Data
    image_size: tuple = (112, 112)
    balanced_sampling: bool = True

    # Hardware
    device: str = "cuda"
    distributed: bool = False
