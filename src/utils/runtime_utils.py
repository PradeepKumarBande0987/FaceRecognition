from __future__ import annotations

import os
import random
import sys
from datetime import datetime, timezone

import cv2
import numpy as np
import torch


def get_device(requested: str = "cuda") -> torch.device:
    """Resolve device with safe CUDA fallback."""
    if requested == "cuda" and not torch.cuda.is_available():
        return torch.device("cpu")
    return torch.device(requested)


def set_seeds(seed: int = 42):
    """Seed random generators for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def print_banner():
    """Print startup banner."""
    ts = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    print("\n" + "╔" + "═" * 63 + "╗")
    print("║" + "  FACE RECOGNITION SYSTEM  v3.0.0".center(63) + "║")
    print("║" + "  LFW + Webcam Real-Time Recognition".center(63) + "║")
    print("║" + f"  {ts}".center(63) + "║")
    print("╚" + "═" * 63 + "╝")


def print_system_info():
    """Print environment summary."""
    cuda = torch.cuda.is_available()
    print("\n" + "─" * 65)
    print("  🖥️  SYSTEM")
    print("─" * 65)
    print(f"  Python  : {sys.version.split()[0]}")
    print(f"  PyTorch : {torch.__version__}")
    print(f"  OpenCV  : {cv2.__version__}")
    print(f"  CUDA    : {cuda}")
    if cuda:
        for i in range(torch.cuda.device_count()):
            p = torch.cuda.get_device_properties(i)
            print(f"  GPU[{i}]  : {p.name} ({p.total_memory/1e9:.1f} GB)")
    else:
        print(f"  CPU     : {os.cpu_count()} cores")
    print("─" * 65 + "\n")
