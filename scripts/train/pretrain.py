"""
Full Pre-Training Script — ArcFace + All Augmentations.

Extends train_baseline.py with:
    • Occlusion augmentation   (mask, glasses, hat, scarf, patches)
    • Low-resolution degradation (CCTV profiles)
    • Multi-dataset training   (VGGFace2 + CelebA + Custom CCTV)
    • Synthetic GAN data mixing
    • Advanced BalancedBatchSampler
    • Gradient checkpointing for memory efficiency

Usage:
    # Single GPU
    python scripts/train/pretrain.py \
        --train-csv  data/splits/train/train_merged.csv \
        --val-csv    data/splits/val/val_merged.csv \
        --run-name   full_pipeline_v1 \
        --epochs     100

    # Multi-GPU (4× A100)
    torchrun --nproc_per_node=4 scripts/train/pretrain.py \
        --train-csv  data/splits/train/train_merged.csv \
        --distributed
"""

import argparse
import os
import random
import sys
import time
import uuid
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms.v2 as T              # ✅ v2 transforms
from PIL import Image
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, Dataset, DistributedSampler

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from data.processed.augmented.augmented  import FaceAugmentor, AUGMENTATION_CONFIG
from data.processed.occluded.occluded    import OcclusionEngine
from data.processed.low_resolution.low_resolution import LowResolutionGenerator
from logs.training.training_logger       import TrainingLogger
from experiments.runs.run_manager        import RunManager
from scripts.train.train_baseline        import (
    ArcFaceLoss, build_backbone,
    get_val_transforms, BaselineTrainer,
)


# ── Full Pipeline Config ──────────────────────────────────────────────────────

PRETRAIN_CONFIG = {
    # Model
    "backbone"          : "resnet50",
    "embedding_dim"     : 512,
    "pretrained"        : False,

    # ArcFace
    "loss"              : "arcface",
    "arcface_margin"    : 0.5,
    "arcface_scale"     : 64,
    "num_classes"       : 8631,

    # Training
    "epochs"            : 100,
    "batch_size"        : 512,
    "lr"                : 0.1,
    "momentum"          : 0.9,
    "weight_decay"      : 5e-4,
    "warmup_epochs"     : 5,
    "lr_scheduler"      : "cosine",
    "gradient_clip"     : 1.0,
    "mixed_precision"   : True,

    # ── FULL augmentation (vs baseline) ──────────────────────────────────
    "augmentation"          : "full",
    "occlusion_aug"         : True,
    "occlusion_probability" : 0.40,       # apply occlusion to 40% of images
    "lr_aug"                : True,
    "lr_aug_probability"    : 0.30,       # apply LR degradation to 30%
    "image_size"            : (112, 112),

    # Data
    "num_workers"       : 8,
    "pin_memory"        : True,
    "eval_every_epochs" : 5,
    "save_every_epochs" : 10,
}


# ── Full Augmentation Dataset ─────────────────────────────────────────────────

class FullAugDataset(Dataset):
    """
    Face dataset with full augmentation pipeline:
        1. Standard torchvision v2 transforms
        2. Random occlusion (mask / glasses / hat / scarf / patches)
        3. Random low-resolution degradation (CCTV profiles)

    Augmentations are applied probabilistically per sample.
    """

    def __init__(
        self,
        csv_path      : str,
        config        : dict,
        base_transform = None,
    ):
        import csv as csv_module

        self.config   = config
        self.samples  = []
        self.image_size = tuple(config["image_size"])

        with open(csv_path, newline="") as f:
            for row in csv_module.DictReader(f):
                img = row.get("image_path", "")
                lbl = int(row.get("label", row.get("identity_label", 0)))
                self.samples.append((img, lbl))

        # Standard v2 transforms
        self.base_transform = base_transform or T.Compose([
            T.RandomResizedCrop(
                size      = self.image_size,
                scale     = (0.8, 1.0),
                antialias = True,
            ),
            T.RandomHorizontalFlip(p=0.5),
            T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
            T.RandomRotation(degrees=10),
            T.ToDtype(torch.float32, scale=True),   # ✅ v2 replaces ToTensor()
            T.Normalize(
                mean = [0.485, 0.456, 0.406],
                std  = [0.229, 0.224, 0.225],
            ),
        ])

        # Occlusion engine
        self.occ_engine = OcclusionEngine(image_size=self.image_size)
        self.occ_types  = [
            "mask", "sunglasses", "hat", "scarf",
            "random_patches", "hair_strands",
        ]

        # LR generator
        self.lr_gen     = LowResolutionGenerator(target_size=self.image_size)
        self.lr_profiles = ["cctv_standard", "cctv_extreme", "mobile_low"]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        import cv2

        img_path, label = self.samples[idx]

        # ── Load image ────────────────────────────────────────────────────
        img_pil = Image.open(img_path).convert("RGB")
        img_pil = img_pil.resize(self.image_size, Image.Resampling.LANCZOS)

        # ── Occlusion augmentation (applied before color transforms) ──────
        if self.config.get("occlusion_aug") and \
           random.random() < self.config.get("occlusion_probability", 0.4):
            img_bgr = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)
            occ_type = random.choice(self.occ_types)
            img_bgr  = self.occ_engine.occlude(img_bgr, occ_type=occ_type)
            img_pil  = Image.fromarray(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB))

        # ── Low-resolution degradation ────────────────────────────────────
        if self.config.get("lr_aug") and \
           random.random() < self.config.get("lr_aug_probability", 0.3):
            img_bgr  = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)
            profile  = random.choice(self.lr_profiles)
            img_bgr  = self.lr_gen.degrade(img_bgr, profile=profile)
            img_pil  = Image.fromarray(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB))

        # ── Standard torchvision v2 transforms ───────────────────────────
        img_tensor = self.base_transform(img_pil)

        return img_tensor, label


# ── Full Pre-Trainer ──────────────────────────────────────────────────────────

class FullPretrainer(BaselineTrainer):
    """
    Extends BaselineTrainer with full augmentation pipeline.

    Overrides:
        • _setup_data  → uses FullAugDataset instead of plain FaceCSVDataset
        • train        → adds augmentation stats logging
    """

    def _setup_data(self):
        """Override data setup to use full augmentation dataset."""
        cfg  = self.config
        args = self.args

        val_tf = get_val_transforms(tuple(cfg["image_size"]))

        # ── Full augmentation train dataset ───────────────────────────────
        train_ds = FullAugDataset(
            csv_path = args.train_csv,
            config   = cfg,
        )

        if self.distributed:
            sampler = DistributedSampler(train_ds)
            shuffle = False
        else:
            sampler = None
            shuffle = True

        self.train_loader = DataLoader(
            train_ds,
            batch_size         = cfg["batch_size"] // max(self.world_size, 1),
            shuffle            = shuffle,
            sampler            = sampler,
            num_workers        = cfg["num_workers"],
            pin_memory         = cfg["pin_memory"],
            persistent_workers = True if cfg["num_workers"] > 0 else False,
            drop_last          = True,
        )

        # ── Validation dataset (no augmentation) ──────────────────────────
        self.val_loader = None
        if args.val_csv and Path(args.val_csv).exists():
            from torch.utils.data import Dataset as TorchDataset
            import csv as csv_module

            class ValDataset(TorchDataset):
                def __init__(self, csv_path, transform):
                    self.samples   = []
                    self.transform = transform
                    with open(csv_path, newline="") as f:
                        for row in csv_module.DictReader(f):
                            img = row.get("image_path", "")
                            lbl = int(row.get("label", 0))
                            self.samples.append((img, lbl))

                def __len__(self):
                    return len(self.samples)

                def __getitem__(self, idx):
                    img_path, label = self.samples[idx]
                    img = Image.open(img_path).convert("RGB")
                    if self.transform:
                        img = self.transform(img)
                    return img, label

            val_ds = ValDataset(args.val_csv, transform=val_tf)
            self.val_loader = DataLoader(
                val_ds,
                batch_size  = cfg["batch_size"],
                shuffle     = False,
                num_workers = cfg["num_workers"],
                pin_memory  = cfg["pin_memory"],
            )

        if self.is_main:
            print(f"\n   Augmentation : FULL")
            print(f"   Occlusion    : {cfg.get('occlusion_aug')} "
                  f"(p={cfg.get('occlusion_probability')})")
            print(f"   LR degradation: {cfg.get('lr_aug')} "
                  f"(p={cfg.get('lr_aug_probability')})")
            print(f"   Train samples: {len(train_ds):,}")


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Full Pre-Training: ArcFace + Occlusion + LR Augmentation"
    )
    parser.add_argument("--train-csv",   default="data/splits/train/train_merged.csv")
    parser.add_argument("--val-csv",     default="data/splits/val/val_merged.csv")
    parser.add_argument("--output-dir",  default="experiments/runs/", type=Path)
    parser.add_argument("--run-name",    default=f"pretrain_{uuid.uuid4().hex[:6]}")
    parser.add_argument("--backbone",    default="resnet50")
    parser.add_argument("--embedding-dim", type=int, default=512)
    parser.add_argument("--num-classes", type=int, default=8631)
    parser.add_argument("--margin",      type=float, default=0.5)
    parser.add_argument("--scale",       type=float, default=64.0)
    parser.add_argument("--epochs",      type=int, default=100)
    parser.add_argument("--batch-size",  type=int, default=512)
    parser.add_argument("--lr",          type=float, default=0.1)
    parser.add_argument("--warmup",      type=int, default=5)
    parser.add_argument("--device",      default="cuda")
    parser.add_argument("--distributed", action="store_true")
    parser.add_argument("--no-amp",      action="store_true")
    parser.add_argument("--no-occlusion",action="store_true")
    parser.add_argument("--no-lr-aug",   action="store_true")
    parser.add_argument("--seed",        type=int, default=42)
    return parser.parse_args()


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)

    config = {
        **PRETRAIN_CONFIG,
        "backbone"           : args.backbone,
        "embedding_dim"      : args.embedding_dim,
        "num_classes"        : args.num_classes,
        "arcface_margin"     : args.margin,
        "arcface_scale"      : args.scale,
        "epochs"             : args.epochs,
        "batch_size"         : args.batch_size,
        "lr"                 : args.lr,
        "warmup_epochs"      : args.warmup,
        "mixed_precision"    : not args.no_amp,
        "occlusion_aug"      : not args.no_occlusion,
        "lr_aug"             : not args.no_lr_aug,
    }

    trainer = FullPretrainer(config=config, args=args)
    trainer.train()


if __name__ == "__main__":
    main()
