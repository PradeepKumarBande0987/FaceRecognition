"""
Baseline Face Recognition Training Script.

Trains a face recognition model from scratch using:
    • ArcFace loss (additive angular margin)
    • ResNet-50 backbone (IResNet variant)
    • VGGFace2 + CelebA training data
    • Standard augmentation pipeline
    • Mixed precision (torch.amp — PyTorch 2.4+)
    • Cosine annealing LR schedule with warmup

This is the BASELINE experiment — no occlusion or
low-resolution augmentation applied.
Compare results against pretrain.py (full pipeline).

Usage:
    # Single GPU
    python scripts/train/train_baseline.py \
        --train-csv   data/splits/train/vggface2_train.csv \
        --val-csv     data/splits/val/vggface2_val.csv \
        --output-dir  experiments/runs/ \
        --run-name    baseline_resnet50_v1 \
        --epochs      100 \
        --batch-size  512 \
        --lr          0.1 \
        --device      cuda

    # Multi-GPU (torchrun)
    torchrun --nproc_per_node=4 scripts/train/train_baseline.py \
        --train-csv  data/splits/train/vggface2_train.csv \
        --distributed
"""

import argparse
import json
import os
import sys
import time
import uuid
from pathlib import Path

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms.v2 as T              # ✅ v2 transforms
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler

# ── Project Imports ───────────────────────────────────────────────────────────
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from logs.training.training_logger import TrainingLogger
from experiments.runs.run_manager import RunManager


# ── Baseline Config ───────────────────────────────────────────────────────────

BASELINE_CONFIG = {
    # Model
    "backbone"          : "resnet50",
    "embedding_dim"     : 512,
    "pretrained"        : False,

    # ArcFace head
    "loss"              : "arcface",
    "arcface_margin"    : 0.5,
    "arcface_scale"     : 64,
    "num_classes"       : 8631,           # VGGFace2 train identities

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

    # Augmentation (baseline = standard only, NO occlusion/LR)
    "augmentation"      : "standard",
    "image_size"        : (112, 112),

    # Data
    "num_workers"       : 8,
    "pin_memory"        : True,

    # Eval
    "eval_every_epochs" : 5,
    "save_every_epochs" : 10,
}


# ── Transforms ────────────────────────────────────────────────────────────────

def get_train_transforms(image_size=(112, 112)) -> T.Compose:
    """
    Standard baseline augmentation — no occlusion, no LR degradation.

    Pipeline:
        RandomResizedCrop → HorizontalFlip → ColorJitter
        → RandomRotation → ToDtype → Normalize
    """
    return T.Compose([
        T.RandomResizedCrop(
            size      = image_size,
            scale     = (0.8, 1.0),
            ratio     = (0.95, 1.05),
            antialias = True,               # ✅ required in torchvision v2
        ),
        T.RandomHorizontalFlip(p=0.5),
        T.ColorJitter(
            brightness = 0.2,
            contrast   = 0.2,
            saturation = 0.2,
            hue        = 0.1,
        ),
        T.RandomRotation(degrees=10),
        T.ToDtype(torch.float32, scale=True),   # ✅ replaces ToTensor()
        T.Normalize(
            mean = [0.485, 0.456, 0.406],
            std  = [0.229, 0.224, 0.225],
        ),
    ])


def get_val_transforms(image_size=(112, 112)) -> T.Compose:
    """Validation transforms — deterministic, no augmentation."""
    return T.Compose([
        T.Resize(image_size, antialias=True),
        T.ToDtype(torch.float32, scale=True),
        T.Normalize(
            mean = [0.485, 0.456, 0.406],
            std  = [0.229, 0.224, 0.225],
        ),
    ])


# ── ArcFace Loss ──────────────────────────────────────────────────────────────

class ArcFaceLoss(nn.Module):
    """
    ArcFace: Additive Angular Margin Loss.

    Reference:
        Deng et al. ArcFace: Additive Angular Margin Loss for
        Deep Face Recognition. CVPR 2019.

    Math:
        L = -log( e^{s·cos(θ_yi + m)} /
                  (e^{s·cos(θ_yi + m)} + Σ_{j≠yi} e^{s·cos(θ_j)}) )

    Args:
        embedding_dim : size of face embedding (e.g. 512)
        num_classes   : number of training identities
        margin_m      : angular margin in radians (default: 0.5)
        scale_s       : feature scale (default: 64)
    """

    def __init__(
        self,
        embedding_dim : int   = 512,
        num_classes   : int   = 8631,
        margin_m      : float = 0.5,
        scale_s       : float = 64.0,
        easy_margin   : bool  = False,
    ):
        super().__init__()
        import math

        self.embedding_dim = embedding_dim
        self.num_classes   = num_classes
        self.margin_m      = margin_m
        self.scale_s       = scale_s
        self.easy_margin   = easy_margin

        # Class weight matrix — shape: (num_classes, embedding_dim)
        self.weight = nn.Parameter(
            torch.FloatTensor(num_classes, embedding_dim)
        )
        nn.init.xavier_uniform_(self.weight)

        # Precompute margin trig values
        self.cos_m = math.cos(margin_m)
        self.sin_m = math.sin(margin_m)
        self.th    = math.cos(math.pi - margin_m)
        self.mm    = math.sin(math.pi - margin_m) * margin_m

        self.ce = nn.CrossEntropyLoss()

    def forward(
        self,
        embeddings : torch.Tensor,   # (B, embedding_dim), L2-normalized
        labels     : torch.Tensor,   # (B,) int64
    ) -> torch.Tensor:
        import torch.nn.functional as F

        # Normalize weights and embeddings
        weight_norm = F.normalize(self.weight, dim=1)
        emb_norm    = F.normalize(embeddings,  dim=1)

        # Cosine similarity: (B, num_classes)
        cos_theta = emb_norm @ weight_norm.t()
        cos_theta = cos_theta.clamp(-1 + 1e-7, 1 - 1e-7)

        # sin(θ) via Pythagorean identity
        sin_theta = torch.sqrt(1.0 - cos_theta.pow(2))

        # cos(θ + m) = cos(θ)·cos(m) - sin(θ)·sin(m)
        cos_theta_m = cos_theta * self.cos_m - sin_theta * self.sin_m

        if self.easy_margin:
            cos_theta_m = torch.where(
                cos_theta > 0, cos_theta_m, cos_theta
            )
        else:
            cos_theta_m = torch.where(
                cos_theta > self.th,
                cos_theta_m,
                cos_theta - self.mm,
            )

        # One-hot encode labels
        one_hot = torch.zeros_like(cos_theta)
        one_hot.scatter_(1, labels.view(-1, 1).long(), 1)

        # Apply margin only to the correct class
        logits = (one_hot * cos_theta_m) + ((1.0 - one_hot) * cos_theta)
        logits = logits * self.scale_s

        return self.ce(logits, labels)


# ── Backbone ──────────────────────────────────────────────────────────────────

def build_backbone(
    name          : str = "resnet50",
    embedding_dim : int = 512,
    pretrained    : bool = False,
) -> nn.Module:
    """
    Build face recognition backbone.

    For the baseline we use a standard ResNet-50 with a
    custom embedding head replacing the final FC layer.

    TODO: Replace with IResNet from models/backbones/arcface.py
    """
    import torchvision.models as models

    if name == "resnet50":
        weights = (
            models.ResNet50_Weights.IMAGENET1K_V2
            if pretrained else None
        )
        backbone = models.resnet50(weights=weights)

        # Replace classifier with embedding head
        in_features = backbone.fc.in_features
        backbone.fc = nn.Sequential(
            nn.BatchNorm1d(in_features),
            nn.Linear(in_features, embedding_dim, bias=False),
            nn.BatchNorm1d(embedding_dim),
        )
    else:
        raise ValueError(f"Unknown backbone: {name}")

    return backbone


# ── Trainer ───────────────────────────────────────────────────────────────────

class BaselineTrainer:
    """
    Baseline face recognition trainer.

    Handles:
        • Single-GPU and multi-GPU (DDP) training
        • ArcFace loss with cosine LR schedule
        • Mixed precision via torch.amp (PyTorch 2.4+)
        • Epoch logging + checkpoint saving
        • LFW evaluation every N epochs

    Usage:
        trainer = BaselineTrainer(config, args)
        trainer.train()
    """

    def __init__(self, config: dict, args: argparse.Namespace):
        self.config = config
        self.args   = args

        # ── Distributed Setup ─────────────────────────────────────────────
        self.distributed = args.distributed
        self.local_rank  = int(os.environ.get("LOCAL_RANK", 0))
        self.world_size  = int(os.environ.get("WORLD_SIZE", 1))
        self.is_main     = (self.local_rank == 0)

        if self.distributed:
            dist.init_process_group(backend="nccl")
            torch.cuda.set_device(self.local_rank)
            self.device = torch.device(f"cuda:{self.local_rank}")
        else:
            self.device = torch.device(
                args.device if torch.cuda.is_available() else "cpu"
            )

        if self.is_main:
            print(f"\n🚀 BaselineTrainer")
            print(f"   Device     : {self.device}")
            print(f"   World size : {self.world_size}")
            print(f"   Backbone   : {config['backbone']}")
            print(f"   Batch size : {config['batch_size']}")
            print(f"   Epochs     : {config['epochs']}\n")

        # ── Run Manager ───────────────────────────────────────────────────
        if self.is_main:
            self.run_manager = RunManager(runs_dir=str(args.output_dir))
            self.run = self.run_manager.create_run(
                config   = config,
                run_name = args.run_name,
                tags     = {"type": "baseline", "backbone": config["backbone"]},
            )
            self.run_dir = Path(self.run.run_dir)

            # Training logger
            self.train_logger = TrainingLogger(
                run_id = self.run.run_id,
                config = config,
            )
        else:
            self.run_dir = Path(args.output_dir)

        # ── Model Setup ───────────────────────────────────────────────────
        self._setup_model()
        self._setup_data()
        self._setup_optimizer()

    # ── Model ─────────────────────────────────────────────────────────────────

    def _setup_model(self):
        """Build backbone + ArcFace loss and move to device."""
        cfg = self.config

        self.backbone = build_backbone(
            name          = cfg["backbone"],
            embedding_dim = cfg["embedding_dim"],
            pretrained    = cfg["pretrained"],
        ).to(self.device)

        self.loss_fn = ArcFaceLoss(
            embedding_dim = cfg["embedding_dim"],
            num_classes   = cfg["num_classes"],
            margin_m      = cfg["arcface_margin"],
            scale_s       = cfg["arcface_scale"],
        ).to(self.device)

        if self.distributed:
            self.backbone = DDP(
                self.backbone,
                device_ids=[self.local_rank],
            )

        # ✅ PyTorch 2.4+: torch.amp.GradScaler with device arg
        self.scaler = torch.amp.GradScaler(
            device  = self.device.type,
            enabled = self.config["mixed_precision"],
        )

        n_params = sum(p.numel() for p in self.backbone.parameters()) / 1e6
        if self.is_main:
            print(f"   Parameters : {n_params:.1f}M")

    # ── Data ──────────────────────────────────────────────────────────────────

    def _setup_data(self):
        """Build DataLoaders from CSV split files."""
        from torch.utils.data import Dataset
        from PIL import Image
        import csv

        cfg  = self.config
        args = self.args

        class FaceCSVDataset(Dataset):
            """Simple CSV-based face dataset."""

            def __init__(self, csv_path: str, transform=None):
                self.samples   = []
                self.transform = transform
                with open(csv_path, newline="") as f:
                    for row in csv.DictReader(f):
                        img  = row.get("image_path", "")
                        lbl  = int(row.get("label", row.get("identity_label", 0)))
                        self.samples.append((img, lbl))

            def __len__(self):
                return len(self.samples)

            def __getitem__(self, idx):
                img_path, label = self.samples[idx]
                img = Image.open(img_path).convert("RGB")
                if self.transform:
                    img = self.transform(img)
                return img, label

        # Build transforms
        train_tf = get_train_transforms(tuple(cfg["image_size"]))
        val_tf   = get_val_transforms(tuple(cfg["image_size"]))

        # Train dataset
        train_ds = FaceCSVDataset(args.train_csv, transform=train_tf)

        if self.distributed:
            train_sampler = DistributedSampler(train_ds)
            shuffle       = False
        else:
            train_sampler = None
            shuffle       = True

        self.train_loader = DataLoader(
            train_ds,
            batch_size      = cfg["batch_size"] // max(self.world_size, 1),
            shuffle         = shuffle,
            sampler         = train_sampler,
            num_workers     = cfg["num_workers"],
            pin_memory      = cfg["pin_memory"],
            persistent_workers = True if cfg["num_workers"] > 0 else False,
            drop_last       = True,
        )

        # Val dataset
        self.val_loader = None
        if args.val_csv and Path(args.val_csv).exists():
            val_ds = FaceCSVDataset(args.val_csv, transform=val_tf)
            self.val_loader = DataLoader(
                val_ds,
                batch_size  = cfg["batch_size"],
                shuffle     = False,
                num_workers = cfg["num_workers"],
                pin_memory  = cfg["pin_memory"],
            )

        if self.is_main:
            print(f"   Train samples : {len(train_ds):,}")
            print(f"   Val samples   : {len(self.val_loader.dataset):,}" if self.val_loader else "")

    # ── Optimizer ─────────────────────────────────────────────────────────────

    def _setup_optimizer(self):
        """Configure SGD + cosine annealing scheduler."""
        cfg = self.config

        # Combine backbone + loss params
        all_params = list(self.backbone.parameters()) + \
                     list(self.loss_fn.parameters())

        self.optimizer = optim.SGD(
            all_params,
            lr           = cfg["lr"],
            momentum     = cfg["momentum"],
            weight_decay = cfg["weight_decay"],
        )

        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max  = cfg["epochs"],
            eta_min = cfg["lr"] * 1e-3,
        )

    # ── Warmup ────────────────────────────────────────────────────────────────

    def _warmup_lr(self, epoch: int):
        """Linear LR warmup for first N epochs."""
        warmup = self.config["warmup_epochs"]
        if epoch < warmup:
            warmup_lr = self.config["lr"] * (epoch + 1) / warmup
            for pg in self.optimizer.param_groups:
                pg["lr"] = warmup_lr

    # ── Train Epoch ───────────────────────────────────────────────────────────

    def _train_epoch(self, epoch: int) -> dict:
        """Run one training epoch."""
        self.backbone.train()
        self.loss_fn.train()

        total_loss  = 0.0
        n_batches   = 0
        n_correct   = 0
        n_total     = 0
        start       = time.perf_counter()

        for images, labels in self.train_loader:
            images = images.to(self.device, non_blocking=True)
            labels = labels.to(self.device, non_blocking=True)

            self.optimizer.zero_grad(set_to_none=True)

            # ✅ PyTorch 2.4+: torch.amp.autocast with device_type
            with torch.amp.autocast(
                device_type = self.device.type,
                enabled     = self.config["mixed_precision"],
            ):
                embeddings = self.backbone(images)
                loss       = self.loss_fn(embeddings, labels)

            self.scaler.scale(loss).backward()

            # Gradient clipping
            if self.config["gradient_clip"] > 0:
                self.scaler.unscale_(self.optimizer)
                nn.utils.clip_grad_norm_(
                    self.backbone.parameters(),
                    self.config["gradient_clip"],
                )

            self.scaler.step(self.optimizer)
            self.scaler.update()

            total_loss += loss.item()
            n_batches  += 1

        elapsed = time.perf_counter() - start
        throughput = len(self.train_loader.dataset) / elapsed

        return {
            "train_loss"  : round(total_loss / max(n_batches, 1), 4),
            "throughput"  : round(throughput, 1),
            "epoch_sec"   : round(elapsed, 1),
            "lr"          : self.optimizer.param_groups[0]["lr"],
        }

    # ── Val Epoch ─────────────────────────────────────────────────────────────

    @torch.no_grad()
    def _val_epoch(self) -> dict:
        """Run validation pass."""
        if self.val_loader is None:
            return {}

        self.backbone.eval()
        total_loss = 0.0
        n_batches  = 0

        for images, labels in self.val_loader:
            images = images.to(self.device, non_blocking=True)
            labels = labels.to(self.device, non_blocking=True)

            with torch.amp.autocast(
                device_type = self.device.type,
                enabled     = self.config["mixed_precision"],
            ):
                embeddings = self.backbone(images)
                loss       = self.loss_fn(embeddings, labels)

            total_loss += loss.item()
            n_batches  += 1

        return {
            "val_loss": round(total_loss / max(n_batches, 1), 4),
        }

    # ── Save Checkpoint ───────────────────────────────────────────────────────

    def _save_checkpoint(self, epoch: int, is_best: bool = False):
        """Save model checkpoint to run directory."""
        if not self.is_main:
            return

        ckpt_dir = self.run_dir / "checkpoints"
        ckpt_dir.mkdir(exist_ok=True)

        # Unwrap DDP if needed
        model_state = (
            self.backbone.module.state_dict()
            if self.distributed else
            self.backbone.state_dict()
        )

        state = {
            "epoch"               : epoch,
            "model_state_dict"    : model_state,
            "optimizer_state_dict": self.optimizer.state_dict(),
            "loss_fn_state_dict"  : self.loss_fn.state_dict(),
            "config"              : self.config,
        }

        name = "best_model.pt" if is_best else f"checkpoint_epoch_{epoch:03d}.pt"
        torch.save(state, ckpt_dir / name)

        if is_best:
            print(f"  ⭐ Best checkpoint saved → {ckpt_dir / name}")

    # ── Main Train Loop ───────────────────────────────────────────────────────

    def train(self):
        """Full training loop across all epochs."""
        cfg        = self.config
        best_loss  = float("inf")

        for epoch in range(cfg["epochs"]):

            # Warmup LR
            self._warmup_lr(epoch)

            # Train
            train_metrics = self._train_epoch(epoch)

            # Val
            val_metrics = self._val_epoch()

            # LR step (after warmup)
            if epoch >= cfg["warmup_epochs"]:
                self.scheduler.step()

            # Combine metrics
            metrics = {**train_metrics, **val_metrics}

            # Check best
            monitor   = val_metrics.get("val_loss", train_metrics["train_loss"])
            is_best   = monitor < best_loss
            if is_best:
                best_loss = monitor

            # Logging (main process only)
            if self.is_main:
                self.train_logger.log_epoch(epoch, metrics)

                if is_best:
                    self._save_checkpoint(epoch, is_best=True)

                if (epoch + 1) % cfg["save_every_epochs"] == 0:
                    self._save_checkpoint(epoch, is_best=False)

                # Sync run manager
                self.run_manager.log_epoch(
                    run_id  = self.run.run_id,
                    epoch   = epoch,
                    metrics = metrics,
                )

        # Finalize
        if self.is_main:
            final_metrics = {"best_train_loss": best_loss}
            self.run_manager.finalize_run(
                run_id        = self.run.run_id,
                final_metrics = final_metrics,
            )
            self.train_logger.log_event("training_complete", final_metrics)
            self.train_logger.close()

        if self.distributed:
            dist.destroy_process_group()


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description = "Baseline Face Recognition Training (ArcFace + ResNet50)"
    )

    # Data
    parser.add_argument("--train-csv",   default="data/splits/train/vggface2_train.csv")
    parser.add_argument("--val-csv",     default="data/splits/val/vggface2_val.csv")
    parser.add_argument("--output-dir",  default="experiments/runs/", type=Path)
    parser.add_argument("--run-name",    default=f"baseline_{uuid.uuid4().hex[:6]}")

    # Model
    parser.add_argument("--backbone",       default="resnet50")
    parser.add_argument("--embedding-dim",  type=int, default=512)
    parser.add_argument("--num-classes",    type=int, default=8631)

    # ArcFace
    parser.add_argument("--margin",  type=float, default=0.5)
    parser.add_argument("--scale",   type=float, default=64.0)

    # Training
    parser.add_argument("--epochs",      type=int,   default=100)
    parser.add_argument("--batch-size",  type=int,   default=512)
    parser.add_argument("--lr",          type=float, default=0.1)
    parser.add_argument("--weight-decay",type=float, default=5e-4)
    parser.add_argument("--warmup",      type=int,   default=5)
    parser.add_argument("--device",      default="cuda")
    parser.add_argument("--distributed", action="store_true")
    parser.add_argument("--no-amp",      action="store_true")
    parser.add_argument("--seed",        type=int, default=42)

    return parser.parse_args()


def main():
    args = parse_args()

    # Reproducibility
    torch.manual_seed(args.seed)

    # Build config from args
    config = {
        **BASELINE_CONFIG,
        "backbone"       : args.backbone,
        "embedding_dim"  : args.embedding_dim,
        "num_classes"    : args.num_classes,
        "arcface_margin" : args.margin,
        "arcface_scale"  : args.scale,
        "epochs"         : args.epochs,
        "batch_size"     : args.batch_size,
        "lr"             : args.lr,
        "weight_decay"   : args.weight_decay,
        "warmup_epochs"  : args.warmup,
        "mixed_precision": not args.no_amp,
    }

    trainer = BaselineTrainer(config=config, args=args)
    trainer.train()


if __name__ == "__main__":
    main()
