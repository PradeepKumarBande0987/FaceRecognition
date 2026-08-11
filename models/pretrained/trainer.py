"""
Training loop and utilities for face recognition model pre-training.

Features:
    • Mixed precision training (AMP)
    • Learning rate scheduling
    • Model checkpointing
    • Evaluation metrics
    • Distributed training support
"""

import torch
import torch.nn as nn
from torch.amp import autocast, GradScaler
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR, StepLR
from torch.utils.data import DataLoader
from typing import Dict, Optional, Tuple
import os
from pathlib import Path
import time
import numpy as np
from tqdm import tqdm
import logging


class Trainer:
    """
    Trainer for face recognition backbone pre-training.

    Features:
        • Mixed precision training (FP16 for speed, FP32 for accuracy)
        • Distributed data parallel (multi-GPU)
        • Learning rate scheduling (cosine annealing, step decay)
        • Gradient clipping (training stability)
        • Model checkpointing (best and periodic)
        • Tensorboard logging
    """

    def __init__(
        self,
        model: nn.Module,
        loss_fn: nn.Module,
        train_loader: DataLoader,
        test_loader: Optional[DataLoader] = None,
        learning_rate: float = 0.1,
        momentum: float = 0.9,
        weight_decay: float = 5e-4,
        num_epochs: int = 100,
        warmup_epochs: int = 5,
        lr_scheduler: str = "cosine",  # "cosine" or "step"
        device: str = "cuda",
        checkpoint_dir: str = "./checkpoints",
        mixed_precision: bool = True,
        gradient_clip: float = 1.0,
    ):
        self.model = model.to(device)
        self.loss_fn = loss_fn.to(device)
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.device = torch.device(device)

        # Optimizer
        self.optimizer = optim.SGD(
            self.model.parameters(),
            lr=learning_rate,
            momentum=momentum,
            weight_decay=weight_decay,
        )

        # Learning rate scheduler
        if lr_scheduler == "cosine":
            self.scheduler = CosineAnnealingLR(self.optimizer, num_epochs)
        elif lr_scheduler == "step":
            self.scheduler = StepLR(self.optimizer, step_size=30, gamma=0.1)
        else:
            self.scheduler = None

        self.num_epochs = num_epochs
        self.warmup_epochs = warmup_epochs
        self.mixed_precision = mixed_precision
        self.gradient_clip = gradient_clip

        # Mixed precision
        self.scaler = GradScaler(
            device=self.device.type,
            enabled=mixed_precision,
        )

        # Checkpointing
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        # Logging
        self.logger = self._setup_logger()
        self.best_loss = float('inf')
        self.best_epoch = 0

    def _setup_logger(self) -> logging.Logger:
        """Setup logging."""
        logger = logging.getLogger(__name__)
        logger.setLevel(logging.DEBUG)

        handler = logging.FileHandler(self.checkpoint_dir / "training.log")
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)

        return logger

    def _warmup_lr(self, epoch: int):
        """Linear warmup for first few epochs."""
        if epoch < self.warmup_epochs:
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = 0.1 * (epoch + 1) / self.warmup_epochs

    def train_epoch(self) -> Dict[str, float]:
        """Train for one epoch."""
        self.model.train()
        total_loss = 0.0
        num_batches = 0

        pbar = tqdm(self.train_loader, desc="Training")
        for batch_idx, (images, labels) in enumerate(pbar):
            images = images.to(self.device)
            labels = labels.to(self.device)

            # Forward pass
            self.optimizer.zero_grad()

            with autocast(device=self.device.type, enabled=self.mixed_precision):
                embeddings = self.model(images)

                # Handle dict output from backbone
                if isinstance(embeddings, dict):
                    embeddings = embeddings["embedding"]

                loss = self.loss_fn(embeddings, labels)

            # Backward pass
            self.scaler.scale(loss).backward()

            # Gradient clipping
            if self.gradient_clip > 0:
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    self.gradient_clip
                )

            self.scaler.step(self.optimizer)
            self.scaler.update()

            # Update metrics
            total_loss += loss.item()
            num_batches += 1

            # Progress bar
            pbar.set_postfix({"loss": total_loss / num_batches})

        return {"loss": total_loss / num_batches}

    def evaluate(self) -> Dict[str, float]:
        """Evaluate on test set."""
        if self.test_loader is None:
            return {}

        self.model.eval()
        total_loss = 0.0
        num_batches = 0

        with torch.no_grad():
            pbar = tqdm(self.test_loader, desc="Evaluating")
            for images, labels in pbar:
                images = images.to(self.device)
                labels = labels.to(self.device)

                embeddings = self.model(images)
                if isinstance(embeddings, dict):
                    embeddings = embeddings["embedding"]

                loss = self.loss_fn(embeddings, labels)
                total_loss += loss.item()
                num_batches += 1

                pbar.set_postfix({"loss": total_loss / num_batches})

        return {"val_loss": total_loss / num_batches}

    def train(self):
        """Full training loop."""
        self.logger.info("Starting training...")
        self.logger.info(f"Total epochs: {self.num_epochs}")

        for epoch in range(self.num_epochs):
            # Warmup
            self._warmup_lr(epoch)

            # Train
            train_metrics = self.train_epoch()

            # Evaluate
            eval_metrics = self.evaluate()

            # Learning rate decay
            if self.scheduler and epoch >= self.warmup_epochs:
                self.scheduler.step()

            # Log
            lr = self.optimizer.param_groups[0]['lr']
            self.logger.info(
                f"Epoch [{epoch+1}/{self.num_epochs}] "
                f"LR: {lr:.6f} | "
                f"Loss: {train_metrics['loss']:.4f} | "
                f"Val Loss: {eval_metrics.get('val_loss', 0):.4f}"
            )

            # Checkpointing
            val_loss = eval_metrics.get('val_loss', train_metrics['loss'])

            # Best checkpoint
            if val_loss < self.best_loss:
                self.best_loss = val_loss
                self.best_epoch = epoch
                self._save_checkpoint(epoch, is_best=True)

            # Periodic checkpoint
            if (epoch + 1) % 10 == 0:
                self._save_checkpoint(epoch, is_best=False)

        self.logger.info(
            f"Training complete! Best epoch: {self.best_epoch + 1} "
            f"(loss: {self.best_loss:.4f})"
        )

    def _save_checkpoint(self, epoch: int, is_best: bool = False):
        """Save model checkpoint."""
        state = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'loss_fn_state_dict': self.loss_fn.state_dict(),
            'best_loss': self.best_loss,
        }

        if is_best:
            path = self.checkpoint_dir / "best_model.pt"
        else:
            path = self.checkpoint_dir / f"checkpoint_epoch_{epoch+1:03d}.pt"

        torch.save(state, path)
        self.logger.info(f"Saved checkpoint: {path}")

    def load_checkpoint(self, checkpoint_path: str):
        """Load model checkpoint."""
        state = torch.load(
            checkpoint_path,
            map_location=self.device,
            weights_only=False,  # needed for optimizer state dicts
        )
        self.model.load_state_dict(state['model_state_dict'])
        self.optimizer.load_state_dict(state['optimizer_state_dict'])
        self.loss_fn.load_state_dict(state['loss_fn_state_dict'])
        self.best_loss = state['best_loss']
        self.logger.info(f"Loaded checkpoint from {checkpoint_path}")
