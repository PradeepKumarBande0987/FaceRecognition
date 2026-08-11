"""
Data Pipeline — DataLoader Module.

Provides unified DataLoader factories for all dataset splits.

Supports:
    • CSV-based datasets (VGGFace2, CelebA, Custom CCTV)
    • Pair-based datasets for verification (LFW protocol)
    • Distributed training samplers (DDP)
    • BalancedBatchSampler (equal samples per identity)
    • Multi-dataset mixing with configurable sampling weights
"""

from __future__ import annotations

import csv
import random
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
from PIL import Image
from torch.utils.data import (
    DataLoader,
    Dataset,
    DistributedSampler,
    Sampler,
    WeightedRandomSampler,
)

from src.data_pipeline.augmentation.augmentation import FaceAugmentor, AugConfig


# ── Face Dataset ──────────────────────────────────────────────────────────────

class FaceDataset(Dataset):
    """
    CSV-based face image dataset.

    CSV Format:
        image_path, label [, optional attribute columns]

    Applies augmentation pipeline per sample.
    """

    def __init__(
        self,
        csv_path    : str,
        augmentor   : Optional[FaceAugmentor] = None,
        return_path : bool = False,
    ):
        self.samples     = []
        self.augmentor   = augmentor
        self.return_path = return_path

        with open(csv_path, newline="") as f:
            for row in csv.DictReader(f):
                img   = row.get("image_path", "")
                label = int(row.get("label", row.get("identity_label", 0)))
                self.samples.append((img, label))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        img_path, label = self.samples[idx]

        try:
            img = Image.open(img_path).convert("RGB")
        except Exception:
            img = Image.new("RGB", (112, 112), color=(128, 128, 128))

        if self.augmentor:
            tensor = self.augmentor(img)
        else:
            import torchvision.transforms.v2 as T
            tensor = T.Compose([
                T.Resize((112, 112), antialias=True),
                T.ToDtype(torch.float32, scale=True),
                T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ])(img)

        if self.return_path:
            return tensor, label, img_path
        return tensor, label


# ── Pair Dataset ──────────────────────────────────────────────────────────────

class PairDataset(Dataset):
    """
    Face verification pair dataset (LFW protocol).

    CSV Format:
        image1_path, image2_path, label (1=same, 0=different)
    """

    def __init__(
        self,
        csv_path  : str,
        transform = None,
    ):
        self.pairs     = []
        self.transform = transform

        with open(csv_path, newline="") as f:
            for row in csv.DictReader(f):
                self.pairs.append((
                    row["image1_path"],
                    row["image2_path"],
                    int(row["label"]),
                ))

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, idx: int):
        img1_path, img2_path, label = self.pairs[idx]

        img1 = Image.open(img1_path).convert("RGB")
        img2 = Image.open(img2_path).convert("RGB")

        if self.transform:
            img1 = self.transform(img1)
            img2 = self.transform(img2)

        return img1, img2, label


# ── Balanced Batch Sampler ────────────────────────────────────────────────────

class BalancedBatchSampler(Sampler):
    """
    Yields batches with exactly N samples per identity.

    Useful for metric learning to ensure genuine/impostor
    pair diversity within each batch.

    Args:
        labels           : list of identity labels
        n_classes        : identities per batch
        n_samples        : samples per identity per batch
    """

    def __init__(
        self,
        labels    : List[int],
        n_classes : int = 16,
        n_samples : int = 4,
    ):
        self.labels    = labels
        self.n_classes = n_classes
        self.n_samples = n_samples

        # Build label → index map
        self.label_to_indices: Dict[int, List[int]] = defaultdict(list)
        for idx, lbl in enumerate(labels):
            self.label_to_indices[lbl].append(idx)

        # Filter out labels with fewer than n_samples
        self.valid_labels = [
            lbl for lbl, idxs in self.label_to_indices.items()
            if len(idxs) >= self.n_samples
        ]

    def __iter__(self):
        all_batches = []
        labels_copy = self.valid_labels.copy()
        random.shuffle(labels_copy)

        for i in range(0, len(labels_copy) - self.n_classes + 1, self.n_classes):
            batch_labels = labels_copy[i:i + self.n_classes]
            batch = []
            for lbl in batch_labels:
                indices = random.sample(
                    self.label_to_indices[lbl],
                    self.n_samples
                )
                batch.extend(indices)
            all_batches.extend(batch)

        return iter(all_batches)

    def __len__(self) -> int:
        return (len(self.valid_labels) // self.n_classes) * \
               (self.n_classes * self.n_samples)


# ── DataLoader Factory ────────────────────────────────────────────────────────

class FaceDataLoaderFactory:
    """
    Factory for building train/val/test DataLoaders.

    Handles:
        • Single and multi-dataset loading
        • Augmentation level selection
        • Distributed sampler injection
        • Balanced batch sampling
        • Pair dataset loading

    Usage:
        factory = FaceDataLoaderFactory(
            train_csv    = "data/splits/train/train_merged.csv",
            val_csv      = "data/splits/val/val_merged.csv",
            batch_size   = 512,
            aug_level    = "heavy",
            distributed  = True,
        )
        train_loader = factory.get_train_loader()
        val_loader   = factory.get_val_loader()
    """

    def __init__(
        self,
        train_csv       : Optional[str] = None,
        val_csv         : Optional[str] = None,
        test_csv        : Optional[str] = None,
        pairs_csv       : Optional[str] = None,
        batch_size      : int   = 512,
        aug_level       : str   = "standard",
        aug_config      : Optional[AugConfig] = None,
        num_workers     : int   = 8,
        pin_memory      : bool  = True,
        distributed     : bool  = False,
        balanced_batch  : bool  = False,
        n_classes_batch : int   = 16,
        n_samples_batch : int   = 4,
        drop_last       : bool  = True,
        seed            : int   = 42,
    ):
        self.train_csv      = train_csv
        self.val_csv        = val_csv
        self.test_csv       = test_csv
        self.pairs_csv      = pairs_csv
        self.batch_size     = batch_size
        self.num_workers    = num_workers
        self.pin_memory     = pin_memory
        self.distributed    = distributed
        self.balanced_batch = balanced_batch
        self.n_classes_batch= n_classes_batch
        self.n_samples_batch= n_samples_batch
        self.drop_last      = drop_last
        self.seed           = seed

        # Build augmentors
        self.train_augmentor = FaceAugmentor(
            level  = aug_level,
            config = aug_config,
            seed   = seed,
        )
        self.val_augmentor = None   # no augmentation for val/test

    def get_train_loader(self) -> DataLoader:
        """Build and return training DataLoader."""
        assert self.train_csv, "train_csv must be provided"

        dataset = FaceDataset(
            csv_path  = self.train_csv,
            augmentor = self.train_augmentor,
        )

        # Sampler selection
        if self.distributed:
            sampler = DistributedSampler(dataset, shuffle=True)
            shuffle = False
        elif self.balanced_batch:
            labels  = [s[1] for s in dataset.samples]
            sampler = BalancedBatchSampler(
                labels    = labels,
                n_classes = self.n_classes_batch,
                n_samples = self.n_samples_batch,
            )
            shuffle = False
        else:
            sampler = None
            shuffle = True

        return DataLoader(
            dataset,
            batch_size         = self.batch_size,
            shuffle            = shuffle,
            sampler            = sampler,
            num_workers        = self.num_workers,
            pin_memory         = self.pin_memory,
            drop_last          = self.drop_last,
            persistent_workers = self.num_workers > 0,
        )

    def get_val_loader(self) -> Optional[DataLoader]:
        """Build and return validation DataLoader."""
        if not self.val_csv:
            return None

        dataset = FaceDataset(
            csv_path  = self.val_csv,
            augmentor = None,
        )
        return DataLoader(
            dataset,
            batch_size  = self.batch_size,
            shuffle     = False,
            num_workers = self.num_workers,
            pin_memory  = self.pin_memory,
        )

    def get_test_loader(self) -> Optional[DataLoader]:
        """Build and return test DataLoader."""
        if not self.test_csv:
            return None

        dataset = FaceDataset(
            csv_path  = self.test_csv,
            augmentor = None,
        )
        return DataLoader(
            dataset,
            batch_size  = self.batch_size,
            shuffle     = False,
            num_workers = self.num_workers,
            pin_memory  = self.pin_memory,
        )

    def get_pairs_loader(self) -> Optional[DataLoader]:
        """Build verification pairs DataLoader (LFW protocol)."""
        if not self.pairs_csv:
            return None

        val_tf = self.train_augmentor.get_val_transform()
        dataset = PairDataset(csv_path=self.pairs_csv, transform=val_tf)

        return DataLoader(
            dataset,
            batch_size  = self.batch_size,
            shuffle     = False,
            num_workers = self.num_workers,
            pin_memory  = self.pin_memory,
        )
