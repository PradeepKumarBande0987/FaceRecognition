"""
Data loading and preprocessing for face recognition pre-training.

Supports:
    • Multiple face datasets (CASIA-WebFace, VGGFace2, MS-Celeb-1M)
    • Data augmentation (random crops, flips, color jittering)
    • Balanced sampling (class-wise balanced mini-batches)
    • Distributed data loading
"""

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, Sampler
import torchvision.transforms.v2 as transforms
from torchvision.transforms import InterpolationMode
from pathlib import Path
from typing import Tuple, List, Optional, Dict
import os
import numpy as np
from PIL import Image
import pickle


class FaceDataset(Dataset):
    """
    Face recognition dataset.

    Directory structure:
        data/
        ├── id_001/
        │   ├── 001.jpg
        │   ├── 002.jpg
        │   └── ...
        ├── id_002/
        │   ├── 001.jpg
        │   └── ...
        └── ...
    """

    def __init__(
        self,
        root_dir: str,
        transform: Optional[transforms.Compose] = None,
        image_size: Tuple[int, int] = (112, 112),
    ):
        super().__init__()
        self.root_dir = Path(root_dir)
        self.transform = transform
        self.image_size = image_size

        # Build dataset
        self.samples = []  # (image_path, label)
        self.class_to_idx = {}
        self.idx_to_class = {}

        # Scan directories
        for label, class_dir in enumerate(sorted(self.root_dir.iterdir())):
            if not class_dir.is_dir():
                continue

            self.class_to_idx[class_dir.name] = label
            self.idx_to_class[label] = class_dir.name

            # Add images
            for img_path in sorted(class_dir.glob("*.jpg")) + sorted(
                class_dir.glob("*.png")
            ):
                self.samples.append((str(img_path), label))

        print(f"[FaceDataset] Loaded {len(self.samples)} images from "
              f"{len(self.class_to_idx)} classes")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        img_path, label = self.samples[idx]

        # Load image
        img = Image.open(img_path).convert("RGB")

        # Resize
        img = img.resize(self.image_size, Image.Resampling.LANCZOS)

        # Transform
        if self.transform:
            img = self.transform(img)

        return img, label


class BalancedBatchSampler(Sampler):
    """
    Balanced batch sampler: ensures each batch has samples from different classes.

    Prevents mini-batches from being dominated by single class,
    leading to better triplet loss convergence.

    Batch composition: num_classes × samples_per_class
    Example: 8 classes × 4 samples/class = 32 batch size
    """

    def __init__(
        self,
        dataset: FaceDataset,
        batch_size: int = 32,
        num_classes: int = 8,
        drop_last: bool = True,
    ):
        super().__init__()
        self.dataset = dataset
        self.batch_size = batch_size
        self.num_classes = num_classes
        self.samples_per_class = batch_size // num_classes
        self.drop_last = drop_last

        # Group indices by class
        self.class_indices = {}
        for idx, (_, label) in enumerate(dataset.samples):
            if label not in self.class_indices:
                self.class_indices[label] = []
            self.class_indices[label].append(idx)

        # Shuffle within class
        for label in self.class_indices:
            np.random.shuffle(self.class_indices[label])

    def __iter__(self):
        # Available classes
        available_classes = list(self.class_indices.keys())

        batch = []
        while available_classes:
            # Select random classes
            selected_classes = np.random.choice(
                available_classes,
                min(self.num_classes, len(available_classes)),
                replace=False,
            )

            # Add samples from selected classes
            for class_label in selected_classes:
                indices = self.class_indices[class_label]
                if len(indices) >= self.samples_per_class:
                    # Sample without replacement
                    samples = np.random.choice(
                        indices,
                        self.samples_per_class,
                        replace=False,
                    )
                    batch.extend(samples)

            # Yield batch if full
            if len(batch) >= self.batch_size:
                yield batch[:self.batch_size]
                batch = batch[self.batch_size:]

                # Remove exhausted classes
                available_classes = [
                    c for c in available_classes
                    if len(self.class_indices[c]) > 0
                ]

        # Handle remainder
        if batch and not self.drop_last:
            yield batch

    def __len__(self) -> int:
        total_samples = len(self.dataset)
        if self.drop_last:
            return total_samples // self.batch_size
        else:
            return (total_samples + self.batch_size - 1) // self.batch_size


def get_train_transforms(
    image_size: Tuple[int, int] = (112, 112),
) -> transforms.Compose:
    """Get training data augmentation pipeline."""
    return transforms.Compose([
        # Random crop
        transforms.RandomResizedCrop(
            image_size[0],
            scale=(0.8, 1.0),
            ratio=(0.95, 1.05),
            antialias=True,  
        ),
        # Random flip
        transforms.RandomHorizontalFlip(p=0.5),
        # Color jittering
        transforms.ColorJitter(
            brightness=0.2,
            contrast=0.2,
            saturation=0.2,
            hue=0.1,
        ),
        # Random rotation
        transforms.RandomRotation(degrees=10),
        # To tensor
        transforms.ToDtype(torch.float32, scale=True), 
        # transforms.ToTensor(),
        # Normalize (ImageNet statistics)
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ])


def get_test_transforms(
    image_size: Tuple[int, int] = (112, 112),
) -> transforms.Compose:
    """Get test data preprocessing (no augmentation)."""
    return transforms.Compose([
        transforms.Resize(image_size, antialias=True),
        transforms.ToDtype(torch.float32, scale=True),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ])


def get_dataloaders(
    train_dir: str,
    test_dir: Optional[str] = None,
    batch_size: int = 128,
    num_workers: int = 4,
    balanced_sampling: bool = True,
    image_size: Tuple[int, int] = (112, 112),
) -> Tuple[DataLoader, Optional[DataLoader]]:
    """
    Create train and test dataloaders.

    Args:
        train_dir: path to training data
        test_dir : path to test data
        batch_size: batch size
        num_workers: number of data loading workers
        balanced_sampling: use balanced batch sampler
        image_size: image size (H, W)

    Returns:
        train_loader, test_loader
    """
    # Train loader
    train_transform = get_train_transforms(image_size)
    train_dataset = FaceDataset(train_dir, transform=train_transform,
                               image_size=image_size)

    if balanced_sampling:
        sampler = BalancedBatchSampler(
            train_dataset,
            batch_size=batch_size,
            num_classes=8,  # 8 classes per batch
            drop_last=True,
        )
        train_loader = DataLoader(
            train_dataset,
            batch_sampler=sampler,
            num_workers=num_workers,
            pin_memory=True,
        )
    else:
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=True,
        )

    # Test loader
    test_loader = None
    if test_dir is not None:
        test_transform = get_test_transforms(image_size)
        test_dataset = FaceDataset(test_dir, transform=test_transform,
                                  image_size=image_size)
        test_loader = DataLoader(
            test_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True,
        )

    return train_loader, test_loader
