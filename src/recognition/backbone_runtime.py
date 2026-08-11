from __future__ import annotations

from pathlib import Path
from typing import Any, Optional, Tuple

import torch
import torch.nn as nn
import torchvision.models as tvm

try:
    import torchvision.transforms.v2 as T
    HAS_TV2 = True
except ImportError:
    import torchvision.transforms as T
    HAS_TV2 = False


def build_resnet50_backbone(
    embedding_dim: int = 512,
    pretrained: bool = True,
) -> nn.Module:
    """Build ResNet-50 with ArcFace-style embedding head."""
    weights = tvm.ResNet50_Weights.IMAGENET1K_V2 if pretrained else None
    backbone = tvm.resnet50(weights=weights)
    in_feat = backbone.fc.in_features

    backbone.fc = nn.Sequential(
        nn.BatchNorm1d(in_feat),
        nn.Linear(in_feat, embedding_dim, bias=False),
        nn.BatchNorm1d(embedding_dim),
    )
    return backbone


def load_backbone(
    checkpoint: Optional[str] = None,
    embedding_dim: int = 512,
    device: torch.device = torch.device("cpu"),
    pretrained: bool = True,
) -> nn.Module:
    """Build backbone and optionally load checkpoint weights."""
    backbone = build_resnet50_backbone(embedding_dim, pretrained)

    if checkpoint and Path(checkpoint).exists():
        ckpt = torch.load(checkpoint, map_location=device, weights_only=False)
        sd = ckpt.get("model_state_dict", ckpt)
        sd = {k.replace("module.", ""): v for k, v in sd.items()}
        backbone.load_state_dict(sd, strict=False)

    return backbone.eval().to(device)


def get_inference_transform(image_size: Tuple[int, int] = (112, 112)) -> Any:
    """Build deterministic inference transforms for face crops."""
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]

    if HAS_TV2:
        return T.Compose([
            T.Resize(image_size, antialias=True),
            T.ToImage(),
            T.ToDtype(torch.float32, scale=True),
            T.Normalize(mean=mean, std=std),
        ])

    return T.Compose([
        T.Resize(image_size),
        T.ToTensor(),
        T.Normalize(mean=mean, std=std),
    ])
