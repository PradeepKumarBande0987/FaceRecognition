"""
Face Recognition Backbone Module
Provides multiple neural network architectures for face recognition.

Available backbones:
    • ArcFace: ResNet-based backbone with ArcFace loss support
    • EfficientNet: Scalable efficient backbone
    • ViT-Face: Vision Transformer backbone for faces
    • GhostFaceNet: Lightweight ghost module backbone
    • FaceNet: Original FaceNet triplet loss backbone
    • MobileFaceNet: Ultra-lightweight mobile backbone
"""

from .arcface.arcface import (
    BasicBlock,
    Bottleneck,
    ResNet,
    ResNetArcFace,
    get_resnet,
)

from .efficientnet.efficientnet import (
    EfficientNet,
    EfficientNetB0,
    EfficientNetB4,
    get_efficientnet,
)

from .efficientnet.vit_face import (
    PatchEmbedding,
    TransformerBlock,
    ViTFace,
    get_vit_face,
)

from .ghostfacenet.ghostfacenet import (
    GhostModule,
    GhostBottleneck,
    GhostFaceNet,
    get_ghostfacenet,
)

from .mobilefacenet.facenet import (
    InceptionResNetV1,
    FaceNet,
)

from .mobilefacenet.mobilefacenet import (
    DepthwiseSeparableConv,
    MobileFaceNet,
    get_mobilefacenet,
)

__all__ = [
    # ArcFace
    "BasicBlock",
    "Bottleneck",
    "ResNet",
    "ResNetArcFace",
    "get_resnet",

    # EfficientNet
    "EfficientNet",
    "EfficientNetB0",
    "EfficientNetB4",
    "get_efficientnet",

    # ViT-Face
    "PatchEmbedding",
    "TransformerBlock",
    "ViTFace",
    "get_vit_face",

    # GhostFaceNet
    "GhostModule",
    "GhostBottleneck",
    "GhostFaceNet",
    "get_ghostfacenet",

    # FaceNet
    "InceptionResNetV1",
    "FaceNet",

    # MobileFaceNet
    "DepthwiseSeparableConv",
    "MobileFaceNet",
    "get_mobilefacenet",
]
