"""
Anti-Spoofing Module
Detects presentation attacks: print, replay, 3D mask attacks.
Architecture: Binary classifier on depth + texture features.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional


class DepthMapEstimator(nn.Module):
    """
    Lightweight depth estimation head.
    Estimates facial depth map from RGB input.
    Real faces have smooth depth variation;
    fake faces (prints/screens) are approximately flat.
    """

    def __init__(self, in_ch: int = 128):
        super().__init__()
        self.decode = nn.Sequential(
            nn.ConvTranspose2d(in_ch, 64, 4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(64,  32, 4, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(32,  1,  4, stride=2, padding=1),
            nn.Sigmoid(),    # depth ∈ [0, 1]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.decode(x)


class TextureAnalyzer(nn.Module):
    """
    Texture analysis branch.
    Captures high-frequency artifacts from printed/replayed attacks.
    Uses multi-scale LBP-inspired convolutions to detect unnatural
    texture patterns (moiré patterns, pixelation, color distortion).
    """

    def __init__(self, in_ch: int = 3):
        super().__init__()

        # Multi-scale texture feature extraction
        # Scale 1: Fine-grained details (moiré, noise)
        self.scale1 = nn.Sequential(
            nn.Conv2d(in_ch, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),      # H/2 x W/2
        )

        # Scale 2: Mid-level patterns (screen-door, pixel grid)
        self.scale2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=2, dilation=2),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=2, dilation=2),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),      # H/4 x W/4
        )

        # Scale 3: Coarse patterns (color cast, reflections)
        self.scale3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=4, dilation=4),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=4, dilation=4),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),      # H/8 x W/8
        )

        # High-frequency emphasis: Laplacian-like edge filter
        # Detects sharp compression artifacts / print grain
        self.hf_filter = nn.Sequential(
            nn.Conv2d(in_ch, 16, kernel_size=5, padding=2),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(8),      # aggressive pool → global texture summary
        )

        # Fuse multi-scale features
        self.fuse = nn.Sequential(
            nn.AdaptiveAvgPool2d(4),   # unify spatial dims → 4x4
            nn.Flatten(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: RGB face crop  [B, 3, H, W]  (H=W=64 recommended)
        Returns:
            texture_feat: [B, (128+16)*4*4]  flattened multi-scale features
        """
        s1 = self.scale1(x)        # [B, 32,  H/2, W/2]
        s2 = self.scale2(s1)       # [B, 64,  H/4, W/4]
        s3 = self.scale3(s2)       # [B, 128, H/8, W/8]
        hf = self.hf_filter(x)     # [B, 16,  H/8, W/8]

        # Fuse coarse + high-frequency streams
        s3_pooled = self.fuse(s3)  # [B, 128*16]
        hf_pooled = self.fuse(hf)  # [B,  16*16]

        return torch.cat([s3_pooled, hf_pooled], dim=1)  # [B, 2304]


class AntiSpoofingClassifier(nn.Module):
    """
    Final binary classifier head.
    Fuses depth cues + texture features → real / spoof logit.
    """

    def __init__(self, texture_dim: int = 2304, depth_feat_dim: int = 256):
        super().__init__()

        # Compact depth feature projection
        self.depth_proj = nn.Sequential(
            nn.Flatten(),
            nn.Linear(depth_feat_dim, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
        )

        # Compact texture feature projection
        self.texture_proj = nn.Sequential(
            nn.Linear(texture_dim, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
        )

        # Joint classifier
        self.classifier = nn.Sequential(
            nn.Linear(128 + 128, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.4),
            nn.Linear(128, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 1),     # single logit → BCEWithLogitsLoss
        )

    def forward(
        self,
        depth_feat: torch.Tensor,
        texture_feat: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            depth_feat  : flattened depth features  [B, depth_feat_dim]
            texture_feat: flattened texture features [B, texture_dim]
        Returns:
            logit: [B, 1]  (positive = real face)
        """
        d = self.depth_proj(depth_feat)      # [B, 128]
        t = self.texture_proj(texture_feat)  # [B, 128]
        fused = torch.cat([d, t], dim=1)     # [B, 256]
        return self.classifier(fused)        # [B, 1]


class AntiSpoofingModule(nn.Module):
    """
    Full Anti-Spoofing pipeline.

    Flow:
        RGB face crop
            │
            ├─── BackboneEncoder ──► feature map [B, 128, H', W']
            │         │
            │    DepthMapEstimator ──► depth map [B, 1, H_d, W_d]
            │
            ├─── TextureAnalyzer  ──► texture feat [B, texture_dim]
            │
            └─── AntiSpoofingClassifier ──► logit [B, 1]
    """

    def __init__(
        self,
        backbone_out_ch: int = 128,
        input_size: int = 64,
    ):
        """
        Args:
            backbone_out_ch: channels output by your shared face backbone.
            input_size     : spatial size of the face crop (square assumed).
        """
        super().__init__()

        # ── Lightweight backbone (used when no shared backbone exists) ──────
        self.backbone = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(inplace=True),
            nn.MaxPool2d(2),                                        # → H/2
            nn.Conv2d(64, backbone_out_ch, 3, padding=1),
            nn.BatchNorm2d(backbone_out_ch), nn.ReLU(inplace=True),
            nn.MaxPool2d(2),                                        # → H/4
        )

        self.depth_estimator  = DepthMapEstimator(in_ch=backbone_out_ch)
        self.texture_analyzer = TextureAnalyzer(in_ch=3)

        # Compute depth feature dim dynamically
        dummy      = torch.zeros(1, 3, input_size, input_size)
        bb_out     = self.backbone(dummy)                 # [1, 128, H/4, W/4]
        depth_out  = self.depth_estimator(bb_out)         # [1,   1, H/2, W/2]  (3× upsample from H/4)
        depth_flat = depth_out.view(1, -1).shape[1]

        tex_out    = self.texture_analyzer(dummy)
        tex_dim    = tex_out.shape[1]

        self.classifier = AntiSpoofingClassifier(
            texture_dim=tex_dim,
            depth_feat_dim=depth_flat,
        )

        self.input_size = input_size

    def forward(
        self,
        x: torch.Tensor,
        return_maps: bool = False,
    ) -> dict:
        """
        Args:
            x          : face crop tensor  [B, 3, H, W]
            return_maps: if True, also return depth map for visualization/aux loss

        Returns:
            dict with keys:
                'logit'     : [B, 1]   raw score  (>0 → real)
                'prob'      : [B, 1]   sigmoid probability of being real
                'is_real'   : [B]      bool predictions
                'depth_map' : [B,1,H',W'] (only when return_maps=True)
        """
        # ── Backbone features ────────────────────────────────────────────────
        feat = self.backbone(x)                  # [B, 128, H/4, W/4]

        # ── Depth branch ─────────────────────────────────────────────────────
        depth_map  = self.depth_estimator(feat)  # [B, 1, H/2, W/2]
        depth_flat = depth_map.view(x.size(0), -1)   # [B, D]

        # ── Texture branch ───────────────────────────────────────────────────
        tex_feat   = self.texture_analyzer(x)    # [B, texture_dim]

        # ── Classification ───────────────────────────────────────────────────
        logit = self.classifier(depth_flat, tex_feat)   # [B, 1]
        prob  = torch.sigmoid(logit)                     # [B, 1]

        out = {
            "logit"  : logit,
            "prob"   : prob,
            "is_real": (prob.squeeze(1) > 0.5),
        }
        if return_maps:
            out["depth_map"] = depth_map

        return out


# ── Loss ─────────────────────────────────────────────────────────────────────

class AntiSpoofingLoss(nn.Module):
    """
    Combined loss:
        L = α · BCE(logit, label) + β · DepthConsistency(depth_map, label)

    DepthConsistency:
        - Real faces  → encourage high depth variance  (non-flat)
        - Spoof faces → penalize high depth variance   (should be flat)
    """

    def __init__(self, alpha: float = 1.0, beta: float = 0.5):
        super().__init__()
        self.alpha = alpha
        self.beta  = beta
        self.bce   = nn.BCEWithLogitsLoss()

    def forward(
        self,
        logit    : torch.Tensor,   # [B, 1]
        depth_map: torch.Tensor,   # [B, 1, H, W]
        labels   : torch.Tensor,   # [B]  1=real, 0=spoof
    ) -> Tuple[torch.Tensor, dict]:

        # ── BCE classification loss ──────────────────────────────────────────
        bce_loss = self.bce(logit.squeeze(1), labels.float())

        # ── Depth consistency loss ───────────────────────────────────────────
        # Compute per-sample spatial variance of depth map
        B = depth_map.size(0)
        d_flat    = depth_map.view(B, -1)                        # [B, H*W]
        d_var     = d_flat.var(dim=1)                            # [B]

        # Real  (label=1): want HIGH variance → penalize low variance
        # Spoof (label=0): want LOW  variance → penalize high variance
        # Unified: loss = label*(1-var) + (1-label)*var
        depth_loss = (labels * (1.0 - d_var) + (1.0 - labels) * d_var).mean()

        total = self.alpha * bce_loss + self.beta * depth_loss

        return total, {
            "bce_loss"  : bce_loss.item(),
            "depth_loss": depth_loss.item(),
            "total_loss": total.item(),
        }


# ── Utility: integrate into face recognition pipeline ────────────────────────

class AntiSpoofingGate(nn.Module):
    """
    Drop-in gate for your face recognition pipeline.
    Wraps AntiSpoofingModule and blocks spoof faces before recognition.

    Usage:
        gate   = AntiSpoofingGate(weights_path="anti_spoof.pt")
        result = gate(face_crop)
        if result["passed"]:
            embedding = face_recognizer(face_crop)
    """

    def __init__(
        self,
        weights_path: Optional[str] = None,
        threshold   : float = 0.6,
        device      : str   = "cpu",
    ):
        super().__init__()
        self.threshold = threshold
        self.device    = torch.device(device)
        self.model     = AntiSpoofingModule().to(self.device)
        self.model.eval()

        if weights_path is not None:
            state = torch.load(weights_path, map_location=self.device)
            self.model.load_state_dict(state)
            print(f"[AntiSpoofingGate] Loaded weights from: {weights_path}")
        else:
            print("[AntiSpoofingGate] ⚠️  No weights loaded — running with random init.")

    @torch.no_grad()
    def forward(self, face_crop: torch.Tensor) -> dict:
        """
        Args:
            face_crop: [B, 3, H, W] normalized face tensor

        Returns:
            dict:
                'passed'  : bool tensor [B] — True if real face
                'prob'    : float tensor [B] — liveness probability
                'blocked' : indices of blocked (spoof) samples
        """
        face_crop = face_crop.to(self.device)
        out       = self.model(face_crop, return_maps=False)

        prob    = out["prob"].squeeze(1)           # [B]
        passed  = prob >= self.threshold           # [B] bool
        blocked = (~passed).nonzero(as_tuple=False).squeeze(1)

        return {
            "passed" : passed,
            "prob"   : prob,
            "blocked": blocked,
        }
