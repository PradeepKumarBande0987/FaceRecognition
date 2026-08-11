"""
Super Resolution Module
Enhances low-resolution face images before recognition pipeline.
Architecture: RRDB (Residual-in-Residual Dense Block) based ESRGAN-inspired
              network with attention mechanisms + perceptual loss.
Supports: 2x, 4x upscaling factors.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Dict, List
import math

# ── Building Blocks ───────────────────────────────────────────────────────────

class ConvBNReLU(nn.Module):
    """Standard Conv → BatchNorm → ReLU block."""

    def __init__(
        self,
        in_ch   : int,
        out_ch  : int,
        kernel  : int = 3,
        stride  : int = 1,
        padding : int = 1,
        dilation: int = 1,
        use_bn  : bool = True,
    ):
        super().__init__()
        layers = [
            nn.Conv2d(in_ch, out_ch, kernel,
                      stride=stride, padding=padding,
                      dilation=dilation, bias=not use_bn),
        ]
        if use_bn:
            layers.append(nn.BatchNorm2d(out_ch))
        layers.append(nn.LeakyReLU(0.2, inplace=True))
        self.block = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class ChannelAttention(nn.Module):
    """
    Squeeze-and-Excitation channel attention.
    Helps SR network focus on important frequency channels.
    """

    def __init__(self, ch: int, reduction: int = 16):
        super().__init__()
        self.gate = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(ch, ch // reduction),
            nn.ReLU(inplace=True),
            nn.Linear(ch // reduction, ch),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        w = self.gate(x).view(x.size(0), x.size(1), 1, 1)
        return x * w


class SpatialAttention(nn.Module):
    """
    Spatial attention gate.
    Focuses SR enhancement on salient face regions
    (eyes, nose, mouth) where detail matters most for recognition.
    """

    def __init__(self, kernel: int = 7):
        super().__init__()
        self.conv = nn.Conv2d(2, 1, kernel,
                              padding=kernel // 2, bias=False)
        self.sig  = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        avg_out = x.mean(dim=1, keepdim=True)         # [B, 1, H, W]
        max_out = x.max(dim=1, keepdim=True).values   # [B, 1, H, W]
        attn    = self.sig(self.conv(
            torch.cat([avg_out, max_out], dim=1)
        ))                                             # [B, 1, H, W]
        return x * attn


class CBAM(nn.Module):
    """
    Convolutional Block Attention Module.
    Combines ChannelAttention + SpatialAttention sequentially.
    Enhances discriminative face features critical for recognition.
    """

    def __init__(self, ch: int, reduction: int = 16, spatial_kernel: int = 7):
        super().__init__()
        self.channel = ChannelAttention(ch, reduction)
        self.spatial = SpatialAttention(spatial_kernel)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.channel(x)
        x = self.spatial(x)
        return x


# ── Dense Block ───────────────────────────────────────────────────────────────

class DenseLayer(nn.Module):
    """
    Single dense layer: concatenates all previous feature maps.
    Used inside DenseBlock for feature reuse.
    """

    def __init__(self, in_ch: int, growth_rate: int = 32):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, growth_rate, 3, padding=1, bias=True),
            nn.LeakyReLU(0.2, inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.conv(x)
        return torch.cat([x, out], dim=1)      # dense concat


class DenseBlock(nn.Module):
    """
    Dense Block with N dense layers.
    Each layer receives concatenated outputs of all previous layers.
    Feature reuse ensures rich multi-scale representations.
    """

    def __init__(
        self,
        in_ch      : int,
        growth_rate: int = 32,
        num_layers : int = 5,
    ):
        super().__init__()
        layers   = []
        ch       = in_ch
        for _ in range(num_layers):
            layers.append(DenseLayer(ch, growth_rate))
            ch += growth_rate
        self.layers  = nn.Sequential(*layers)
        self.out_ch  = ch

        # 1×1 conv to compress back to in_ch
        self.compress = nn.Conv2d(ch, in_ch, 1, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.layers(x)
        return self.compress(out)              # [B, in_ch, H, W]


# ── RRDB (Residual-in-Residual Dense Block) ───────────────────────────────────

class RRDB(nn.Module):
    """
    Residual-in-Residual Dense Block (ESRGAN core unit).

    Structure:
        Input
          ├── DenseBlock 1
          ├── DenseBlock 2
          └── DenseBlock 3
               └── × β (residual scaling) → + Input → Output

    The nested residual structure allows very deep networks to
    train stably while preserving high-frequency texture details.
    """

    def __init__(
        self,
        ch         : int = 64,
        growth_rate: int = 32,
        num_dense  : int = 3,       # dense blocks per RRDB
        num_layers : int = 5,       # dense layers per block
        beta       : float = 0.2,   # residual scaling factor
    ):
        super().__init__()
        self.beta   = beta
        self.blocks = nn.ModuleList([
            DenseBlock(ch, growth_rate, num_layers)
            for _ in range(num_dense)
        ])
        self.cbam   = CBAM(ch)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = x
        for block in self.blocks:
            out = out + self.beta * block(out)   # inner residual
        out = self.cbam(out)
        return x + self.beta * out               # outer residual


# ── Pixel Shuffle Upsampler ───────────────────────────────────────────────────

class UpsampleBlock(nn.Module):
    """
    Sub-pixel convolution upsampling (Pixel Shuffle).
    More artifact-free than transposed convolutions for SR tasks.
    Upsamples by factor of 2 per block.
    """

    def __init__(self, ch: int, scale: int = 2):
        super().__init__()
        self.conv = nn.Conv2d(ch, ch * (scale ** 2), 3, padding=1)
        self.ps   = nn.PixelShuffle(scale)          # ch*(s²) → ch, H*s, W*s
        self.act  = nn.LeakyReLU(0.2, inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.ps(self.conv(x)))


# ── Super Resolution Generator ────────────────────────────────────────────────

class SRGenerator(nn.Module):
    """
    ESRGAN-inspired Super Resolution Generator.

    Architecture:
        LR Input [B, 3, H, W]
            │
            ├── Shallow Feature Extraction  [B, 64, H, W]
            │
            ├── RRDB Trunk (N blocks)       [B, 64, H, W]
            │       residual feature learning
            │
            ├── Post-trunk conv             [B, 64, H, W]
            │       + trunk skip connection
            │
            ├── Upsample Blocks             [B, 64, H*s, W*s]
            │       PixelShuffle ×2 per block (2× or 4× total)
            │
            └── Output conv [B, 3, H*s, W*s]
                    HR face image ∈ [0, 1]
    """

    def __init__(
        self,
        in_ch      : int   = 3,
        base_ch    : int   = 64,
        num_rrdb   : int   = 16,        # number of RRDB blocks
        growth_rate: int   = 32,
        scale      : int   = 4,         # 2 or 4
        beta       : float = 0.2,
    ):
        assert scale in (2, 4), "scale must be 2 or 4"
        super().__init__()
        self.scale = scale

        # ── Shallow feature extraction ────────────────────────────────────
        self.head = nn.Conv2d(in_ch, base_ch, 3, padding=1)

        # ── RRDB trunk ───────────────────────────────────────────────────
        self.trunk = nn.Sequential(*[
            RRDB(base_ch, growth_rate, beta=beta)
            for _ in range(num_rrdb)
        ])

        # ── Post-trunk conv ───────────────────────────────────────────────
        self.post_trunk = nn.Conv2d(base_ch, base_ch, 3, padding=1)

        # ── Upsampling ────────────────────────────────────────────────────
        up_blocks = [UpsampleBlock(base_ch, scale=2)]
        if scale == 4:
            up_blocks.append(UpsampleBlock(base_ch, scale=2))
        self.upsample = nn.Sequential(*up_blocks)

        # ── HR output head ────────────────────────────────────────────────
        self.tail = nn.Sequential(
            nn.Conv2d(base_ch, base_ch, 3, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(base_ch, in_ch, 3, padding=1),
            nn.Sigmoid(),                # output ∈ [0, 1]
        )

    def forward(
        self,
        x          : torch.Tensor,
        return_feat: bool = False,
    ) -> Dict[str, torch.Tensor]:
        """
        Args:
            x          : LR face  [B, 3, H, W]   ∈ [0, 1]
            return_feat: if True, return intermediate feature maps

        Returns:
            dict:
                'sr'     : [B, 3, H*scale, W*scale]  super-resolved face
                'feat'   : [B, 64, H, W]              trunk features (optional)
        """
        # Shallow features
        shallow = self.head(x)                    # [B, 64, H, W]

        # RRDB trunk + residual skip
        trunk   = self.trunk(shallow)             # [B, 64, H, W]
        trunk   = self.post_trunk(trunk)          # [B, 64, H, W]
        feat    = shallow + trunk                 # global residual [B, 64, H, W]

        # Upsample
        up      = self.upsample(feat)             # [B, 64, H*s, W*s]

        # Output
        sr      = self.tail(up)                   # [B, 3, H*s, W*s]

        out = {"sr": sr}
        if return_feat:
            out["feat"] = feat

        return out


# ── Discriminator (for GAN training) ─────────────────────────────────────────

class SRDiscriminator(nn.Module):
    """
    PatchGAN-style discriminator for ESRGAN adversarial training.
    Classifies overlapping patches as real HR or fake SR.
    Patch-level discrimination better preserves local texture fidelity.

    Input : HR or SR face  [B, 3, H, W]
    Output: patch logit map [B, 1, H', W']
    """

    def __init__(self, in_ch: int = 3, base_ch: int = 64):
        super().__init__()

        def disc_block(in_c, out_c, stride=2):
            return nn.Sequential(
                nn.Conv2d(in_c, out_c, 4, stride=stride, padding=1, bias=False),
                nn.BatchNorm2d(out_c),
                nn.LeakyReLU(0.2, inplace=True),
            )

        self.net = nn.Sequential(
            # No BN on first layer
            nn.Conv2d(in_ch, base_ch, 4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),

            disc_block(base_ch,      base_ch * 2),   # /4
            disc_block(base_ch * 2,  base_ch * 4),   # /8
            disc_block(base_ch * 4,  base_ch * 8),   # /16
            disc_block(base_ch * 8,  base_ch * 8, stride=1),

            nn.Conv2d(base_ch * 8, 1, 4, stride=1, padding=1),  # patch logit
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)                    # [B, 1, H', W']


# ── Feature Extractor for Perceptual Loss ─────────────────────────────────────

class PerceptualFeatureExtractor(nn.Module):
    """
    Lightweight VGG-like feature extractor for perceptual loss.
    Avoids torchvision dependency by implementing a compact version.

    Extracts multi-scale features used to compare
    SR output with HR ground truth in feature space
    (rather than pixel space), preserving perceptual quality.
    """

    def __init__(self, in_ch: int = 3):
        super().__init__()

        # Block 1: low-level edges/colors
        self.block1 = nn.Sequential(
            nn.Conv2d(in_ch, 64, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.ReLU(inplace=True),
        )

        # Block 2: textures
        self.block2 = nn.Sequential(
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, 3, padding=1),
            nn.ReLU(inplace=True),
        )

        # Block 3: semantic parts (eyes, nose, mouth)
        self.block3 = nn.Sequential(
            nn.MaxPool2d(2),
            nn.Conv2d(128, 256, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 3, padding=1),
            nn.ReLU(inplace=True),
        )

        # Freeze weights — used only for loss computation
        for p in self.parameters():
            p.requires_grad = False

    def forward(
        self,
        x: torch.Tensor,
    ) -> List[torch.Tensor]:
        """Returns multi-scale feature maps: [f1, f2, f3]"""
        f1 = self.block1(x)
        f2 = self.block2(f1)
        f3 = self.block3(f2)
        return [f1, f2, f3]


# ── Loss Functions ────────────────────────────────────────────────────────────

class SRLoss(nn.Module):
    """
    Combined Super Resolution loss:

        L = α · L1(sr, hr)
          + β · Perceptual(sr, hr)
          + γ · GAN(disc(sr))
          + δ · TV(sr)                [total variation smoothness]

    L1         → pixel-level fidelity
    Perceptual → feature-level / semantic quality
    GAN        → photorealism (texture sharpness)
    TV         → suppress checkerboard / noise artifacts
    """

    def __init__(
        self,
        alpha: float = 1.0,     # L1
        beta : float = 0.1,     # perceptual
        gamma: float = 0.005,   # GAN
        delta: float = 1e-4,    # TV
    ):
        super().__init__()
        self.alpha    = alpha
        self.beta     = beta
        self.gamma    = gamma
        self.delta    = delta
        self.l1       = nn.L1Loss()
        self.feat_ext = PerceptualFeatureExtractor()

    def _perceptual_loss(
        self,
        sr: torch.Tensor,
        hr: torch.Tensor,
    ) -> torch.Tensor:
        sr_feats = self.feat_ext(sr)
        hr_feats = self.feat_ext(hr)
        loss = sum(
            F.l1_loss(sf, hf)
            for sf, hf in zip(sr_feats, hr_feats)
        )
        return loss / len(sr_feats)

    def _tv_loss(self, x: torch.Tensor) -> torch.Tensor:
        """Total variation loss to reduce noise artifacts."""
        diff_h = x[:, :, 1:, :] - x[:, :, :-1, :]
        diff_w = x[:, :, :, 1:] - x[:, :, :, :-1]
        return diff_h.abs().mean() + diff_w.abs().mean()

    def forward(
        self,
        sr          : torch.Tensor,            # [B, 3, H*s, W*s]
        hr          : torch.Tensor,            # [B, 3, H*s, W*s]
        disc_fake   : Optional[torch.Tensor],  # discriminator output on SR
    ) -> Tuple[torch.Tensor, Dict[str, float]]:

        l1_loss   = self.l1(sr, hr)
        perc_loss = self._perceptual_loss(sr, hr)
        tv_loss   = self._tv_loss(sr)

        total     = (self.alpha * l1_loss
                   + self.beta  * perc_loss
                   + self.delta * tv_loss)

        metrics   = {
            "l1_loss"  : l1_loss.item(),
            "perc_loss": perc_loss.item(),
            "tv_loss"  : tv_loss.item(),
        }

        # GAN generator loss (if discriminator output provided)
        if disc_fake is not None:
            # Relativistic average GAN loss
            gan_loss  = F.binary_cross_entropy_with_logits(
                disc_fake,
                torch.ones_like(disc_fake),
            )
            total    += self.gamma * gan_loss
            metrics["gan_loss"] = gan_loss.item()

        metrics["total_loss"] = total.item()
        return total, metrics


class DiscriminatorLoss(nn.Module):
    """
    Relativistic average discriminator loss.
        L_D = BCE(real - mean(fake), 1) + BCE(fake - mean(real), 0)
    """

    def forward(
        self,
        real_logits: torch.Tensor,   # [B, 1, H', W']
        fake_logits: torch.Tensor,   # [B, 1, H', W']
    ) -> torch.Tensor:
        real_rel = real_logits - fake_logits.mean()
        fake_rel = fake_logits - real_logits.mean()

        loss_real = F.binary_cross_entropy_with_logits(
            real_rel, torch.ones_like(real_rel)
        )
        loss_fake = F.binary_cross_entropy_with_logits(
            fake_rel, torch.zeros_like(fake_rel)
        )
        return (loss_real + loss_fake) / 2.0


# ── Face-Aware SR Module ──────────────────────────────────────────────────────

class FaceSuperResolution(nn.Module):
    """
    Full face-aware super resolution module.

    Wraps SRGenerator with:
        • Bicubic upsampling fallback for already-HR inputs
        • Face region attention bias
          (higher fidelity for eye/nose/mouth landmarks)
        • Quality gate: skips SR if input is already high-res enough

    Input  : LR face crop  [B, 3, H,   W  ]   ∈ [0, 1]
    Output : HR face crop  [B, 3, H*s, W*s]   ∈ [0, 1]
    """

    def __init__(
        self,
        scale      : int   = 4,
        num_rrdb   : int   = 16,
        base_ch    : int   = 64,
        growth_rate: int   = 32,
        min_size   : int   = 64,     # skip SR if input H or W >= min_size*scale
    ):
        super().__init__()
        self.scale    = scale
        self.min_size = min_size

        self.generator = SRGenerator(
            in_ch=3,
            base_ch=base_ch,
            num_rrdb=num_rrdb,
            growth_rate=growth_rate,
            scale=scale,
        )

    def _bicubic_fallback(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:
        """Bicubic upsampling when SR is not needed."""
        return F.interpolate(
            x,
            scale_factor=self.scale,
            mode="bicubic",
            align_corners=False,
        ).clamp(0, 1)

    def forward(
        self,
        x          : torch.Tensor,
        force_sr   : bool = False,
        return_feat: bool = False,
    ) -> Dict[str, torch.Tensor]:
        """
        Args:
            x          : LR face  [B, 3, H, W]
            force_sr   : always apply SR regardless of input size
            return_feat: return intermediate RRDB features

        Returns:
            dict:
                'hr'        : [B, 3, H*s, W*s]  upscaled face
                'used_sr'   : bool               True if SR network was used
                'feat'      : [B, 64, H, W]      RRDB features (optional)
        """
        H, W     = x.shape[2], x.shape[3]
        use_sr   = force_sr or (H < self.min_size or W < self.min_size)

        if use_sr:
            gen_out  = self.generator(x, return_feat=return_feat)
            hr       = gen_out["sr"]
            feat     = gen_out.get("feat", None)
        else:
            hr       = self._bicubic_fallback(x)
            feat     = None

        out = {
            "hr"     : hr,
            "used_sr": use_sr,
        }
        if return_feat and feat is not None:
            out["feat"] = feat

        return out


# ── Super Resolution Gate (pipeline integration) ──────────────────────────────

class SuperResolutionGate(nn.Module):
    """
    Drop-in super resolution gate for your face recognition pipeline.

    Slot order:
        DenoiserGate → SuperResolutionGate → LivenessGate
            → AntiSpoofingGate → FaceRecognizer

    Automatically upscales low-resolution face crops before
    passing them to downstream modules, improving recognition
    accuracy on low-quality inputs (CCTV, mobile, compressed video).

    Usage:
        sr_gate = SuperResolutionGate(weights_path="sr.pt", scale=4)
        result  = sr_gate(lr_face_crop)
        hr_face = result["hr"]
    """

    def __init__(
        self,
        weights_path: Optional[str] = None,
        scale       : int   = 4,
        num_rrdb    : int   = 16,
        min_size    : int   = 64,
        device      : str   = "cpu",
    ):
        super().__init__()
        self.device = torch.device(device)
        self.model  = FaceSuperResolution(
            scale=scale,
            num_rrdb=num_rrdb,
            min_size=min_size,
        ).to(self.device)
        self.model.eval()

        if weights_path is not None:
            state = torch.load(weights_path, map_location=self.device)
            self.model.load_state_dict(state)
            print(f"[SuperResolutionGate] Loaded weights from: {weights_path}")
        else:
            print("[SuperResolutionGate] ⚠️  No weights loaded — running with random init.")

    @torch.no_grad()
    def forward(
        self,
        x          : torch.Tensor,
        force_sr   : bool = False,
        return_feat: bool = False,
    ) -> Dict[str, torch.Tensor]:
        """
        Args:
            x          : LR face crop  [B, 3, H, W]  ∈ [0, 1]
            force_sr   : always apply SR network
            return_feat: return RRDB feature maps

        Returns:
            dict:
                'hr'     : [B, 3, H*scale, W*scale]  upscaled face
                'used_sr': bool — True if SR network was applied
                'feat'   : RRDB features (if return_feat=True)
        """
        x   = x.to(self.device)
        out = self.model(x, force_sr=force_sr, return_feat=return_feat)
        return out
