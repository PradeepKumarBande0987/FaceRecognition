"""
GAN Discriminator Module
Advanced discriminators for adversarial training of face generation models.
Architectures:
    • StyleGAN2 Discriminator (baseline)
    • Progressive Discriminator (coarse-to-fine)
    • Residual Discriminator (skip connections)
    • Attention-Augmented Discriminator (self-attention)
    • Multi-Scale Discriminator (pyramid)
    • Patch-based Discriminator (local detail focus)
Loss functions:
    • Hinge Loss
    • Wasserstein Loss
    • Spectral Normalization
    • Gradient Penalty (R1, R2, WGAN-GP)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Dict, List
import math


# ── Spectral Normalization ────────────────────────────────────────────────────

class SpectralNorm(nn.Module):
    """
    Spectral Normalization for weight matrices.
    Stabilizes discriminator training by constraining Lipschitz constant.

    Normalizes weights by largest singular value:
        w_normalized = w / σ(w)

    Prevents discriminator from becoming too powerful too quickly,
    enabling stable adversarial training.
    """

    def __init__(self, module: nn.Module, name: str = "weight", n_power_iterations: int = 1):
        super().__init__()
        self.module = module
        self.name = name
        self.n_power_iterations = n_power_iterations

        # Register buffer for u vector (left singular vector)
        if not self._has_spectral_norm():
            self._create_spectral_norm()

    def _has_spectral_norm(self) -> bool:
        return hasattr(self.module, f"{self.name}_sn")

    def _create_spectral_norm(self):
        w = getattr(self.module, self.name)
        if isinstance(w, nn.Parameter):
            # Create u vector
            h, w_shape = w.shape[0], w.shape[1:].numel()
            u = torch.randn(h, device=w.device)
            u = F.normalize(u, dim=0)
            self.register_buffer(f"{self.name}_u", u)

    def _spectral_norm(self) -> float:
        """Compute spectral norm (largest singular value)."""
        w = getattr(self.module, self.name)
        if isinstance(w, nn.Parameter):
            # Reshape to 2D: [out_features, in_features]
            w_mat = w.view(w.shape[0], -1)

            u = getattr(self, f"{self.name}_u")

            # Power iteration
            for _ in range(self.n_power_iterations):
                v = F.normalize(w_mat.t() @ u, dim=0)
                u = F.normalize(w_mat @ v, dim=0)

            # Update u
            setattr(self, f"{self.name}_u", u)

            # Compute spectral norm: σ = u^T W v
            sigma = u @ w_mat @ v
            return sigma.item()
        return 1.0

    def forward(self, *args, **kwargs):
        # Apply spectral normalization to weights
        w = getattr(self.module, self.name)
        if isinstance(w, nn.Parameter):
            sigma = self._spectral_norm()
            w_normalized = w / (sigma + 1e-12)
            setattr(self.module, self.name, w_normalized)

        return self.module(*args, **kwargs)


def spectral_norm(module: nn.Module, name: str = "weight") -> nn.Module:
    """Apply spectral normalization to a module."""
    return SpectralNorm(module, name=name)


# ── Building Blocks ───────────────────────────────────────────────────────────

class SNConv2d(nn.Module):
    """Spectral-Normalized 2D Convolution."""

    def __init__(
        self,
        in_ch: int,
        out_ch: int,
        kernel: int = 3,
        stride: int = 1,
        padding: int = 1,
        bias: bool = True,
    ):
        super().__init__()
        self.conv = spectral_norm(
            nn.Conv2d(in_ch, out_ch, kernel, stride=stride,
                     padding=padding, bias=bias),
            name="weight",
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv.module(x)


class SNLinear(nn.Module):
    """Spectral-Normalized Linear layer."""

    def __init__(self, in_dim: int, out_dim: int, bias: bool = True):
        super().__init__()
        self.linear = spectral_norm(
            nn.Linear(in_dim, out_dim, bias=bias),
            name="weight",
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear.module(x)


class ResidualBlock(nn.Module):
    """
    Residual block for discriminator.
    Helps train very deep discriminators without vanishing gradients.
    """

    def __init__(
        self,
        in_ch: int,
        out_ch: int,
        downsample: bool = False,
    ):
        super().__init__()
        self.conv1 = SNConv2d(in_ch, out_ch, kernel=3, padding=1)
        self.conv2 = SNConv2d(out_ch, out_ch, kernel=3, padding=1)

        # Skip connection
        if downsample or in_ch != out_ch:
            self.skip = nn.Sequential(
                nn.AvgPool2d(2) if downsample else nn.Identity(),
                SNConv2d(in_ch, out_ch, kernel=1, padding=0),
            )
        else:
            self.skip = nn.Identity()

        self.downsample = downsample

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        skip = self.skip(x)

        h = F.leaky_relu(x, 0.2)
        h = self.conv1(h)
        h = F.leaky_relu(h, 0.2)
        h = self.conv2(h)

        if self.downsample:
            h = F.avg_pool2d(h, 2)

        return h + skip


class SelfAttentionBlock(nn.Module):
    """
    Self-Attention block for discriminator.
    Allows discriminator to attend to distant spatial regions,
    improving detection of global inconsistencies in synthetic faces.

    Usage: identifies unnatural facial structures, misaligned features.
    """

    def __init__(self, ch: int, reduction: int = 8):
        super().__init__()
        self.ch = ch

        # Query, Key, Value projections
        self.query = SNConv2d(ch, ch // reduction, kernel=1)
        self.key   = SNConv2d(ch, ch // reduction, kernel=1)
        self.value = SNConv2d(ch, ch, kernel=1)

        # Output projection
        self.out_proj = SNConv2d(ch, ch, kernel=1)

        # Attention scale
        self.gamma = nn.Parameter(torch.tensor(0.0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: [B, ch, H, W]"""
        B, C, H, W = x.shape

        # Project to query, key, value
        q = self.query(x).view(B, -1, H * W)          # [B, C//r, HW]
        k = self.key(x).view(B, -1, H * W)            # [B, C//r, HW]
        v = self.value(x).view(B, C, H * W)           # [B, C, HW]

        # Attention: softmax(Q^T K) V
        attn = torch.bmm(q.transpose(1, 2), k)        # [B, HW, HW]
        attn = F.softmax(attn / math.sqrt(self.ch), dim=-1)  # normalize

        # Apply attention to values
        out = torch.bmm(v, attn.transpose(1, 2))      # [B, C, HW]
        out = out.view(B, C, H, W)

        # Output projection + residual
        out = self.out_proj(out)
        return x + self.gamma * out


class MinibatchStd(nn.Module):
    """
    Minibatch Standard Deviation (StyleGAN trick).
    Reduces mode collapse by feeding discriminator diversity statistics.

    Appends std of activations across batch as additional feature,
    preventing discriminator from ignoring inter-sample variation.
    """

    def __init__(self, group_size: int = 4):
        super().__init__()
        self.group_size = group_size

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: [B, C, H, W] → [B, C+1, H, W]"""
        B, C, H, W = x.shape

        if B < self.group_size:
            # If batch too small, just return input
            return x

        # Compute std within groups
        group_std = x.view(self.group_size, -1, C, H, W)
        group_std = group_std.std(dim=1)                # [G, C, H, W]
        group_std = group_std.mean(dim=[1, 2, 3], keepdim=True)  # [G, 1, 1, 1]
        group_std = group_std.expand(self.group_size, 1, H, W)   # [G, 1, H, W]

        # Expand to full batch
        group_std = torch.cat([group_std for _ in range(B // self.group_size)], dim=0)

        return torch.cat([x, group_std], dim=1)  # [B, C+1, H, W]


# ── Base Discriminator ────────────────────────────────────────────────────────

class BaseDiscriminator(nn.Module):
    """
    Base discriminator architecture.
    Downsamples: input → final classification.
    """

    def __init__(
        self,
        input_res: int = 256,
        fmap_base: int = 16384,
        fmap_max: int = 512,
        use_attention: bool = True,
        use_minibatch_std: bool = True,
    ):
        super().__init__()
        assert input_res in (64, 128, 256)

        self.input_res = input_res
        self.use_attention = use_attention
        self.use_minibatch_std = use_minibatch_std

        # Compute number of downsampling layers
        res_log2 = int(math.log2(input_res))

        # FromRGB: first layer to encode RGB
        in_ch = min(int(fmap_base / input_res), fmap_max)
        self.from_rgb = SNConv2d(3, in_ch, kernel=1)

        # Downsampling blocks
        self.blocks = nn.ModuleList()
        self.attention_blocks = nn.ModuleList()

        for i in range(res_log2 - 1, 1, -1):  # from input_res down to 4×4
            res = 2 ** i
            in_ch_  = min(int(fmap_base / res), fmap_max)
            out_ch_ = min(int(fmap_base / (res // 2)), fmap_max)

            self.blocks.append(
                self._make_downsample_block(in_ch_, out_ch_)
            )

            # Add attention at lower resolutions (16×16 and below)
            if use_attention and res <= 16:
                self.attention_blocks.append(SelfAttentionBlock(out_ch_))
            else:
                self.attention_blocks.append(nn.Identity())

        # Final classification block: 4×4
        final_ch = min(int(fmap_base / 4), fmap_max)

        if use_minibatch_std:
            self.minibatch_std = MinibatchStd()
            final_in_ch = final_ch + 1
        else:
            self.minibatch_std = None
            final_in_ch = final_ch

        self.final_block = nn.Sequential(
            SNConv2d(final_in_ch, final_ch, kernel=3, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            SNConv2d(final_ch, final_ch, kernel=4, padding=0),  # 4×4 → 1×1
            nn.LeakyReLU(0.2, inplace=True),
            nn.Flatten(),
            SNLinear(final_ch, 1),
        )

    def _make_downsample_block(self, in_ch: int, out_ch: int) -> nn.Module:
        return nn.Sequential(
            SNConv2d(in_ch, out_ch, kernel=3, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.AvgPool2d(2),
            SNConv2d(out_ch, out_ch, kernel=3, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
        )

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """x: [B, 3, res, res]"""
        x = self.from_rgb(x)

        # Progressive downsampling
        for block, attn in zip(self.blocks, self.attention_blocks):
            x = block(x)
            x = attn(x)

        # Minibatch std
        if self.minibatch_std is not None:
            x = self.minibatch_std(x)

        # Final classification
        logit = self.final_block(x)  # [B, 1]

        return {
            "logit": logit,
            "feat" : x,  # intermediate features for losses
        }


# ── Progressive Discriminator ─────────────────────────────────────────────────

class ProgressiveDiscriminator(nn.Module):
    """
    Progressive Discriminator for stable adversarial training.

    Starts training at low resolution, gradually adds detail-capturing layers.
    Prevents discriminator from overfitting to fine details early,
    improves convergence stability.

    Architecture grows: 4×4 → 8×8 → 16×16 → ... → full resolution
    """

    def __init__(
        self,
        max_res: int = 256,
        fmap_base: int = 16384,
        fmap_max: int = 512,
    ):
        super().__init__()
        self.max_res = max_res
        self.current_res = 4

        # Build all possible layers
        self.from_rgb_blocks = nn.ModuleDict()
        self.downsample_blocks = nn.ModuleDict()

        res_log2 = int(math.log2(max_res))

        for res_idx in range(2, res_log2 + 1):
            res = 2 ** res_idx
            in_ch = min(int(fmap_base / res), fmap_max)
            out_ch = min(int(fmap_base / (res // 2)), fmap_max)

            # FromRGB at this resolution
            self.from_rgb_blocks[f"res_{res}"] = SNConv2d(3, in_ch, kernel=1)

            # Downsampling block to next lower res
            self.downsample_blocks[f"res_{res}"] = nn.Sequential(
                SNConv2d(in_ch, out_ch, kernel=3, padding=1),
                nn.LeakyReLU(0.2, inplace=True),
                nn.AvgPool2d(2),
                SNConv2d(out_ch, out_ch, kernel=3, padding=1),
                nn.LeakyReLU(0.2, inplace=True),
            )

        # Final 4×4 block
        final_ch = min(int(fmap_base / 4), fmap_max)
        self.final_block = nn.Sequential(
            MinibatchStd(),
            SNConv2d(final_ch + 1, final_ch, kernel=3, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            SNConv2d(final_ch, final_ch, kernel=4, padding=0),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Flatten(),
            SNLinear(final_ch, 1),
        )

        self.alpha = 0.0  # blending factor for smooth transitions
        self.fade_in_res = 4

    def set_resolution(self, res: int, alpha: float = 1.0):
        """Set current resolution for progressive training."""
        assert res in [2**i for i in range(2, int(math.log2(self.max_res)) + 1)]
        self.current_res = res
        self.alpha = alpha

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """x: [B, 3, current_res, current_res]"""
        res = x.shape[2]
        assert res == self.current_res

        # Upsample if training at lower res
        if res < self.max_res:
            # For now, assume input is at correct resolution
            pass

        # FromRGB at current resolution
        x = self.from_rgb_blocks[f"res_{res}"](x)

        # Progressive downsampling to 4×4
        res_log2 = int(math.log2(res))
        for res_idx in range(res_log2, 1, -1):
            curr_res = 2 ** res_idx
            x = self.downsample_blocks[f"res_{curr_res}"](x)

        # Final classification
        logit = self.final_block(x)

        return {
            "logit": logit,
            "feat" : x,
        }


# ── Multi-Scale Discriminator ────────────────────────────────────────────────

class MultiScaleDiscriminator(nn.Module):
    """
    Multi-Scale Discriminator.
    Operates on image pyramid: original, 1/2, 1/4, 1/8 resolution.

    Benefits:
        • Captures both local and global inconsistencies
        • Improves detection of artifacts at multiple scales
        • Stabilizes training via multi-task learning
    """

    def __init__(
        self,
        input_res: int = 256,
        num_scales: int = 3,
    ):
        super().__init__()
        self.num_scales = num_scales
        self.discriminators = nn.ModuleList()

        for _ in range(num_scales):
            self.discriminators.append(
                BaseDiscriminator(input_res=input_res)
            )

        self.downsample = nn.AvgPool2d(2, stride=2)

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """x: [B, 3, res, res]"""
        logits = []
        feats = []

        curr_x = x
        for disc in self.discriminators:
            out = disc(curr_x)
            logits.append(out["logit"])
            feats.append(out["feat"])
            curr_x = self.downsample(curr_x)

        # Combine multi-scale logits (averaged)
        combined_logit = torch.stack(logits, dim=0).mean(dim=0)

        return {
            "logit"  : combined_logit,
            "logits" : logits,  # individual scale logits
            "feats"  : feats,
        }


# ── Patch-based Discriminator ────────────────────────────────────────────────

class PatchDiscriminator(nn.Module):
    """
    Patch-based Discriminator (PatchGAN).
    Classifies overlapping patches as real/fake instead of global label.

    Benefits:
        • Focuses on local texture quality
        • Better preservation of fine details
        • Reduces checkerboard artifacts
    """

    def __init__(
        self,
        input_res: int = 256,
        patch_size: int = 70,  # 70×70 patches (N layers = log2(res/patch))
    ):
        super().__init__()
        self.patch_size = patch_size

        num_layers = int(math.log2(input_res / patch_size)) + 1

        channels = [3, 64, 128, 256, 512, 512]
        layers = []

        for i in range(len(channels) - 1):
            in_ch, out_ch = channels[i], channels[i + 1]
            layers.append(
                SNConv2d(in_ch, out_ch, kernel=4, stride=2, padding=1)
            )
            if i < len(channels) - 2:
                layers.append(nn.LeakyReLU(0.2, inplace=True))

        self.net = nn.Sequential(*layers)

        # Final layer: predict per-patch
        self.classifier = SNConv2d(channels[-1], 1, kernel=4, padding=1)

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """x: [B, 3, res, res] → logit map [B, 1, H', W']"""
        feat = self.net(x)
        logit_map = self.classifier(feat)  # [B, 1, H', W']

        # Average over spatial dimensions for global logit
        global_logit = logit_map.mean()

        return {
            "logit"    : global_logit,
            "logit_map": logit_map,  # per-patch logits
            "feat"     : feat,
        }


# ── Loss Functions ────────────────────────────────────────────────────────────

class DiscriminatorLoss(nn.Module):
    """
    Advanced discriminator losses for stable training.
    """

    def __init__(
        self,
        loss_type: str = "hinge",  # "hinge", "wasserstein", "bce"
        lambda_gp: float = 10.0,   # gradient penalty weight
    ):
        super().__init__()
        assert loss_type in ("hinge", "wasserstein", "bce")
        self.loss_type = loss_type
        self.lambda_gp = lambda_gp

    def adversarial_loss(
        self,
        real_logit: torch.Tensor,  # [B] or [B, 1]
        fake_logit: torch.Tensor,  # [B] or [B, 1]
    ) -> torch.Tensor:
        """Main adversarial loss."""
        real_logit = real_logit.squeeze()
        fake_logit = fake_logit.squeeze()

        if self.loss_type == "hinge":
            # Hinge loss: max(0, 1 - real) + max(0, 1 + fake)
            loss = F.relu(1.0 - real_logit).mean() + F.relu(1.0 + fake_logit).mean()

        elif self.loss_type == "wasserstein":
            # Wasserstein loss: -real + fake
            loss = -real_logit.mean() + fake_logit.mean()

        elif self.loss_type == "bce":
            # Binary cross-entropy
            loss = (F.binary_cross_entropy_with_logits(real_logit, torch.ones_like(real_logit))
                  + F.binary_cross_entropy_with_logits(fake_logit, torch.zeros_like(fake_logit)))

        return loss

    def r1_penalty(
        self,
        real_images: torch.Tensor,
        discriminator: nn.Module,
    ) -> torch.Tensor:
        """
        R1 Gradient Penalty (StyleGAN2).
        Penalizes large gradients of D w.r.t. real images.
        """
        real_images.requires_grad = True

        real_out = discriminator(real_images)
        real_logit = real_out["logit"] if isinstance(real_out, dict) else real_out

        # Compute gradients
        real_loss = real_logit.sum()
        grads = torch.autograd.grad(
            outputs=real_loss,
            inputs=real_images,
            create_graph=True,
            retain_graph=True,
        )[0]

        # R1: penalize gradient norm
        penalty = (grads.view(grads.size(0), -1) ** 2).sum(dim=1).mean()
        return penalty

    def r2_penalty(
        self,
        fake_images: torch.Tensor,
        discriminator: nn.Module,
    ) -> torch.Tensor:
        """
        R2 Gradient Penalty (StyleGAN2).
        Same as R1 but on fake images.
        """
        fake_images.requires_grad = True

        fake_out = discriminator(fake_images)
        fake_logit = fake_out["logit"] if isinstance(fake_out, dict) else fake_out

        fake_loss = fake_logit.sum()
        grads = torch.autograd.grad(
            outputs=fake_loss,
            inputs=fake_images,
            create_graph=True,
            retain_graph=True,
        )[0]

        penalty = (grads.view(grads.size(0), -1) ** 2).sum(dim=1).mean()
        return penalty

    def wgan_gp(
        self,
        real_images: torch.Tensor,
        fake_images: torch.Tensor,
        discriminator: nn.Module,
    ) -> torch.Tensor:
        """
        WGAN Gradient Penalty (Wasserstein GAN).
        Penalizes gradients on interpolated images.
        """
        B = real_images.size(0)

        # Random interpolation
        alpha = torch.rand(B, 1, 1, 1, device=real_images.device)
        interp = alpha * real_images + (1 - alpha) * fake_images
        interp.requires_grad = True

        interp_out = discriminator(interp)
        interp_logit = interp_out["logit"] if isinstance(interp_out, dict) else interp_out

        interp_loss = interp_logit.sum()
        grads = torch.autograd.grad(
            outputs=interp_loss,
            inputs=interp,
            create_graph=True,
            retain_graph=True,
        )[0]

        # Gradient norm: should be ~1
        grad_norms = torch.sqrt((grads.view(B, -1) ** 2).sum(dim=1) + 1e-12)
        penalty = ((grad_norms - 1.0) ** 2).mean()
        return penalty

    def forward(
        self,
        real_logit: torch.Tensor,
        fake_logit: torch.Tensor,
        real_images: Optional[torch.Tensor] = None,
        fake_images: Optional[torch.Tensor] = None,
        discriminator: Optional[nn.Module] = None,
        penalty_type: str = "r1",
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Args:
            real_logit    : discriminator output on real
            fake_logit    : discriminator output on fake
            real_images   : real image batch (for R1 penalty)
            fake_images   : fake image batch (for WGAN-GP)
            discriminator : discriminator module (for penalties)
            penalty_type  : "r1", "r2", or "wgan_gp"

        Returns:
            loss, metrics
        """
        # Main adversarial loss
        adv_loss = self.adversarial_loss(real_logit, fake_logit)
        total = adv_loss
        metrics = {"adv_loss": adv_loss.item()}

        # Gradient penalty
        if discriminator is not None and real_images is not None:
            if penalty_type == "r1":
                gp = self.r1_penalty(real_images, discriminator)
                total += self.lambda_gp * gp
                metrics["r1_penalty"] = gp.item()

            elif penalty_type == "r2" and fake_images is not None:
                gp = self.r2_penalty(fake_images, discriminator)
                total += self.lambda_gp * gp
                metrics["r2_penalty"] = gp.item()

            elif penalty_type == "wgan_gp" and fake_images is not None:
                gp = self.wgan_gp(real_images, fake_images, discriminator)
                total += self.lambda_gp * gp
                metrics["wgan_gp"] = gp.item()

        metrics["total_loss"] = total.item()
        return total, metrics


# ── Discriminator Gate (pipeline integration) ─────────────────────────────────

class GANDiscriminatorGate(nn.Module):
    """
    Drop-in discriminator gate for GAN training pipeline.

    Integrates with StyleGAN2Generator for end-to-end adversarial training.

    Usage:
        gen_gate  = fr.GANGeneratorGate(weights_path="gen.pt")
        disc_gate = fr.GANDiscriminatorGate(weights_path="disc.pt")
        loss_fn   = fr.DiscriminatorLoss(loss_type="hinge", lambda_gp=10.0)

        # Training loop
        real = ...
        z    = torch.randn(B, 512)
        fake = gen_gate.generate_batch(B)

        real_out = disc_gate(real)
        fake_out = disc_gate(fake.detach())

        d_loss, metrics = loss_fn(
            real_out["logit"], fake_out["logit"],
            real_images=real, discriminator=disc_gate
        )
    """

    def __init__(
        self,
        weights_path: Optional[str] = None,
        discriminator_type: str = "base",      # "base", "progressive", "multiscale", "patch"
        input_res: int = 256,
        fmap_base: int = 16384,
        fmap_max: int = 512,
        device: str = "cpu",
    ):
        super().__init__()
        self.device = torch.device(device)
        self.disc_type = discriminator_type

        if discriminator_type == "base":
            self.discriminator = BaseDiscriminator(
                input_res=input_res,
                fmap_base=fmap_base,
                fmap_max=fmap_max,
            ).to(self.device)

        elif discriminator_type == "progressive":
            self.discriminator = ProgressiveDiscriminator(
                max_res=input_res,
                fmap_base=fmap_base,
                fmap_max=fmap_max,
            ).to(self.device)

        elif discriminator_type == "multiscale":
            self.discriminator = MultiScaleDiscriminator(
                input_res=input_res,
                num_scales=3,
            ).to(self.device)

        elif discriminator_type == "patch":
            self.discriminator = PatchDiscriminator(
                input_res=input_res,
                patch_size=70,
            ).to(self.device)

        else:
            raise ValueError(f"Unknown discriminator type: {discriminator_type}")

        self.discriminator.eval()

        if weights_path is not None:
            state = torch.load(weights_path, map_location=self.device)
            self.discriminator.load_state_dict(state)
            print(f"[GANDiscriminatorGate] Loaded weights from: {weights_path}")
        else:
            print("[GANDiscriminatorGate] ⚠️  No weights loaded — random init.")

    def forward(
        self,
        x: torch.Tensor,   # [B, 3, res, res] ∈ [-1, 1]
    ) -> Dict[str, torch.Tensor]:
        """
        Args:
            x: image batch

        Returns:
            dict with keys:
                'logit'  : [B] or [B, 1]
                'feat'   : intermediate features
                (others depending on discriminator type)
        """
        x = x.to(self.device)
        return self.discriminator(x)
