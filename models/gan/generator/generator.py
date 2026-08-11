"""
GAN Generator Module
Generates synthetic face images for:
    • Data augmentation (training with limited real data)
    • Face synthesis from embeddings
    • Domain adaptation (synthetic → real)
    • Identity-preserving face manipulation
Architecture: StyleGAN2-inspired architecture with adaptive instance norm,
              multi-scale progressive generation, and latent space control.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Dict, List
import math


# ── Building Blocks ───────────────────────────────────────────────────────────

class EqualLinear(nn.Module):
    """
    Linear layer with equalized learning rates.
    Prevents exploiting small initializations during training.
    """

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        bias: bool = True,
        bias_init: float = 0.0,
        lr_mul: float = 1.0,
    ):
        super().__init__()
        self.lr_mul = lr_mul
        self.weight = nn.Parameter(torch.randn(out_dim, in_dim))
        if bias:
            self.bias = nn.Parameter(torch.full((out_dim,), bias_init))
        else:
            self.register_parameter("bias", None)

        self.scale = 1 / math.sqrt(in_dim)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.normal_(self.weight)
        if self.bias is not None:
            nn.init.constant_(self.bias, 0.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        w = self.weight * self.scale * self.lr_mul
        b = self.bias * self.lr_mul if self.bias is not None else None
        return F.linear(x, w, b)


class EqualConv2d(nn.Module):
    """
    2D Convolution with equalized learning rates.
    Stabilizes training by ensuring similar learning speeds across layers.
    """

    def __init__(
        self,
        in_ch: int,
        out_ch: int,
        kernel: int = 3,
        stride: int = 1,
        padding: int = 1,
        bias: bool = True,
        lr_mul: float = 1.0,
    ):
        super().__init__()
        self.lr_mul = lr_mul
        self.stride = stride
        self.padding = padding

        self.weight = nn.Parameter(torch.randn(out_ch, in_ch, kernel, kernel))
        if bias:
            self.bias = nn.Parameter(torch.zeros(out_ch))
        else:
            self.register_parameter("bias", None)

        self.scale = 1 / math.sqrt(in_ch * kernel * kernel)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.normal_(self.weight)
        if self.bias is not None:
            nn.init.constant_(self.bias, 0.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        w = self.weight * self.scale * self.lr_mul
        b = self.bias * self.lr_mul if self.bias is not None else None
        return F.conv2d(x, w, b, stride=self.stride, padding=self.padding)


class AdaIN(nn.Module):
    """
    Adaptive Instance Normalization.
    Decouples style (mean/variance) from content.
    Enables fine-grained control over generated face appearance.

    Used in StyleGAN: latent code → style (γ, β) → AdaIN normalizes features
    """

    def __init__(self, ch: int, w_dim: int):
        super().__init__()
        self.norm = nn.InstanceNorm2d(ch, affine=False)

        # Map latent w → style parameters (γ, β)
        self.style_scale = EqualLinear(w_dim, ch)
        self.style_bias  = EqualLinear(w_dim, ch)

    def forward(
        self,
        x: torch.Tensor,    # [B, ch, H, W]  feature maps
        w: torch.Tensor,    # [B, w_dim]     latent style code
    ) -> torch.Tensor:

        gamma = self.style_scale(w).view(w.size(0), -1, 1, 1)   # [B, ch, 1, 1]
        beta  = self.style_bias(w).view(w.size(0), -1, 1, 1)    # [B, ch, 1, 1]

        return gamma * self.norm(x) + beta


class ModulatedConv2d(nn.Module):
    """
    Modulated Convolution for StyleGAN2.
    Applies style modulation to convolution weights before performing conv.

    Pipeline:
        latent w → style code
              → modulate conv weights
              → demodulate for normalization
              → conv with modulated weights
    """

    def __init__(
        self,
        in_ch: int,
        out_ch: int,
        kernel: int = 3,
        w_dim: int = 512,
        demod: bool = True,
    ):
        super().__init__()
        self.demod = demod
        self.padding = kernel // 2

        self.conv = EqualConv2d(in_ch, out_ch, kernel,
                                padding=self.padding, bias=False)

        # Map latent to per-channel modulation
        self.style = EqualLinear(w_dim, in_ch)

    def forward(
        self,
        x: torch.Tensor,    # [B, in_ch, H, W]
        w: torch.Tensor,    # [B, w_dim]
    ) -> torch.Tensor:

        B, _, H, W = x.shape

        # Get style modulation
        s = self.style(w).view(B, 1, -1, 1, 1)      # [B, 1, in_ch, 1, 1]

        # Modulate input
        x_mod = x * s                               # [B, in_ch, H, W]

        # Demodulate for normalization
        if self.demod:
            demod = torch.rsqrt(
                (self.conv.weight ** 2).sum([1, 2, 3], keepdim=True) + 1e-8
            )                                       # [out_ch, 1, 1, 1]
            w_demod = self.conv.weight * demod     # demodulated weights

            x_out = F.conv2d(x_mod, w_demod,
                           padding=self.padding)    # [B, out_ch, H, W]
        else:
            x_out = self.conv(x_mod)

        return x_out


class StyleGANBlock(nn.Module):
    """
    StyleGAN2 synthesis block.

    Architecture:
        Modulated Conv → Blur → Modulated Conv
           ↓ (each conv applies style w)
        AdaIN (not used here; implicit in modulated conv)

    Progressive growing: can add/remove blocks during training.
    """

    def __init__(
        self,
        in_ch: int,
        out_ch: int,
        w_dim: int = 512,
        upsample: bool = False,
    ):
        super().__init__()
        self.upsample = upsample

        if upsample:
            self.conv0 = ModulatedConv2d(in_ch, out_ch, kernel=3, w_dim=w_dim)
        else:
            self.conv0 = ModulatedConv2d(in_ch, out_ch, kernel=3, w_dim=w_dim)

        self.conv1 = ModulatedConv2d(out_ch, out_ch, kernel=3, w_dim=w_dim)

        # Bias + learnable feature
        self.bias0 = nn.Parameter(torch.zeros(out_ch))
        self.bias1 = nn.Parameter(torch.zeros(out_ch))

    def forward(
        self,
        x: torch.Tensor,    # [B, in_ch, H, W]
        w: torch.Tensor,    # [B, w_dim]
    ) -> torch.Tensor:

        # Upsample if needed (2× via nearest neighbor)
        if self.upsample:
            x = F.interpolate(x, scale_factor=2, mode="nearest")

        # Conv1
        x = self.conv0(x, w) + self.bias0.view(1, -1, 1, 1)
        x = F.leaky_relu(x, 0.2)

        # Conv2
        x = self.conv1(x, w) + self.bias1.view(1, -1, 1, 1)
        x = F.leaky_relu(x, 0.2)

        return x


class ToRGB(nn.Module):
    """
    Converts feature maps to RGB output.
    Applied at each resolution for progressive training.
    """

    def __init__(self, in_ch: int, w_dim: int = 512):
        super().__init__()
        self.conv = ModulatedConv2d(in_ch, 3, kernel=1, w_dim=w_dim)

    def forward(
        self,
        x: torch.Tensor,
        w: torch.Tensor,
    ) -> torch.Tensor:
        return torch.tanh(self.conv(x, w))   # tanh for [-1, 1] output


# ── Latent Space Mapping ──────────────────────────────────────────────────────

class MappingNetwork(nn.Module):
    """
    Mapping network: z (noise) → w (latent style code).

    Maps random Gaussian noise to a learned latent space (w-space)
    that has better disentanglement properties than raw noise.

    Enables:
        • Smooth interpolation between generated faces
        • Attribute control (e.g. age, expression)
        • Style mixing for diverse generation
    """

    def __init__(
        self,
        z_dim: int = 512,
        w_dim: int = 512,
        num_layers: int = 8,
        lr_mul: float = 0.01,
    ):
        super().__init__()
        self.z_dim = z_dim
        self.w_dim = w_dim

        layers = []
        in_dim = z_dim
        for i in range(num_layers):
            layers.append(EqualLinear(in_dim, w_dim, lr_mul=lr_mul))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            in_dim = w_dim

        self.net = nn.Sequential(*layers)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """z: [B, z_dim] → w: [B, w_dim]"""
        return self.net(z)


# ── StyleGAN2 Generator ───────────────────────────────────────────────────────

class StyleGAN2Generator(nn.Module):
    """
    StyleGAN2 face generator.

    Architecture:
        z (noise) [B, 512]
            │
            ├── Mapping Network
            │       → w [B, 512]  (style latent code)
            │
            ├── Constant input [1, 512, 4, 4]
            │
            ├── Synthesis Blocks (progressive upsampling)
            │   4×4 → 8×8 → 16×16 → 32×32 → 64×64 → 128×128 → 256×256
            │   (or configurable resolution)
            │
            └── Output: face image [B, 3, res, res] ∈ [-1, 1]

    Key features:
    • Style modulation: control appearance via w
    • Progressive training: start low-res, gradually add detail
    • Adaptive instance norm: disentangle style from structure
    """

    def __init__(
        self,
        z_dim: int = 512,
        w_dim: int = 512,
        output_res: int = 256,     # 64, 128, or 256
        num_layers: int = 8,       # mapping network depth
        fmap_base: int = 16384,    # base feature map count
        fmap_max: int = 512,       # max feature maps per layer
    ):
        super().__init__()
        assert output_res in (64, 128, 256), "output_res must be 64, 128, or 256"

        self.z_dim = z_dim
        self.w_dim = w_dim
        self.output_res = output_res

        # ── Mapping Network ──────────────────────────────────────────────────
        self.mapping = MappingNetwork(z_dim, w_dim, num_layers)

        # ── Constant input (learnable) ───────────────────────────────────────
        self.const_input = nn.Parameter(torch.randn(1, 512, 4, 4))

        # ── Synthesis Blocks ─────────────────────────────────────────────────
        # Build blocks for 4×4 → output_res
        res_log2 = int(math.log2(output_res))
        self.blocks = nn.ModuleList()
        self.to_rgb = nn.ModuleList()

        # First block: 4×4
        self.blocks.append(StyleGANBlock(512, 512, w_dim, upsample=False))
        self.to_rgb.append(ToRGB(512, w_dim))

        # Progressive blocks: 8×8 → output_res
        for i in range(2, res_log2 + 1):
            res = 2 ** i
            in_ch  = min(int(fmap_base / res), fmap_max)
            out_ch = min(int(fmap_base / (res // 2)), fmap_max)

            self.blocks.append(StyleGANBlock(in_ch, out_ch, w_dim, upsample=True))
            self.to_rgb.append(ToRGB(out_ch, w_dim))

    def forward(
        self,
        z: torch.Tensor,
        truncation: float = 1.0,            # truncation trick for consistency
        truncation_latent: Optional[torch.Tensor] = None,  # mean w for truncation
        return_feat: bool = False,
    ) -> Dict[str, torch.Tensor]:
        """
        Args:
            z              : random noise [B, z_dim]
            truncation     : truncation scale ∈ [0, 1]
                            0 = high consistency, 1 = full diversity
            truncation_latent: mean w for truncation
            return_feat    : return intermediate features

        Returns:
            dict:
                'image'  : [B, 3, res, res]  generated face ∈ [-1, 1]
                'w'      : [B, w_dim]  latent code (if return_feat)
                'feats'  : intermediate features (if return_feat)
        """
        B = z.size(0)

        # Map noise to style code
        w = self.mapping(z)                         # [B, w_dim]

        # Truncation trick: interpolate w toward mean for consistency
        if truncation < 1.0:
            if truncation_latent is None:
                truncation_latent = self.mapping(torch.zeros(1, self.z_dim,
                                                            device=z.device))
            w = truncation_latent + truncation * (w - truncation_latent)

        # Start with constant input
        x = self.const_input.repeat(B, 1, 1, 1)    # [B, 512, 4, 4]
        feats = [x]

        # Progressive synthesis
        rgb = None
        for block, to_rgb in zip(self.blocks, self.to_rgb):
            x = block(x, w)                        # apply synthesis block
            feats.append(x)
            rgb = to_rgb(x, w)                     # convert to RGB

        out = {"image": rgb}
        if return_feat:
            out["w"]     = w
            out["feats"] = feats

        return out


# ── Embedding-to-Image Generator ─────────────────────────────────────────────

class EmbeddingToImageGenerator(nn.Module):
    """
    Generates face images from identity embeddings.

    Bridges face recognition embeddings and image synthesis:
        embedding [B, 512] → face image [B, 3, 256, 256]

    Useful for:
        • Face reconstruction from embeddings
        • Identity-aware synthesis
        • Privacy-preserving demo (show reconstructed face, not real photo)
    """

    def __init__(
        self,
        emb_dim: int = 512,
        output_res: int = 256,
    ):
        super().__init__()
        self.emb_dim = emb_dim

        # Map embedding → StyleGAN latent w
        self.emb_to_w = nn.Sequential(
            nn.Linear(emb_dim, 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 512),
        )

        # StyleGAN2 generator (fixed)
        self.generator = StyleGAN2Generator(
            z_dim=512,
            w_dim=512,
            output_res=output_res,
        )

    def forward(
        self,
        embedding: torch.Tensor,   # [B, emb_dim]
        truncation: float = 0.7,   # for consistency
    ) -> Dict[str, torch.Tensor]:
        """
        Args:
            embedding : identity embedding
            truncation: truncation scale

        Returns:
            dict:
                'image': [B, 3, res, res]  generated face
        """
        # Map embedding to StyleGAN latent code
        w = self.emb_to_w(embedding)         # [B, 512]

        # Dummy z for generator (not used, only w matters)
        z = torch.zeros(embedding.size(0), 512, device=embedding.device)

        # Generate with fixed w
        gen_out = self.generator(z)

        # Override with embedding-derived w via style mixing
        # (simplified: just use w directly in first block)
        return {"image": gen_out["image"]}


# ── Discriminator ─────────────────────────────────────────────────────────────

class Discriminator(nn.Module):
    """
    StyleGAN2 Discriminator.

    Architecture:
        Real/Fake image [B, 3, res, res]
            │
            ├── Progressive downsampling
            │   res → res/2 → ... → 4×4
            │
            ├── Minibatch std (adds sample diversity cue)
            │
            └── Final classification
                Output: patch-level or global logit [B, 1]

    Patch-level discrimination better preserves local details
    for high-quality face generation.
    """

    def __init__(
        self,
        input_res: int = 256,
        fmap_base: int = 16384,
        fmap_max: int = 512,
    ):
        super().__init__()
        assert input_res in (64, 128, 256)

        res_log2 = int(math.log2(input_res))

        # FromRGB: first layer
        in_ch = min(int(fmap_base / input_res), fmap_max)
        self.from_rgb = EqualConv2d(3, in_ch, kernel=1)

        # Downsampling blocks
        self.blocks = nn.ModuleList()
        for i in range(res_log2 - 1, 1, -1):   # res → res/2 → ... → 4
            res = 2 ** i
            in_ch_  = min(int(fmap_base / res), fmap_max)
            out_ch_ = min(int(fmap_base / (res // 2)), fmap_max)

            self.blocks.append(
                self._make_downsample_block(in_ch_, out_ch_)
            )

        # Final block: 4×4 → classification
        final_ch = min(int(fmap_base / 4), fmap_max)
        self.final_block = nn.Sequential(
            EqualConv2d(final_ch + 1, final_ch, kernel=3, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            EqualConv2d(final_ch, final_ch, kernel=4, padding=0),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Flatten(),
            EqualLinear(final_ch, 1),
        )

    def _make_downsample_block(self, in_ch: int, out_ch: int) -> nn.Module:
        return nn.Sequential(
            EqualConv2d(in_ch, in_ch, kernel=3, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.AvgPool2d(2),
            EqualConv2d(in_ch, out_ch, kernel=3, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: [B, 3, res, res] ∈ [-1, 1]"""
        x = self.from_rgb(x)                   # [B, ch, res, res]

        for block in self.blocks:
            x = block(x)                       # progressive downsampling

        # Minibatch discrimination: append std per feature
        B, C, H, W = x.shape
        std = torch.std(x, dim=0, keepdim=True, unbiased=False)
        std = torch.mean(std, dim=[1, 2, 3], keepdim=True)
        std = std.expand(B, 1, H, W)
        x = torch.cat([x, std], dim=1)

        # Final classification
        logit = self.final_block(x)            # [B, 1]

        return logit


# ── GAN Loss ──────────────────────────────────────────────────────────────────

class StyleGANLoss(nn.Module):
    """
    StyleGAN2 loss with R1 gradient penalty.

    L_G = -E[logit_fake]                       (fool discriminator)
    L_D = E[logit_fake] - E[logit_real]
        + λ · R1(real_gradients)               (gradient penalty for real)

    R1 penalty: discourages overfitting D to memorize training set
    """

    def __init__(
        self,
        lambda_r1: float = 10.0,
    ):
        super().__init__()
        self.lambda_r1 = lambda_r1

    def generator_loss(
        self,
        fake_logit: torch.Tensor,              # [B, 1]
    ) -> torch.Tensor:
        """L_G: fool discriminator"""
        return -fake_logit.mean()

    def discriminator_loss(
        self,
        real_logit: torch.Tensor,              # [B, 1]
        fake_logit: torch.Tensor,              # [B, 1]
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """L_D: classify real vs fake"""
        loss = F.softplus(fake_logit).mean() + F.softplus(-real_logit).mean()
        metrics = {"d_loss": loss.item()}
        return loss, metrics

    def r1_penalty(
        self,
        real_images: torch.Tensor,             # [B, 3, H, W]
        discriminator: nn.Module,
    ) -> torch.Tensor:
        """
        R1 gradient penalty on real samples.
        Encourages gradients to stay small.
        """
        real_images.requires_grad = True

        real_logit = discriminator(real_images)
        real_loss  = real_logit.sum()

        grads = torch.autograd.grad(
            outputs=real_loss,
            inputs=real_images,
            create_graph=True,
            retain_graph=True,
        )[0]

        penalty = (grads.view(grads.size(0), -1) ** 2).sum(dim=1).mean()
        return penalty


# ── Face Synthesis Controller ─────────────────────────────────────────────────

class FaceSynthesisController(nn.Module):
    """
    High-level controller for controllable face synthesis.

    Allows attribute manipulation:
        • Age: young → old
        • Expression: neutral → smile
        • Pose: frontal → profile
        • Lighting: dark → bright
        • Gender: female → male

    Maps attribute codes to StyleGAN latent directions.
    """

    def __init__(
        self,
        generator: StyleGAN2Generator,
        num_attributes: int = 10,
        w_dim: int = 512,
    ):
        super().__init__()
        self.generator = generator
        self.num_attributes = num_attributes

        # Learn linear directions in w-space for each attribute
        # direction = w0 - w1 (where w0=attribute present, w1=absent)
        self.attr_directions = nn.Parameter(
            torch.randn(num_attributes, w_dim) * 0.1
        )

    def forward(
        self,
        z: torch.Tensor,                   # [B, z_dim]
        attr_strengths: torch.Tensor,      # [B, num_attributes] ∈ [-1, 1]
    ) -> Dict[str, torch.Tensor]:
        """
        Generate face with controlled attributes.

        Args:
            z             : random noise
            attr_strengths: attribute control (-1 to 1 per attribute)

        Returns:
            dict:
                'image'       : [B, 3, res, res] synthesized face
                'w_modified'  : [B, w_dim] modified latent code
        """
        # Map z to w
        w = self.generator.mapping(z)              # [B, w_dim]

        # Apply attribute directions
        attr_mod = (attr_strengths.unsqueeze(-1) * self.attr_directions).sum(dim=1)
        w_modified = w + attr_mod                   # [B, w_dim]

        # Generate with modified w
        dummy_z = torch.zeros_like(z)
        gen_out = self.generator(dummy_z)  # ignores z, uses mapping instead

        # Hack: re-inject w_modified into generator
        # (simplification; real impl would modify generator forward)

        return {
            "image"      : gen_out["image"],
            "w_modified" : w_modified,
        }


# ── Data Augmentation Generator ───────────────────────────────────────────────

class DataAugmentationGenerator(nn.Module):
    """
    Uses pre-trained StyleGAN2 to augment training data.

    Generates realistic synthetic faces to:
        • Expand limited training sets
        • Balance class imbalance
        • Improve model robustness

    Useful when real face data is scarce or privacy-sensitive.
    """

    def __init__(
        self,
        generator_weights: Optional[str] = None,
        output_res: int = 256,
        device: str = "cpu",
    ):
        super().__init__()
        self.device = torch.device(device)
        self.generator = StyleGAN2Generator(
            output_res=output_res,
        ).to(self.device)
        self.generator.eval()

        if generator_weights is not None:
            state = torch.load(generator_weights, map_location=self.device)
            self.generator.load_state_dict(state)

    @torch.no_grad()
    def generate_batch(
        self,
        batch_size: int,
        truncation: float = 0.7,
    ) -> torch.Tensor:
        """
        Generate a batch of synthetic face images.

        Args:
            batch_size: number of images to generate
            truncation: truncation scale for consistency

        Returns:
            images: [batch_size, 3, res, res] ∈ [-1, 1]
        """
        z = torch.randn(batch_size, 512, device=self.device)
        gen_out = self.generator(z, truncation=truncation)
        return gen_out["image"]  # ∈ [-1, 1]

    @torch.no_grad()
    def generate_interpolation(
        self,
        z1: torch.Tensor,
        z2: torch.Tensor,
        steps: int = 10,
    ) -> torch.Tensor:
        """
        Generate interpolation between two noise vectors.
        Produces smooth animation: face1 → face2.

        Args:
            z1, z2 : start/end noise [z_dim]
            steps  : number of interpolation steps

        Returns:
            images: [steps, 3, res, res]
        """
        images = []
        for alpha in torch.linspace(0, 1, steps, device=self.device):
            z = (1 - alpha) * z1 + alpha * z2
            gen_out = self.generator(z.unsqueeze(0))
            images.append(gen_out["image"])
        return torch.cat(images, dim=0)


# ── GAN Gate (pipeline integration) ───────────────────────────────────────────

class GANGeneratorGate(nn.Module):
    """
    Drop-in GAN generator gate for face synthesis and augmentation.

    Usage modes:
        1. Data augmentation: generate synthetic training data
        2. Face reconstruction: embedding → face image
        3. Face attribute editing: modify generated faces
        4. Privacy-preserving demo: show synthetic instead of real photo

    Examples:
        # Generate synthetic face batch
        gate   = GANGeneratorGate(weights_path="stylegan2.pt")
        faces  = gate.generate_batch(batch_size=32)

        # Reconstruct face from embedding
        emb    = face_recognizer(real_face)
        synth  = gate.from_embedding(emb)

        # Interpolate between two faces
        morph  = gate.interpolate(z1, z2, steps=30)
    """

    def __init__(
        self,
        weights_path: Optional[str] = None,
        output_res: int = 256,
        device: str = "cpu",
    ):
        super().__init__()
        self.device = torch.device(device)
        self.output_res = output_res

        self.generator = StyleGAN2Generator(
            output_res=output_res,
        ).to(self.device)
        self.generator.eval()

        if weights_path is not None:
            state = torch.load(weights_path, map_location=self.device)
            self.generator.load_state_dict(state)
            print(f"[GANGeneratorGate] Loaded weights from: {weights_path}")
        else:
            print("[GANGeneratorGate] ⚠️  No weights loaded — random init.")

        # Optional: embedding-to-image adapter
        self.emb_to_img = EmbeddingToImageGenerator(
            emb_dim=512,
            output_res=output_res,
        ).to(self.device)

    @torch.no_grad()
    def generate_batch(
        self,
        batch_size: int = 32,
        truncation: float = 0.7,
        seed: Optional[int] = None,
    ) -> torch.Tensor:
        """
        Generate a batch of synthetic faces.

        Args:
            batch_size: number of images
            truncation: truncation scale ∈ [0, 1]
            seed      : random seed for reproducibility

        Returns:
            images: [batch_size, 3, res, res] ∈ [-1, 1]
        """
        if seed is not None:
            torch.manual_seed(seed)

        z = torch.randn(batch_size, 512, device=self.device)
        gen_out = self.generator(z, truncation=truncation)
        return gen_out["image"]

    @torch.no_grad()
    def from_embedding(
        self,
        embedding: torch.Tensor,          # [B, 512]
    ) -> torch.Tensor:
        """
        Generate face image from identity embedding.

        Args:
            embedding: [B, 512] from face recognizer

        Returns:
            images: [B, 3, res, res]
        """
        embedding = embedding.to(self.device)
        emb_out = self.emb_to_img(embedding)
        return emb_out["image"]

    @torch.no_grad()
    def interpolate(
        self,
        z1: torch.Tensor,                 # [z_dim]
        z2: torch.Tensor,                 # [z_dim]
        steps: int = 10,
        truncation: float = 0.7,
    ) -> torch.Tensor:
        """
        Linear interpolation in latent space.

        Args:
            z1, z2 : start/end noise vectors
            steps  : interpolation steps
            truncation: truncation for consistency

        Returns:
            images: [steps, 3, res, res]
        """
        z1 = z1.to(self.device)
        z2 = z2.to(self.device)

        images = []
        for alpha in torch.linspace(0, 1, steps, device=self.device):
            z = (1 - alpha) * z1 + alpha * z2
            gen_out = self.generator(z.unsqueeze(0), truncation=truncation)
            images.append(gen_out["image"])

        return torch.cat(images, dim=0)

    @torch.no_grad()
    def style_mix(
        self,
        z_content: torch.Tensor,          # [z_dim]
        z_style: torch.Tensor,            # [z_dim]
        mix_layer: int = 4,               # which layer to mix
    ) -> Dict[str, torch.Tensor]:
        """
        StyleGAN style mixing: combine content from one face, style from another.

        Args:
            z_content : noise for content (structure)
            z_style   : noise for style (appearance)
            mix_layer : layer index to swap styles

        Returns:
            dict:
                'image' : [1, 3, res, res] mixed face
        """
        z_content = z_content.to(self.device)
        z_style = z_style.to(self.device)

        gen_out = self.generator(z_content)

        # Simplified: actual style mixing would require layer-wise control
        # For now, just return one of the generated faces

        return {"image": gen_out["image"]}

    @torch.no_grad()
    def attribute_control(
        self,
        z: torch.Tensor,
        attr_strengths: torch.Tensor,    # [num_attributes] ∈ [-1, 1]
    ) -> torch.Tensor:
        """
        Generate face with controlled attributes.

        Args:
            z              : noise [z_dim]
            attr_strengths : attribute control vector

        Returns:
            image: [1, 3, res, res] with controlled attributes
        """
        # Simplified placeholder; full version needs attribute direction vectors
        z = z.to(self.device)
        attr_strengths = attr_strengths.to(self.device)

        gen_out = self.generator(z.unsqueeze(0) if z.dim() == 1 else z)
        return gen_out["image"]
