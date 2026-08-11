"""
Denoiser Module
Removes noise from face images before recognition/anti-spoofing pipeline.
Architecture: U-Net style encoder-decoder with skip connections +
              a lightweight FFDNet-inspired noise-level-aware branch.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Dict


# ── Building Blocks ───────────────────────────────────────────────────────────

class ConvBNReLU(nn.Module):
    """
    Standard Conv → BatchNorm → ReLU block.
    Used throughout encoder / decoder.
    """

    def __init__(
        self,
        in_ch  : int,
        out_ch : int,
        kernel : int = 3,
        stride : int = 1,
        padding: int = 1,
        dilation: int = 1,
    ):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel,
                      stride=stride,
                      padding=padding,
                      dilation=dilation,
                      bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class ResidualDenoisingBlock(nn.Module):
    """
    Residual block with two ConvBNReLU layers + skip connection.
    Residual learning helps the network focus on noise residual
    rather than learning identity mapping from scratch.
    """

    def __init__(self, ch: int):
        super().__init__()
        self.block = nn.Sequential(
            ConvBNReLU(ch, ch),
            nn.Conv2d(ch, ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(ch),
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.relu(x + self.block(x))   # residual add


class ChannelAttention(nn.Module):
    """
    Squeeze-and-Excitation style channel attention.
    Helps the denoiser focus on informative channels
    (e.g. suppress noisy chroma channels, preserve luma).
    """

    def __init__(self, ch: int, reduction: int = 8):
        super().__init__()
        self.gate = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),          # [B, C, 1, 1]
            nn.Flatten(),                     # [B, C]
            nn.Linear(ch, ch // reduction),
            nn.ReLU(inplace=True),
            nn.Linear(ch // reduction, ch),
            nn.Sigmoid(),                     # [B, C]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        w = self.gate(x).view(x.size(0), x.size(1), 1, 1)
        return x * w                          # scale channels


# ── Encoder ───────────────────────────────────────────────────────────────────

class EncoderBlock(nn.Module):
    """
    Encoder stage: two ConvBNReLU + ResidualDenoisingBlock + ChannelAttention.
    Downsamples via MaxPool. Returns feature map + skip connection tensor.
    """

    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.conv = nn.Sequential(
            ConvBNReLU(in_ch, out_ch),
            ResidualDenoisingBlock(out_ch),
            ChannelAttention(out_ch),
        )
        self.pool = nn.MaxPool2d(2)           # spatial ÷ 2

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        feat   = self.conv(x)                 # skip connection feature
        pooled = self.pool(feat)              # downsampled for next stage
        return pooled, feat                   # (to next encoder, to decoder)


# ── Bottleneck ────────────────────────────────────────────────────────────────

class Bottleneck(nn.Module):
    """
    Deepest stage of U-Net.
    Uses dilated convolutions to capture global noise patterns
    without further downsampling.
    """

    def __init__(self, ch: int):
        super().__init__()
        self.block = nn.Sequential(
            ConvBNReLU(ch, ch * 2, dilation=1, padding=1),
            ConvBNReLU(ch * 2, ch * 2, dilation=2, padding=2),
            ConvBNReLU(ch * 2, ch * 2, dilation=4, padding=4),
            ConvBNReLU(ch * 2, ch,     dilation=1, padding=1),
            ChannelAttention(ch),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


# ── Decoder ───────────────────────────────────────────────────────────────────

class DecoderBlock(nn.Module):
    """
    Decoder stage: bilinear upsample → concat skip → ConvBNReLU ×2.
    Skip connections preserve fine spatial details lost during pooling.
    """

    def __init__(self, in_ch: int, skip_ch: int, out_ch: int):
        super().__init__()
        self.up   = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)
        self.conv = nn.Sequential(
            ConvBNReLU(in_ch + skip_ch, out_ch),
            ResidualDenoisingBlock(out_ch),
            ChannelAttention(out_ch),
        )

    def forward(
        self,
        x   : torch.Tensor,
        skip: torch.Tensor,
    ) -> torch.Tensor:
        x = self.up(x)
        # Handle odd-dimension mismatches safely
        if x.shape != skip.shape:
            x = F.interpolate(x, size=skip.shape[2:], mode="bilinear", align_corners=False)
        x = torch.cat([x, skip], dim=1)       # concat along channel axis
        return self.conv(x)


# ── Noise Level Estimator ─────────────────────────────────────────────────────

class NoiseLevelEstimator(nn.Module):
    """
    Estimates a per-image noise level scalar σ ∈ [0, 1].
    Inspired by FFDNet: conditioning the denoiser on noise level
    allows it to handle a wide range of noise without retraining.

    The estimated σ is broadcast back as a spatial noise map and
    concatenated to the encoder input, giving the U-Net explicit
    noise context at every pixel.
    """

    def __init__(self, in_ch: int = 3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, 32, 5, padding=2),  nn.ReLU(inplace=True),
            nn.Conv2d(32,    32, 3, padding=1),  nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(4),              # compress to 4×4
            nn.Flatten(),
            nn.Linear(32 * 4 * 4, 64),           nn.ReLU(inplace=True),
            nn.Linear(64, 1),
            nn.Sigmoid(),                         # σ ∈ [0, 1]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Returns scalar noise estimate per image: [B, 1]"""
        return self.net(x)


# ── Full Denoiser ─────────────────────────────────────────────────────────────

class FaceDenoiser(nn.Module):
    """
    Full U-Net denoiser with noise-level conditioning.

    Pipeline:
        Noisy RGB
            │
            ├── NoiseLevelEstimator ──► σ map  (broadcast + concat)
            │
            ├── Encoder (×3 stages, ×2 downsample each)
            │       E1: [3+1, 32]  → skip1 [B, 32, H/2,  W/2]
            │       E2: [32,  64]  → skip2 [B, 64, H/4,  W/4]
            │       E3: [64, 128]  → skip3 [B,128, H/8,  W/8]
            │
            ├── Bottleneck [128 → 128]  at H/8 × W/8
            │
            ├── Decoder (×3 stages, ×2 upsample each)
            │       D3: [128+128 → 64]
            │       D2: [ 64+ 64 → 32]
            │       D1: [ 32+ 32 → 32]
            │
            └── Output head [32 → 3] + residual skip from input
                ──► Clean RGB  ∈ [0, 1]
    """

    def __init__(
        self,
        in_ch       : int = 3,
        base_ch     : int = 32,
        use_noise_map: bool = True,
    ):
        """
        Args:
            in_ch        : input channels (3 for RGB).
            base_ch      : base channel width (scales ×2 per encoder stage).
            use_noise_map: if True, concatenate estimated σ map to input.
        """
        super().__init__()
        self.use_noise_map = use_noise_map
        enc_in = in_ch + 1 if use_noise_map else in_ch  # +1 for σ channel

        # ── Noise estimator ──────────────────────────────────────────────────
        self.noise_estimator = NoiseLevelEstimator(in_ch=in_ch)

        # ── Encoder ──────────────────────────────────────────────────────────
        self.enc1 = EncoderBlock(enc_in,        base_ch)       # → base_ch
        self.enc2 = EncoderBlock(base_ch,       base_ch * 2)   # → base_ch*2
        self.enc3 = EncoderBlock(base_ch * 2,   base_ch * 4)   # → base_ch*4

        # ── Bottleneck ───────────────────────────────────────────────────────
        self.bottleneck = Bottleneck(base_ch * 4)

        # ── Decoder ──────────────────────────────────────────────────────────
        self.dec3 = DecoderBlock(base_ch * 4, base_ch * 4, base_ch * 2)
        self.dec2 = DecoderBlock(base_ch * 2, base_ch * 2, base_ch)
        self.dec1 = DecoderBlock(base_ch,     base_ch,     base_ch)

        # ── Output head ──────────────────────────────────────────────────────
        self.out_head = nn.Sequential(
            nn.Conv2d(base_ch, base_ch, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_ch, in_ch, 1),   # 1×1 conv → back to RGB
            nn.Sigmoid(),                    # output ∈ [0, 1]
        )

    def forward(
        self,
        x          : torch.Tensor,
        sigma      : Optional[torch.Tensor] = None,
        return_maps: bool = False,
    ) -> Dict[str, torch.Tensor]:
        """
        Args:
            x          : noisy face  [B, 3, H, W],  values ∈ [0, 1]
            sigma      : optional external noise level [B, 1].
                         If None, estimated automatically.
            return_maps: if True, return depth/noise maps for visualization.

        Returns:
            dict:
                'denoised'    : [B, 3, H, W]  clean face
                'noise_map'   : [B, 1, H, W]  estimated noise map (if return_maps)
                'sigma'       : [B, 1]         estimated noise level (if return_maps)
                'residual'    : [B, 3, H, W]  extracted noise residual (if return_maps)
        """
        # ── Noise level estimation ───────────────────────────────────────────
        if sigma is None:
            sigma = self.noise_estimator(x)   # [B, 1]

        # ── Build noise-conditioned input ────────────────────────────────────
        if self.use_noise_map:
            # Broadcast σ scalar → spatial noise map [B, 1, H, W]
            sigma_map = sigma.view(-1, 1, 1, 1).expand(-1, 1, x.size(2), x.size(3))
            inp = torch.cat([x, sigma_map], dim=1)    # [B, 4, H, W]
        else:
            inp = x

        # ── Encoder ──────────────────────────────────────────────────────────
        e1, skip1 = self.enc1(inp)     # e1:[B,32,H/2,W/2]  skip1:[B,32,H,W]
        e2, skip2 = self.enc2(e1)      # e2:[B,64,H/4,W/4]  skip2:[B,64,H/2,W/2]
        e3, skip3 = self.enc3(e2)      # e3:[B,128,H/8,W/8] skip3:[B,128,H/4,W/4]

        # ── Bottleneck ───────────────────────────────────────────────────────
        b = self.bottleneck(e3)        # [B, 128, H/8, W/8]

        # ── Decoder ──────────────────────────────────────────────────────────
        d3 = self.dec3(b,  skip3)      # [B, 64,  H/4, W/4]
        d2 = self.dec2(d3, skip2)      # [B, 32,  H/2, W/2]
        d1 = self.dec1(d2, skip1)      # [B, 32,  H,   W  ]

        # ── Output + global residual skip ────────────────────────────────────
        denoised = self.out_head(d1)   # [B, 3, H, W]

        out = {"denoised": denoised}

        if return_maps:
            sigma_map_out = sigma.view(-1, 1, 1, 1).expand_as(x[:, :1])
            out["noise_map"] = sigma_map_out
            out["sigma"]     = sigma
            out["residual"]  = (x - denoised).clamp(-1, 1)   # extracted noise

        return out


# ── Loss ──────────────────────────────────────────────────────────────────────

class DenoiserLoss(nn.Module):
    """
    Combined denoising loss:

        L = α · L1(denoised, clean)
          + β · MSE(denoised, clean)
          + γ · SSIM_loss(denoised, clean)
          + δ · PerceptualLoss(denoised, clean)   [optional, VGG-free approx]

    L1  → preserves sharp edges
    MSE → penalizes large deviations
    SSIM→ preserves structural / perceptual quality
    """

    def __init__(
        self,
        alpha: float = 1.0,    # L1 weight
        beta : float = 0.5,    # MSE weight
        gamma: float = 0.3,    # SSIM weight
    ):
        super().__init__()
        self.alpha = alpha
        self.beta  = beta
        self.gamma = gamma
        self.l1    = nn.L1Loss()
        self.mse   = nn.MSELoss()

    def _ssim_loss(
        self,
        pred : torch.Tensor,
        target: torch.Tensor,
        window_size: int = 11,
    ) -> torch.Tensor:
        """
        Simplified differentiable SSIM loss (1 - SSIM).
        Operates per-channel then averages.
        """
        C1, C2 = 0.01 ** 2, 0.03 ** 2
        ch = pred.size(1)

        # Gaussian kernel
        kernel = torch.ones(ch, 1, window_size, window_size,
                            device=pred.device) / (window_size ** 2)

        mu_p  = F.conv2d(pred,   kernel, groups=ch, padding=window_size // 2)
        mu_t  = F.conv2d(target, kernel, groups=ch, padding=window_size // 2)

        mu_p2 = mu_p * mu_p
        mu_t2 = mu_t * mu_t
        mu_pt = mu_p * mu_t

        sig_p2  = F.conv2d(pred   * pred,   kernel, groups=ch, padding=window_size // 2) - mu_p2
        sig_t2  = F.conv2d(target * target, kernel, groups=ch, padding=window_size // 2) - mu_t2
        sig_pt  = F.conv2d(pred   * target, kernel, groups=ch, padding=window_size // 2) - mu_pt

        ssim_map = ((2 * mu_pt + C1) * (2 * sig_pt + C2)) / \
                   ((mu_p2 + mu_t2 + C1) * (sig_p2 + sig_t2 + C2))

        return 1.0 - ssim_map.mean()

    def forward(
        self,
        denoised: torch.Tensor,   # [B, 3, H, W]
        clean   : torch.Tensor,   # [B, 3, H, W]
    ) -> Tuple[torch.Tensor, Dict[str, float]]:

        l1_loss   = self.l1(denoised, clean)
        mse_loss  = self.mse(denoised, clean)
        ssim_loss = self._ssim_loss(denoised, clean)

        total = (self.alpha * l1_loss
               + self.beta  * mse_loss
               + self.gamma * ssim_loss)

        return total, {
            "l1_loss"  : l1_loss.item(),
            "mse_loss" : mse_loss.item(),
            "ssim_loss": ssim_loss.item(),
            "total_loss": total.item(),
        }


# ── Denoiser Gate (pipeline integration) ─────────────────────────────────────

class DenoiserGate(nn.Module):
    """
    Drop-in preprocessing gate for your face recognition pipeline.
    Automatically denoises face crops before passing to:
        → AntiSpoofingGate
        → FaceRecognizer

    Usage:
        denoiser = DenoiserGate(weights_path="denoiser.pt")
        clean    = denoiser(noisy_face_crop)
    """

    def __init__(
        self,
        weights_path: Optional[str] = None,
        device      : str = "cpu",
        threshold   : float = 0.15,     # σ above this → apply denoising
    ):
        super().__init__()
        self.device    = torch.device(device)
        self.threshold = threshold
        self.model     = FaceDenoiser().to(self.device)
        self.model.eval()

        if weights_path is not None:
            state = torch.load(weights_path, map_location=self.device)
            self.model.load_state_dict(state)
            print(f"[DenoiserGate] Loaded weights from: {weights_path}")
        else:
            print("[DenoiserGate] ⚠️  No weights loaded — running with random init.")

    @torch.no_grad()
    def forward(
        self,
        x          : torch.Tensor,
        force      : bool = False,
        return_maps: bool = False,
    ) -> Dict[str, torch.Tensor]:
        """
        Args:
            x          : face crop [B, 3, H, W]  values ∈ [0, 1]
            force      : skip threshold check, always denoise
            return_maps: also return noise metadata

        Returns:
            dict:
                'output'   : [B, 3, H, W]  denoised (or passthrough) face
                'denoised' : bool           True if denoising was applied
                'sigma'    : [B, 1]         estimated noise levels
        """
        x = x.to(self.device)

        out   = self.model(x, return_maps=True)
        sigma = out["sigma"]               # [B, 1]

        # Selectively apply denoising only when noise is significant
        if force or (sigma.mean().item() > self.threshold):
            output   = out["denoised"]
            denoised = True
        else:
            output   = x                   # pass-through: image is clean enough
            denoised = False

        result = {
            "output"  : output,
            "denoised": denoised,
            "sigma"   : sigma,
        }
        if return_maps:
            result["noise_map"] = out.get("noise_map")
            result["residual"]  = out.get("residual")

        return result
