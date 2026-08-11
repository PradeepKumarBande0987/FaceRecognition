from .discriminator import (
    BaseDiscriminator,             # base discriminator
    ProgressiveDiscriminator,      # progressive training discriminator
    MultiScaleDiscriminator,       # multi-scale pyramid discriminator
    PatchDiscriminator,            # patch-based discriminator
    DiscriminatorLoss,             # advanced loss functions
    GANDiscriminatorGate,          # drop-in inference gate
    SNConv2d,                      # spectral norm conv
    SNLinear,                      # spectral norm linear
    SpectralNorm,                  # spectral normalization
    ResidualBlock,                 # residual block
    SelfAttentionBlock,            # self-attention block
    MinibatchStd,                  # minibatch std layer
)

__all__ = [
    # ... existing exports ...

    # GAN discriminator
    "BaseDiscriminator",
    "ProgressiveDiscriminator",
    "MultiScaleDiscriminator",
    "PatchDiscriminator",
    "DiscriminatorLoss",
    "GANDiscriminatorGate",
    "SNConv2d",
    "SNLinear",
    "SpectralNorm",
    "ResidualBlock",
    "SelfAttentionBlock",
    "MinibatchStd",
]