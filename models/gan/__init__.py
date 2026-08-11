from .generator.generator import (
    StyleGAN2Generator,            # full StyleGAN2 generator
    GANGeneratorGate,              # drop-in inference gate
    Discriminator,                 # StyleGAN2 discriminator
    StyleGANLoss,                  # GAN loss (R1 penalty)
    EmbeddingToImageGenerator,     # embedding → image
    DataAugmentationGenerator,     # synthetic data generation
    FaceSynthesisController,       # attribute-controlled synthesis
    MappingNetwork,                # latent space mapper
    StyleGANBlock,                 # core synthesis block
    ModulatedConv2d,               # modulated convolution
    AdaIN,                         # adaptive instance norm
    EqualConv2d,                   # equalized learning rate conv
    EqualLinear,                   # equalized learning rate linear
)

from .discriminator.discriminator import (
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

    # GAN generator
    "StyleGAN2Generator",
    "GANGeneratorGate",
    "Discriminator",
    "StyleGANLoss",
    "EmbeddingToImageGenerator",
    "DataAugmentationGenerator",
    "FaceSynthesisController",
    "MappingNetwork",
    "StyleGANBlock",
    "ModulatedConv2d",
    "AdaIN",
    "EqualConv2d",
    "EqualLinear",
    
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
