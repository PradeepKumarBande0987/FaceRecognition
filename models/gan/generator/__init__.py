from .generator import (
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
]