from .fusion.fusion import (
    MultiModalFusion,              # full model        (training)
    MultiModalFusionGate,          # drop-in gate      (inference)
    MultiModalFusionLoss,          # combined loss     (training)
    RGBEncoder,                    # standalone RGB encoder
    DepthEncoder,                  # standalone Depth encoder
    IREncoder,                     # standalone IR encoder
    ThermalEncoder,                # standalone Thermal encoder
    AudioEncoder,                  # standalone Audio encoder
    LipEncoder,                    # standalone Lip encoder
    CrossModalAttention,           # cross-modal attention block
    AdaptiveFusionGate,            # adaptive weight gate
    ModalityTokenTransformer,      # transformer fusion encoder
    AudioVisualSynchronyModule,    # AV sync detector
    IdentityEmbeddingHead,         # L2 identity embedding
    ArcFaceHead,                   # ArcFace classification head
)

from .gait.gait import (
    GaitRecognition,               # full model        (training)
    GaitGate,                      # drop-in gate      (inference)
    GaitRecognitionLoss,           # combined loss     (training)
    SilhouetteEncoder,             # standalone silhouette encoder
    SkeletonEncoder,               # standalone skeleton encoder
    OpticalFlowGaitEncoder,        # standalone flow encoder
    GaitCycleAnalyzer,             # gait cycle detector
    FaceGaitVerification,          # face-gait fusion verification
)
from .voice.voice import (
    VoiceRecognition,              
    VoiceGate,                    
    VoiceActivityDetector,         
    SpeakerEncoder,                
    ResNetSpeakerEncoder,          
    SpeechTransformerEncoder,      
    PhoneticEncoder,               
    VoiceRecognitionLoss,         
    VoiceFaceVerification,         
)

__all__ = [
 # multi-modal fusion
    "MultiModalFusion",
    "MultiModalFusionGate",
    "MultiModalFusionLoss",
    "RGBEncoder",
    "DepthEncoder",
    "IREncoder",
    "ThermalEncoder",
    "AudioEncoder",
    "LipEncoder",
    "CrossModalAttention",
    "AdaptiveFusionGate",
    "ModalityTokenTransformer",
    "AudioVisualSynchronyModule",
    "IdentityEmbeddingHead",
    "ArcFaceHead",

    # gait recognition
    "GaitRecognition",
    "GaitGate",
    "GaitRecognitionLoss",
    "SilhouetteEncoder",
    "SkeletonEncoder",
    "OpticalFlowGaitEncoder",
    "GaitCycleAnalyzer",
    "FaceGaitVerification",

    # voice recognition
    "VoiceActivityDetector",
    "SpeakerEncoder",
    "ResNetSpeakerEncoder",
    "SpeechTransformerEncoder",
    "PhoneticEncoder",
    "VoiceRecognition",
    "VoiceRecognitionLoss",
    "VoiceGate",
    "VoiceFaceVerification",
]
