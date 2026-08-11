"""Fairness and bias mitigation for face recognition"""

from .bias_mitigation import (
    Demographics,
    DemographicBalancedSampler,
    FairnessMetrics,
    DemographicParityLoss,
    EqualizedOddsLoss,
    AdversarialDebiasing,
    FairnessCalibrator,
    FairnessAwareTrainer,
    FairnessAudit,
    FairnessGate,
    BiasAnalyzer,
)

from .demographic_classifier import (
    AgeEstimator,
    GenderClassifier,
    RaceEthnicityClassifier,
    SkinToneClassifier,
    MultiTaskDemographicClassifier,
    MultiTaskDemographicLoss,
    DemographicClassifierGate,
    DemographicAccuracyAnalyzer,
)

__all__ = [
    # Bias mitigation
    "Demographics",
    "DemographicBalancedSampler",
    "FairnessMetrics",
    "DemographicParityLoss",
    "EqualizedOddsLoss",
    "AdversarialDebiasing",
    "FairnessCalibrator",
    "FairnessAwareTrainer",
    "FairnessAudit",
    "FairnessGate",
    "BiasAnalyzer",
    
    # Demographic classification
    "AgeEstimator",
    "GenderClassifier",
    "RaceEthnicityClassifier",
    "SkinToneClassifier",
    "MultiTaskDemographicClassifier",
    "MultiTaskDemographicLoss",
    "DemographicClassifierGate",
    "DemographicAccuracyAnalyzer",
]
