"""
Demographic Classifier Module
Predicts age, gender, race, ethnicity, skin tone from face images.
Auxiliary task that:
    • Improves feature learning via multi-task training
    • Enables fairness analysis and bias detection
    • Supports demographic-aware filtering/segmentation
    • Facilitates intersectional bias audits
Architectures:
    • Age regression/classification (continuous → age groups)
    • Gender classification (binary/multi-class)
    • Race/Ethnicity classification (multi-class)
    • Skin tone classification (Fitzpatrick scale)
    • Multi-task joint learning
    • Auxiliary classifiers on shared embeddings
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
        in_ch: int,
        out_ch: int,
        kernel: int = 3,
        stride: int = 1,
        padding: int = 1,
        use_bn: bool = True,
    ):
        super().__init__()
        layers = [
            nn.Conv2d(in_ch, out_ch, kernel, stride=stride, 
                     padding=padding, bias=not use_bn),
        ]
        if use_bn:
            layers.append(nn.BatchNorm2d(out_ch))
        layers.append(nn.LeakyReLU(0.2, inplace=True))
        self.block = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class ResidualBlock(nn.Module):
    """Residual block for demographic features."""

    def __init__(self, ch: int):
        super().__init__()
        self.block = nn.Sequential(
            ConvBNReLU(ch, ch),
            nn.Conv2d(ch, ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(ch),
        )
        self.relu = nn.LeakyReLU(0.2, inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.relu(x + self.block(x))


class SEBlock(nn.Module):
    """Squeeze-and-Excitation block for channel attention."""

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


# ── Age Estimator ─────────────────────────────────────────────────────────────

class AgeEstimator(nn.Module):
    """
    Age estimator: predicts age from face image.

    Approaches:
        1. Regression: predict exact age (continuous) [0, 100]
        2. Classification: predict age group (discrete) [0-20, 21-30, ...]
        3. Ordinal: classification with ordinal constraint (age groups are ordered)

    Age-related features:
        • Skin texture & wrinkles (increasing with age)
        • Hair color & graying patterns
        • Face shape changes (sagging, loss of elasticity)
        • Eye appearance (age spots, crow's feet)

    Multi-task: jointly learn age + identity for better features
    """

    def __init__(
        self,
        feat_dim: int = 256,
        age_type: str = "classification",  # "regression", "classification", "ordinal"
        num_age_groups: int = 6,           # for classification
        use_se_block: bool = True,
    ):
        super().__init__()
        self.age_type = age_type
        self.num_age_groups = num_age_groups
        self.feat_dim = feat_dim

        # Backbone: shared convolutional encoder
        self.encoder = nn.Sequential(
            ConvBNReLU(3, 32),
            ResidualBlock(32),
            nn.MaxPool2d(2),                  # H/2, W/2

            ConvBNReLU(32, 64),
            ResidualBlock(64),
            if_true(use_se_block, SEBlock(64)),
            nn.MaxPool2d(2),                  # H/4, W/4

            ConvBNReLU(64, 128),
            ResidualBlock(128),
            if_true(use_se_block, SEBlock(128)),
            nn.MaxPool2d(2),                  # H/8, W/8

            ConvBNReLU(128, 256),
            ResidualBlock(256),
            nn.AdaptiveAvgPool2d(1),         # [B, 256, 1, 1]
        )

        self.feat_proj = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256, feat_dim),
            nn.BatchNorm1d(feat_dim),
            nn.GELU(),
        )

        # Age prediction head
        if age_type == "regression":
            # Predict continuous age [0, 100]
            self.age_head = nn.Sequential(
                nn.Linear(feat_dim, 128),
                nn.ReLU(inplace=True),
                nn.Dropout(0.2),
                nn.Linear(128, 1),             # single continuous output
            )

        elif age_type == "classification":
            # Predict age group (softmax)
            self.age_head = nn.Sequential(
                nn.Linear(feat_dim, 128),
                nn.ReLU(inplace=True),
                nn.Dropout(0.2),
                nn.Linear(128, num_age_groups),  # logits per group
            )

        elif age_type == "ordinal":
            # Ordinal regression: predict if age > threshold for each threshold
            # More nuanced than classification, less restrictive than regression
            self.age_head = nn.Sequential(
                nn.Linear(feat_dim, 128),
                nn.ReLU(inplace=True),
                nn.Dropout(0.2),
                nn.Linear(128, num_age_groups),  # binary outputs for each threshold
            )

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Args:
            x: [B, 3, H, W] face image

        Returns:
            dict:
                'feat'     : [B, feat_dim] learned features
                'age_pred' : [B] or [B, num_groups] age prediction
                'logit'    : raw logits if classification
        """
        feat = self.encoder(x)
        feat = self.feat_proj(feat)

        age_logit = self.age_head(feat)

        if self.age_type == "regression":
            age_pred = age_logit.squeeze(-1)  # [B]
        elif self.age_type == "classification":
            age_pred = age_logit.argmax(dim=1)  # [B]
        elif self.age_type == "ordinal":
            age_pred = age_logit.sigmoid()  # [B, num_groups]

        return {
            "feat"    : feat,
            "age_pred": age_pred,
            "logit"   : age_logit,
        }


# ── Gender Classifier ─────────────────────────────────────────────────────────

class GenderClassifier(nn.Module):
    """
    Gender classifier: predicts gender from face.

    Approaches:
        1. Binary: Male/Female
        2. Multi-class: Male/Female/Non-binary
        3. Soft labels: probability distribution (more realistic)

    Gender-related features:
        • Jaw structure (males: wider, more angular)
        • Facial hair patterns (beard, mustache)
        • Cheekbone prominence
        • Lip fullness
        • Nose structure
        • Brow ridge

    Note: Binary classification can reinforce gender stereotypes.
    Consider using soft/non-binary labels when possible.
    """

    def __init__(
        self,
        feat_dim: int = 256,
        num_classes: int = 2,  # 2 for binary, 3+ for multi-class
        use_se_block: bool = True,
    ):
        super().__init__()
        self.feat_dim = feat_dim
        self.num_classes = num_classes

        # Backbone
        self.encoder = nn.Sequential(
            ConvBNReLU(3, 32),
            ResidualBlock(32),
            nn.MaxPool2d(2),

            ConvBNReLU(32, 64),
            ResidualBlock(64),
            if_true(use_se_block, SEBlock(64)),
            nn.MaxPool2d(2),

            ConvBNReLU(64, 128),
            ResidualBlock(128),
            if_true(use_se_block, SEBlock(128)),
            nn.MaxPool2d(2),

            ConvBNReLU(128, 256),
            nn.AdaptiveAvgPool2d(1),
        )

        self.feat_proj = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256, feat_dim),
            nn.BatchNorm1d(feat_dim),
            nn.GELU(),
        )

        # Gender prediction head
        self.gender_head = nn.Sequential(
            nn.Linear(feat_dim, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(128, num_classes),
        )

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Args:
            x: [B, 3, H, W]

        Returns:
            dict:
                'feat'     : [B, feat_dim]
                'gender_pred': [B]  predicted class
                'logit'    : [B, num_classes]
                'prob'     : [B, num_classes]  softmax probabilities
        """
        feat = self.encoder(x)
        feat = self.feat_proj(feat)

        logit = self.gender_head(feat)
        prob = F.softmax(logit, dim=1)
        gender_pred = logit.argmax(dim=1)

        return {
            "feat"        : feat,
            "gender_pred" : gender_pred,
            "logit"       : logit,
            "prob"        : prob,
        }


# ── Race/Ethnicity Classifier ─────────────────────────────────────────────────

class RaceEthnicityClassifier(nn.Module):
    """
    Race/Ethnicity classifier.

    Categories (example; can be customized):
        • White/Caucasian
        • Black/African
        • Asian (East, South, Southeast)
        • Hispanic/Latino
        • Middle Eastern
        • Indian/South Asian
        • Other/Mixed

    Challenges:
        • Race is a social construct, not biological
        • Phenotypic variation within groups > between groups
        • Risk of reinforcing stereotypes
        • Outdated/offensive terminology in some datasets

    Best practice: Use broader regional categories when possible,
    allow "Mixed" or "Other", provide opt-out options.

    Race-related facial features (problematic but used in ML):
        • Skin tone
        • Nasal structure
        • Lip shape & fullness
        • Brow ridge
        • Jaw structure
    """

    def __init__(
        self,
        feat_dim: int = 256,
        num_classes: int = 7,  # number of race/ethnicity categories
        use_se_block: bool = True,
    ):
        super().__init__()
        self.feat_dim = feat_dim
        self.num_classes = num_classes

        # Backbone
        self.encoder = nn.Sequential(
            ConvBNReLU(3, 32),
            ResidualBlock(32),
            nn.MaxPool2d(2),

            ConvBNReLU(32, 64),
            ResidualBlock(64),
            if_true(use_se_block, SEBlock(64)),
            nn.MaxPool2d(2),

            ConvBNReLU(64, 128),
            ResidualBlock(128),
            if_true(use_se_block, SEBlock(128)),
            nn.MaxPool2d(2),

            ConvBNReLU(128, 256),
            ResidualBlock(256),
            nn.AdaptiveAvgPool2d(1),
        )

        self.feat_proj = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256, feat_dim),
            nn.BatchNorm1d(feat_dim),
            nn.GELU(),
        )

        # Race/ethnicity head
        self.race_head = nn.Sequential(
            nn.Linear(feat_dim, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(128, num_classes),
        )

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Args:
            x: [B, 3, H, W]

        Returns:
            dict:
                'feat'     : [B, feat_dim]
                'race_pred': [B]
                'logit'    : [B, num_classes]
                'prob'     : [B, num_classes]
        """
        feat = self.encoder(x)
        feat = self.feat_proj(feat)

        logit = self.race_head(feat)
        prob = F.softmax(logit, dim=1)
        race_pred = logit.argmax(dim=1)

        return {
            "feat"     : feat,
            "race_pred": race_pred,
            "logit"    : logit,
            "prob"     : prob,
        }


# ── Skin Tone Classifier ──────────────────────────────────────────────────────

class SkinToneClassifier(nn.Module):
    """
    Skin tone classifier using Fitzpatrick scale.

    Fitzpatrick scale (1-6):
        1-2: Very light (pale, easily burns)
        3-4: Medium (olive, moderate sun sensitivity)
        5-6: Dark (brown to very dark, minimal sun sensitivity)

    Use cases:
        • Fairness audit: check if model works equally well across skin tones
        • Sunscreen recommendations (dermatology)
        • Photography lighting adjustment
        • Medical imaging (some conditions visible differently by skin tone)

    Note: Can be grouped as:
        • Continuous (1-6 regression)
        • 3-group (light, medium, dark)
        • 2-group (light vs dark)

    Important: Skin tone ≠ race/ethnicity, but often correlated.
    Use as separate dimension for more nuanced fairness analysis.
    """

    def __init__(
        self,
        feat_dim: int = 256,
        num_tones: int = 6,  # 6 for Fitzpatrick, or 3 for grouped
        use_se_block: bool = True,
    ):
        super().__init__()
        self.feat_dim = feat_dim
        self.num_tones = num_tones

        # Backbone: focus on color information
        self.encoder = nn.Sequential(
            ConvBNReLU(3, 32),
            ResidualBlock(32),
            nn.MaxPool2d(2),

            ConvBNReLU(32, 64),
            ResidualBlock(64),
            if_true(use_se_block, SEBlock(64)),
            nn.MaxPool2d(2),

            ConvBNReLU(64, 128),
            ResidualBlock(128),
            if_true(use_se_block, SEBlock(128)),
            nn.MaxPool2d(2),

            ConvBNReLU(128, 256),
            nn.AdaptiveAvgPool2d(1),
        )

        self.feat_proj = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256, feat_dim),
            nn.BatchNorm1d(feat_dim),
            nn.GELU(),
        )

        # Skin tone head
        self.tone_head = nn.Sequential(
            nn.Linear(feat_dim, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(128, num_tones),
        )

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Args:
            x: [B, 3, H, W]

        Returns:
            dict:
                'feat'     : [B, feat_dim]
                'tone_pred': [B]  Fitzpatrick scale 1-6
                'logit'    : [B, num_tones]
                'prob'     : [B, num_tones]
        """
        feat = self.encoder(x)
        feat = self.feat_proj(feat)

        logit = self.tone_head(feat)
        prob = F.softmax(logit, dim=1)
        tone_pred = logit.argmax(dim=1)

        return {
            "feat"     : feat,
            "tone_pred": tone_pred,
            "logit"    : logit,
            "prob"     : prob,
        }


# ── Multi-Task Demographic Classifier ─────────────────────────────────────────

class MultiTaskDemographicClassifier(nn.Module):
    """
    Multi-task demographic classifier.
    Jointly predicts age, gender, race, skin tone from single image.

    Benefits of multi-task learning:
        • Shared features: demographic attributes share common visual patterns
        • Regularization: auxiliary tasks prevent overfitting to primary task
        • Efficiency: single forward pass for multiple predictions
        • Better generalization: hard-sharing of backbone

    Architecture:
        Shared Encoder
            ├── Age Head
            ├── Gender Head
            ├── Race Head
            └── Skin Tone Head

    Loss = α · age_loss + β · gender_loss + γ · race_loss + δ · tone_loss
    """

    def __init__(
        self,
        feat_dim: int = 256,
        age_type: str = "classification",
        num_age_groups: int = 6,
        num_genders: int = 2,
        num_races: int = 7,
        num_skin_tones: int = 6,
        use_se_block: bool = True,
    ):
        super().__init__()
        self.feat_dim = feat_dim

        # ── Shared Encoder ─────────────────────────────────────────────────
        self.shared_encoder = nn.Sequential(
            ConvBNReLU(3, 32),
            ResidualBlock(32),
            nn.MaxPool2d(2),

            ConvBNReLU(32, 64),
            ResidualBlock(64),
            if_true(use_se_block, SEBlock(64)),
            nn.MaxPool2d(2),

            ConvBNReLU(64, 128),
            ResidualBlock(128),
            if_true(use_se_block, SEBlock(128)),
            nn.MaxPool2d(2),

            ConvBNReLU(128, 256),
            ResidualBlock(256),
            if_true(use_se_block, SEBlock(256)),
            nn.AdaptiveAvgPool2d(1),  # [B, 256, 1, 1]
        )

        self.feat_proj = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256, feat_dim),
            nn.BatchNorm1d(feat_dim),
            nn.GELU(),
        )

        # ── Task-Specific Heads ────────────────────────────────────────────

        # Age head
        if age_type == "regression":
            age_out_dim = 1
        else:
            age_out_dim = num_age_groups

        self.age_head = nn.Sequential(
            nn.Linear(feat_dim, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(128, age_out_dim),
        )

        # Gender head
        self.gender_head = nn.Sequential(
            nn.Linear(feat_dim, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(128, num_genders),
        )

        # Race head
        self.race_head = nn.Sequential(
            nn.Linear(feat_dim, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(128, num_races),
        )

        # Skin tone head
        self.tone_head = nn.Sequential(
            nn.Linear(feat_dim, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(128, num_skin_tones),
        )

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Args:
            x: [B, 3, H, W]

        Returns:
            dict with keys:
                'feat'         : [B, feat_dim]  shared features
                'age_logit'    : age logits
                'age_pred'     : age prediction
                'gender_logit' : gender logits
                'gender_pred'  : gender prediction
                'gender_prob'  : gender probabilities
                'race_logit'   : race logits
                'race_pred'    : race prediction
                'race_prob'    : race probabilities
                'tone_logit'   : skin tone logits
                'tone_pred'    : skin tone prediction
                'tone_prob'    : skin tone probabilities
        """
        feat = self.shared_encoder(x)
        feat = self.feat_proj(feat)

        # Age
        age_logit = self.age_head(feat)
        age_pred = age_logit.squeeze(-1) if age_logit.shape[-1] == 1 else age_logit.argmax(dim=1)

        # Gender
        gender_logit = self.gender_head(feat)
        gender_prob = F.softmax(gender_logit, dim=1)
        gender_pred = gender_logit.argmax(dim=1)

        # Race
        race_logit = self.race_head(feat)
        race_prob = F.softmax(race_logit, dim=1)
        race_pred = race_logit.argmax(dim=1)

        # Skin Tone
        tone_logit = self.tone_head(feat)
        tone_prob = F.softmax(tone_logit, dim=1)
        tone_pred = tone_logit.argmax(dim=1)

        return {
            "feat"        : feat,
            "age_logit"   : age_logit,
            "age_pred"    : age_pred,
            "gender_logit": gender_logit,
            "gender_pred" : gender_pred,
            "gender_prob" : gender_prob,
            "race_logit"  : race_logit,
            "race_pred"   : race_pred,
            "race_prob"   : race_prob,
            "tone_logit"  : tone_logit,
            "tone_pred"   : tone_pred,
            "tone_prob"   : tone_prob,
        }


# ── Multi-Task Loss ───────────────────────────────────────────────────────────

class MultiTaskDemographicLoss(nn.Module):
    """
    Combined loss for multi-task demographic classification.

    L_total = α · L_age + β · L_gender + γ · L_race + δ · L_tone

    Balances accuracy across tasks while allowing importance weighting.
    """

    def __init__(
        self,
        age_type: str = "classification",
        alpha: float = 1.0,    # age loss weight
        beta: float = 1.0,     # gender loss weight
        gamma: float = 1.0,    # race loss weight
        delta: float = 1.0,    # skin tone loss weight
    ):
        super().__init__()
        self.age_type = age_type
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.delta = delta

        # Loss functions
        if age_type == "regression":
            self.age_loss_fn = nn.L1Loss()
        else:
            self.age_loss_fn = nn.CrossEntropyLoss()

        self.gender_loss_fn = nn.CrossEntropyLoss()
        self.race_loss_fn = nn.CrossEntropyLoss()
        self.tone_loss_fn = nn.CrossEntropyLoss()

    def forward(
        self,
        age_logit: torch.Tensor,
        age_target: torch.Tensor,
        gender_logit: torch.Tensor,
        gender_target: torch.Tensor,
        race_logit: torch.Tensor,
        race_target: torch.Tensor,
        tone_logit: torch.Tensor,
        tone_target: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Args:
            age_logit      : [B] or [B, num_age]
            age_target     : [B]
            gender_logit   : [B, num_gender]
            gender_target  : [B]
            race_logit     : [B, num_race]
            race_target    : [B]
            tone_logit     : [B, num_tone]
            tone_target    : [B]

        Returns:
            total_loss, metrics_dict
        """
        metrics = {}

        # Age loss
        age_loss = self.age_loss_fn(age_logit, age_target)
        metrics["age_loss"] = age_loss.item()

        # Gender loss
        gender_loss = self.gender_loss_fn(gender_logit, gender_target)
        metrics["gender_loss"] = gender_loss.item()

        # Race loss
        race_loss = self.race_loss_fn(race_logit, race_target)
        metrics["race_loss"] = race_loss.item()

        # Skin tone loss
        tone_loss = self.tone_loss_fn(tone_logit, tone_target)
        metrics["tone_loss"] = tone_loss.item()

        # Weighted combination
        total_loss = (
            self.alpha * age_loss
            + self.beta * gender_loss
            + self.gamma * race_loss
            + self.delta * tone_loss
        )

        metrics["total_loss"] = total_loss.item()

        return total_loss, metrics


# ── Demographic Classifier Gate ───────────────────────────────────────────────

class DemographicClassifierGate(nn.Module):
    """
    Drop-in demographic classifier gate for age, gender, race, skin tone prediction.

    Usage:
        demo_gate = fr.DemographicClassifierGate(weights_path="demo.pt")
        demo_out = demo_gate(face_image)

        age = demo_out["age_pred"]
        gender = demo_out["gender_pred"]
        race = demo_out["race_pred"]
        skin_tone = demo_out["tone_pred"]

    Applications:
        1. Fairness audits: check model accuracy per demographic group
        2. Dataset analysis: understand demographic distribution
        3. Bias detection: identify which groups have lower accuracy
        4. Filtering: select faces for specific demographic groups
        5. Anonymization: redact demographic info if needed
    """

    def __init__(
        self,
        weights_path: Optional[str] = None,
        feat_dim: int = 256,
        age_type: str = "classification",
        num_age_groups: int = 6,
        num_genders: int = 2,
        num_races: int = 7,
        num_skin_tones: int = 6,
        device: str = "cpu",
    ):
        super().__init__()
        self.device = torch.device(device)

        self.classifier = MultiTaskDemographicClassifier(
            feat_dim=feat_dim,
            age_type=age_type,
            num_age_groups=num_age_groups,
            num_genders=num_genders,
            num_races=num_races,
            num_skin_tones=num_skin_tones,
        ).to(self.device)

        self.classifier.eval()

        if weights_path is not None:
            state = torch.load(weights_path, map_location=self.device)
            self.classifier.load_state_dict(state)
            print(f"[DemographicClassifierGate] Loaded weights from: {weights_path}")
        else:
            print("[DemographicClassifierGate] ⚠️  No weights loaded — random init.")

        # Age group mappings
        self.age_groups = {
            0: "0-20",
            1: "21-30",
            2: "31-40",
            3: "41-50",
            4: "51-60",
            5: "60+",
        }

        self.gender_names = {
            0: "Male",
            1: "Female",
        }

        self.race_names = {
            0: "White",
            1: "Black",
            2: "Asian",
            3: "Hispanic",
            4: "Middle East",
            5: "Indian",
            6: "Other",
        }

        self.tone_names = {
            0: "1 (Very Light)",
            1: "2 (Light)",
            2: "3 (Medium Light)",
            3: "4 (Medium)",
            4: "5 (Dark)",
            5: "6 (Very Dark)",
        }

    @torch.no_grad()
    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Args:
            x: [B, 3, H, W] face image

        Returns:
            dict with predictions and probabilities
        """
        x = x.to(self.device)
        return self.classifier(x)

    @torch.no_grad()
    def predict_batch(
        self,
        images: torch.Tensor,
        return_names: bool = True,
    ) -> Dict[str, List]:
        """
        Predict demographics for batch with human-readable names.

        Args:
            images     : [B, 3, H, W]
            return_names: return names instead of indices

        Returns:
            dict with demographic predictions as strings
        """
        out = self(images)

        predictions = {
            "age"      : [],
            "gender"   : [],
            "race"     : [],
            "skin_tone": [],
        }

        for i in range(len(images)):
            if return_names:
                age_idx = out["age_pred"][i].item() if out["age_pred"].dim() > 0 else 0
                predictions["age"].append(self.age_groups.get(age_idx, "Unknown"))

                gender_idx = out["gender_pred"][i].item()
                predictions["gender"].append(self.gender_names.get(gender_idx, "Unknown"))

                race_idx = out["race_pred"][i].item()
                predictions["race"].append(self.race_names.get(race_idx, "Unknown"))

                tone_idx = out["tone_pred"][i].item()
                predictions["skin_tone"].append(self.tone_names.get(tone_idx, "Unknown"))
            else:
                predictions["age"].append(out["age_pred"][i].item())
                predictions["gender"].append(out["gender_pred"][i].item())
                predictions["race"].append(out["race_pred"][i].item())
                predictions["skin_tone"].append(out["tone_pred"][i].item())

        return predictions


# ── Utility Functions ─────────────────────────────────────────────────────────

def if_true(condition: bool, module: nn.Module) -> nn.Module:
    """Helper: return module if condition is true, else Identity."""
    return module if condition else nn.Identity()


# ── Demographic Analysis Tools ───────────────────────────────────────────────

class DemographicAccuracyAnalyzer(nn.Module):
    """
    Analyzes demographic classifier accuracy across groups.
    Computes per-group metrics to identify biased predictions.
    """

    def __init__(self):
        super().__init__()

    def analyze_accuracy(
        self,
        predictions: torch.Tensor,    # [N]
        ground_truth: torch.Tensor,   # [N]
        group_labels: torch.Tensor,   # [N]  e.g. gender
        group_names: Optional[Dict[int, str]] = None,
    ) -> Dict[str, Dict]:
        """
        Analyze accuracy per demographic group.

        Returns:
            dict with accuracy metrics per group
        """
        groups = torch.unique(group_labels).tolist()
        if group_names is None:
            group_names = {g: str(g) for g in groups}

        results = {}

        for group in groups:
            mask = group_labels == group
            group_preds = predictions[mask]
            group_truth = ground_truth[mask]

            accuracy = (group_preds == group_truth).float().mean().item()
            count = mask.sum().item()

            results[group_names[group]] = {
                "accuracy": accuracy,
                "count"   : count,
                "correct" : (group_preds == group_truth).sum().item(),
            }

        return results

    def compute_demographic_parity(
        self,
        predictions: torch.Tensor,
        ground_truth: torch.Tensor,
        demographic_attr: torch.Tensor,
        group_a: int,
        group_b: int,
    ) -> Dict[str, float]:
        """Compute fairness gap between two demographic groups."""
        mask_a = demographic_attr == group_a
        mask_b = demographic_attr == group_b

        acc_a = (predictions[mask_a] == ground_truth[mask_a]).float().mean().item()
        acc_b = (predictions[mask_b] == ground_truth[mask_b]).float().mean().item()

        return {
            "accuracy_a" : acc_a,
            "accuracy_b" : acc_b,
            "gap"        : abs(acc_a - acc_b),
            "ratio"      : acc_a / (acc_b + 1e-8),
        }
