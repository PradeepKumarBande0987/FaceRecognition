"""
Fairness & Bias Mitigation Module
Ensures equitable face recognition across demographics (age, gender, race, skin tone).
Addresses disparities in:
    • Recognition accuracy (demographic parity)
    • False positive/negative rates (equalized odds)
    • Embedding space fairness (balanced feature distributions)
    • Training data representation (balanced sampling)
Architecture: Debiasing techniques, fairness losses, demographic parity enforcement,
              adversarial debiasing, post-processing calibration.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Dict, List
import math
from collections import defaultdict


# ── Demographic Definitions ───────────────────────────────────────────────────

class Demographics:
    """Standard demographic attributes for fairness analysis."""

    # Age groups
    AGE_GROUPS = {
        "0-20"  : (0, 20),
        "21-30" : (21, 30),
        "31-40" : (31, 40),
        "41-50" : (41, 50),
        "51-60" : (51, 60),
        "60+"   : (61, 150),
    }

    # Gender
    GENDERS = {
        "M": 0,
        "F": 1,
    }

    # Race/Ethnicity (example: commonly used groupings)
    RACES = {
        "White"      : 0,
        "Black"      : 1,
        "Asian"      : 2,
        "Hispanic"   : 3,
        "Middle East": 4,
        "Indian"     : 5,
        "Other"      : 6,
    }

    # Skin tone (Fitzpatrick scale: 1-6)
    SKIN_TONES = {
        "1-2": (1, 2),    # very light
        "3-4": (3, 4),    # medium
        "5-6": (5, 6),    # dark
    }


# ── Balanced Data Sampler ─────────────────────────────────────────────────────

class DemographicBalancedSampler:
    """
    Balanced sampler ensuring equal representation of demographic groups.

    Prevents training on imbalanced datasets where majority groups
    dominate learning, causing minority group accuracy to suffer.

    Sampling strategies:
        • Stratified: equal samples per demographic group
        • Weighted: inverse frequency weighting (rare groups get higher prob)
        • Oversampling: upsample minority groups to match majority
    """

    def __init__(
        self,
        dataset_labels: List[Dict],  # [{"id": 0, "age": 25, "gender": "M", ...}, ...]
        protected_attr: str = "gender",  # which demographic to balance on
        sampling_strategy: str = "stratified",  # "stratified", "weighted", "oversample"
    ):
        """
        Args:
            dataset_labels      : list of metadata dicts per sample
            protected_attr      : demographic attribute name
            sampling_strategy   : how to balance
        """
        self.dataset_labels = dataset_labels
        self.protected_attr = protected_attr
        self.sampling_strategy = sampling_strategy

        # Group indices by demographic value
        self.groups = defaultdict(list)
        for idx, label_dict in enumerate(dataset_labels):
            attr_value = label_dict.get(protected_attr)
            if attr_value is not None:
                self.groups[attr_value].append(idx)

        # Compute sampling weights
        self._compute_weights()

    def _compute_weights(self):
        """Compute per-sample weights based on sampling strategy."""
        self.weights = [0.0] * len(self.dataset_labels)

        group_sizes = {g: len(indices) for g, indices in self.groups.items()}
        num_groups = len(self.groups)

        if self.sampling_strategy == "stratified":
            # Equal probability per group
            weight_per_group = 1.0 / num_groups
            for group, indices in self.groups.items():
                weight_per_sample = weight_per_group / len(indices)
                for idx in indices:
                    self.weights[idx] = weight_per_sample

        elif self.sampling_strategy == "weighted":
            # Inverse frequency: rare groups get higher weight
            total_samples = len(self.dataset_labels)
            for group, indices in self.groups.items():
                group_size = len(indices)
                inv_freq = 1.0 / group_size
                # Normalize so sum = 1
                weight_per_sample = (inv_freq / num_groups) / (inv_freq * num_groups)
                for idx in indices:
                    self.weights[idx] = weight_per_sample

        elif self.sampling_strategy == "oversample":
            # Oversample minorities to match largest group
            max_group_size = max(group_sizes.values())
            for group, indices in self.groups.items():
                group_size = len(indices)
                oversample_factor = max_group_size / group_size
                weight_per_sample = oversample_factor / len(self.dataset_labels)
                for idx in indices:
                    self.weights[idx] = weight_per_sample

    def get_weights(self) -> torch.Tensor:
        """Return sampling weights for PyTorch WeightedRandomSampler."""
        return torch.tensor(self.weights, dtype=torch.float32)


# ── Fairness Metrics ──────────────────────────────────────────────────────────

class FairnessMetrics(nn.Module):
    """
    Computes fairness metrics across demographic groups.

    Metrics:
        • Demographic Parity: P(Y_pred=1 | group A) = P(Y_pred=1 | group B)
        • Equalized Odds: FPR(A) = FPR(B) and TPR(A) = TPR(B)
        • Predictive Parity: P(Y_true=1 | Y_pred=1, A) = P(Y_true=1 | Y_pred=1, B)
        • Calibration: Predicted confidence ≈ actual accuracy per group
        • Accuracy Parity: Accuracy(A) ≈ Accuracy(B)
    """

    def __init__(self):
        super().__init__()

    @staticmethod
    def demographic_parity(
        predictions: torch.Tensor,     # [N]  predicted labels 0/1
        protected_attr: torch.Tensor,  # [N]  group membership
        group_a: int,
        group_b: int,
    ) -> Dict[str, float]:
        """
        Demographic Parity: P(Y_pred=1 | group A) should equal P(Y_pred=1 | group B)
        Disparity ratio = P(Y_pred=1 | A) / P(Y_pred=1 | B)
        Ideal: ratio = 1.0
        """
        mask_a = protected_attr == group_a
        mask_b = protected_attr == group_b

        p_positive_a = predictions[mask_a].float().mean().item()
        p_positive_b = predictions[mask_b].float().mean().item()

        disparity_ratio = p_positive_a / (p_positive_b + 1e-8)
        difference = (p_positive_a - p_positive_b).abs().item()

        return {
            "p_positive_a"   : p_positive_a,
            "p_positive_b"   : p_positive_b,
            "disparity_ratio": disparity_ratio,
            "difference"     : difference,
        }

    @staticmethod
    def equalized_odds(
        predictions: torch.Tensor,     # [N]
        ground_truth: torch.Tensor,    # [N]  true labels
        protected_attr: torch.Tensor,  # [N]
        group_a: int,
        group_b: int,
    ) -> Dict[str, float]:
        """
        Equalized Odds: FPR(A) = FPR(B) and TPR(A) = TPR(B)

        TPR (True Positive Rate) = TP / (TP + FN)  [sensitivity]
        FPR (False Positive Rate) = FP / (FP + TN)  [1 - specificity]
        """
        mask_a = protected_attr == group_a
        mask_b = protected_attr == group_b

        pred_a = predictions[mask_a]
        true_a = ground_truth[mask_a]
        pred_b = predictions[mask_b]
        true_b = ground_truth[mask_b]

        # Group A
        tp_a = ((pred_a == 1) & (true_a == 1)).float().sum()
        fn_a = ((pred_a == 0) & (true_a == 1)).float().sum()
        fp_a = ((pred_a == 1) & (true_a == 0)).float().sum()
        tn_a = ((pred_a == 0) & (true_a == 0)).float().sum()

        tpr_a = tp_a / (tp_a + fn_a + 1e-8)
        fpr_a = fp_a / (fp_a + tn_a + 1e-8)

        # Group B
        tp_b = ((pred_b == 1) & (true_b == 1)).float().sum()
        fn_b = ((pred_b == 0) & (true_b == 1)).float().sum()
        fp_b = ((pred_b == 1) & (true_b == 0)).float().sum()
        tn_b = ((pred_b == 0) & (true_b == 0)).float().sum()

        tpr_b = tp_b / (tp_b + fn_b + 1e-8)
        fpr_b = fp_b / (fp_b + tn_b + 1e-8)

        return {
            "tpr_a": tpr_a.item(),
            "tpr_b": tpr_b.item(),
            "tpr_gap": (tpr_a - tpr_b).abs().item(),
            "fpr_a": fpr_a.item(),
            "fpr_b": fpr_b.item(),
            "fpr_gap": (fpr_a - fpr_b).abs().item(),
        }

    @staticmethod
    def accuracy_parity(
        predictions: torch.Tensor,
        ground_truth: torch.Tensor,
        protected_attr: torch.Tensor,
        group_a: int,
        group_b: int,
    ) -> Dict[str, float]:
        """
        Accuracy Parity: Accuracy should be similar across groups.
        """
        mask_a = protected_attr == group_a
        mask_b = protected_attr == group_b

        acc_a = (predictions[mask_a] == ground_truth[mask_a]).float().mean().item()
        acc_b = (predictions[mask_b] == ground_truth[mask_b]).float().mean().item()

        return {
            "accuracy_a"     : acc_a,
            "accuracy_b"     : acc_b,
            "accuracy_gap"   : (acc_a - acc_b).abs().item(),
            "accuracy_ratio" : acc_a / (acc_b + 1e-8),
        }

    @staticmethod
    def predictive_parity(
        predictions: torch.Tensor,
        ground_truth: torch.Tensor,
        protected_attr: torch.Tensor,
        group_a: int,
        group_b: int,
    ) -> Dict[str, float]:
        """
        Predictive Parity (Precision Parity):
        P(Y_true=1 | Y_pred=1, group A) = P(Y_true=1 | Y_pred=1, group B)
        """
        mask_a = protected_attr == group_a
        mask_b = protected_attr == group_b

        # Filter to predicted positives
        pred_pos_a = (predictions[mask_a] == 1)
        pred_pos_b = (predictions[mask_b] == 1)

        if pred_pos_a.sum() > 0:
            precision_a = (ground_truth[mask_a][pred_pos_a] == 1).float().mean().item()
        else:
            precision_a = 0.0

        if pred_pos_b.sum() > 0:
            precision_b = (ground_truth[mask_b][pred_pos_b] == 1).float().mean().item()
        else:
            precision_b = 0.0

        return {
            "precision_a": precision_a,
            "precision_b": precision_b,
            "precision_gap": (precision_a - precision_b).abs().item(),
        }


# ── Fairness Losses ──────────────────────────────────────────────────────────

class DemographicParityLoss(nn.Module):
    """
    Enforces demographic parity in model predictions.
    P(Y_pred=1 | group A) = P(Y_pred=1 | group B)

    Loss minimizes difference in positive prediction rates across groups.
    """

    def __init__(self):
        super().__init__()

    def forward(
        self,
        logits: torch.Tensor,          # [B, num_classes] or [B]  raw predictions
        protected_attr: torch.Tensor,  # [B]  group membership
        protected_groups: List[int],   # which groups to enforce parity between
    ) -> torch.Tensor:
        """
        Args:
            logits         : model predictions
            protected_attr : demographic group per sample
            protected_groups: groups to constrain (e.g. [0, 1] for M/F)

        Returns:
            parity_loss: scalar
        """
        if logits.dim() > 1:
            # Multi-class: take softmax
            probs = F.softmax(logits, dim=-1)
            pos_probs = probs[:, 1] if probs.shape[1] >= 2 else probs.sum(dim=1)
        else:
            # Binary: sigmoid
            pos_probs = torch.sigmoid(logits)

        loss = 0.0
        num_pairs = 0

        # Compare all pairs of groups
        for i, group_a in enumerate(protected_groups):
            for group_b in protected_groups[i + 1:]:
                mask_a = protected_attr == group_a
                mask_b = protected_attr == group_b

                if mask_a.sum() > 0 and mask_b.sum() > 0:
                    mean_a = pos_probs[mask_a].mean()
                    mean_b = pos_probs[mask_b].mean()

                    # KL divergence or L2 distance
                    loss += (mean_a - mean_b).abs()
                    num_pairs += 1

        return loss / max(num_pairs, 1)


class EqualizedOddsLoss(nn.Module):
    """
    Enforces equalized odds: FPR(A) = FPR(B) and TPR(A) = TPR(B)

    More stringent than demographic parity, requires fairness
    conditioned on ground truth label.
    """

    def __init__(self):
        super().__init__()

    def forward(
        self,
        logits: torch.Tensor,          # [B, num_classes] or [B]
        labels: torch.Tensor,          # [B]  ground truth
        protected_attr: torch.Tensor,  # [B]
        protected_groups: List[int],
    ) -> torch.Tensor:
        """
        Args:
            logits         : predictions
            labels         : ground truth labels
            protected_attr : group membership
            protected_groups: groups to enforce parity between

        Returns:
            eq_odds_loss: scalar
        """
        if logits.dim() > 1:
            preds = logits.argmax(dim=1)
        else:
            preds = (logits > 0.5).long()

        loss = 0.0
        num_terms = 0

        for i, group_a in enumerate(protected_groups):
            for group_b in protected_groups[i + 1:]:
                mask_a = protected_attr == group_a
                mask_b = protected_attr == group_b

                # TPR for positives
                pos_mask_a = (labels == 1) & mask_a
                pos_mask_b = (labels == 1) & mask_b

                if pos_mask_a.sum() > 0 and pos_mask_b.sum() > 0:
                    tpr_a = (preds[pos_mask_a] == 1).float().mean()
                    tpr_b = (preds[pos_mask_b] == 1).float().mean()
                    loss += (tpr_a - tpr_b).abs()
                    num_terms += 1

                # FPR for negatives
                neg_mask_a = (labels == 0) & mask_a
                neg_mask_b = (labels == 0) & mask_b

                if neg_mask_a.sum() > 0 and neg_mask_b.sum() > 0:
                    fpr_a = (preds[neg_mask_a] == 1).float().mean()
                    fpr_b = (preds[neg_mask_b] == 1).float().mean()
                    loss += (fpr_a - fpr_b).abs()
                    num_terms += 1

        return loss / max(num_terms, 1)


class AdversarialDebiasing(nn.Module):
    """
    Adversarial Debiasing: makes embeddings invariant to protected attributes.

    Architecture:
        Face Recognizer
            ├── face embedding
            │
            ├── Identity classifier (minimize)
            │
            └── Demographic predictor (adversarial; should maximize loss)

    The recognizer is trained to:
        • Minimize identity classification loss
        • Maximize demographic prediction loss (make demos unrecoverable)

    This forces embeddings to remove demographic information
    while preserving identity information.
    """

    def __init__(
        self,
        emb_dim: int = 512,
        num_demographics: int = 2,  # e.g. binary gender
    ):
        super().__init__()

        # Demographic predictor (adversary)
        self.demographic_predictor = nn.Sequential(
            nn.Linear(emb_dim, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, num_demographics),
        )

    def forward(
        self,
        embedding: torch.Tensor,      # [B, emb_dim]
        protected_attr: torch.Tensor, # [B]
    ) -> Dict[str, torch.Tensor]:
        """
        Args:
            embedding     : face identity embedding
            protected_attr: ground truth demographic

        Returns:
            dict:
                'demo_logit': [B, num_demographics]  predicted demographics
                'demo_loss' : scalar adversarial loss
        """
        demo_logit = self.demographic_predictor(embedding)

        # Cross-entropy: adversary tries to predict demographics
        demo_loss = F.cross_entropy(demo_logit, protected_attr)

        return {
            "demo_logit": demo_logit,
            "demo_loss" : demo_loss,
        }


# ── Post-Processing Calibration ───────────────────────────────────────────────

class FairnessCalibrator(nn.Module):
    """
    Post-processing calibration to enforce fairness on predictions.

    Methods:
        • Threshold Optimization: adjust decision threshold per group
        • Equalized Odds: search for thresholds that equalize TPR/FPR
        • Demographic Parity: search for thresholds that equalize positive rates
    """

    def __init__(self):
        super().__init__()
        self.thresholds = {}  # per-group thresholds

    def compute_thresholds_equalized_odds(
        self,
        scores: torch.Tensor,         # [N]  similarity/confidence scores
        labels: torch.Tensor,         # [N]  ground truth (1=match, 0=no match)
        protected_attr: torch.Tensor, # [N]  group membership
        protected_groups: List[int],
        target_tpr: float = 0.95,
    ) -> Dict[int, float]:
        """
        Find thresholds per group that equalize TPR across groups.

        For each group, find threshold that achieves target_tpr,
        then use that threshold to make predictions.
        """
        self.thresholds = {}

        for group in protected_groups:
            mask = protected_attr == group
            group_scores = scores[mask]
            group_labels = labels[mask]

            # Sort by score
            sorted_indices = torch.argsort(group_scores, descending=True)
            sorted_scores = group_scores[sorted_indices]
            sorted_labels = group_labels[sorted_indices]

            # Find threshold that achieves target_tpr
            cum_tp = torch.cumsum(sorted_labels, dim=0).float()
            total_positives = sorted_labels.sum().float()

            tpr = cum_tp / (total_positives + 1e-8)

            # Find closest to target_tpr
            idx = torch.argmin((tpr - target_tpr).abs())
            threshold = sorted_scores[idx].item()

            self.thresholds[group] = threshold

        return self.thresholds

    def compute_thresholds_demographic_parity(
        self,
        scores: torch.Tensor,
        protected_attr: torch.Tensor,
        protected_groups: List[int],
        target_positive_rate: float = 0.1,
    ) -> Dict[int, float]:
        """
        Find thresholds per group that equalize positive prediction rates.
        """
        self.thresholds = {}

        for group in protected_groups:
            mask = protected_attr == group
            group_scores = scores[mask]

            # Sort descending and find threshold to achieve target_positive_rate
            sorted_scores = torch.sort(group_scores, descending=True).values
            idx = int(len(sorted_scores) * target_positive_rate)
            threshold = sorted_scores[idx].item() if idx < len(sorted_scores) else 0.0

            self.thresholds[group] = threshold

        return self.thresholds

    def calibrate_predictions(
        self,
        scores: torch.Tensor,         # [N]
        protected_attr: torch.Tensor, # [N]
    ) -> torch.Tensor:
        """
        Apply per-group thresholds to scores.

        Returns:
            predictions: [N]  0/1 based on group-specific thresholds
        """
        predictions = torch.zeros_like(scores)

        for group, threshold in self.thresholds.items():
            mask = protected_attr == group
            predictions[mask] = (scores[mask] > threshold).long()

        return predictions


# ── Fairness-Aware Training ───────────────────────────────────────────────────

class FairnessAwareTrainer(nn.Module):
    """
    Combined fairness-aware training framework.

    Loss = α · Identity Loss
         + β · Fairness Loss (demographic parity or equalized odds)
         + γ · Adversarial Debiasing Loss

    Balances accuracy with fairness during training.
    """

    def __init__(
        self,
        emb_dim: int = 512,
        num_classes: int = 10000,
        num_demographics: int = 2,
        fairness_type: str = "parity",  # "parity" or "equalized_odds"
        alpha: float = 1.0,              # identity loss weight
        beta: float = 0.5,               # fairness loss weight
        gamma: float = 0.3,              # adversarial loss weight
    ):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma

        # Identity loss
        self.identity_loss = nn.CrossEntropyLoss()

        # Fairness losses
        if fairness_type == "parity":
            self.fairness_loss = DemographicParityLoss()
        elif fairness_type == "equalized_odds":
            self.fairness_loss = EqualizedOddsLoss()
        else:
            raise ValueError(f"Unknown fairness type: {fairness_type}")

        # Adversarial debiasing
        self.adversarial_debiasing = AdversarialDebiasing(emb_dim, num_demographics)

    def forward(
        self,
        logits: torch.Tensor,          # [B, num_classes]
        embedding: torch.Tensor,       # [B, emb_dim]
        labels: torch.Tensor,          # [B]  identity labels
        protected_attr: torch.Tensor,  # [B]  demographic labels
        protected_groups: List[int] = None,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Args:
            logits         : identity classification logits
            embedding      : face embeddings
            labels         : identity class labels
            protected_attr : demographic attributes
            protected_groups: which groups to enforce fairness between

        Returns:
            total_loss, metrics_dict
        """
        if protected_groups is None:
            protected_groups = list(torch.unique(protected_attr).cpu().tolist())

        metrics = {}

        # ── Identity Loss ────────────────────────────────────────────────────
        id_loss = self.identity_loss(logits, labels)
        total_loss = self.alpha * id_loss
        metrics["id_loss"] = id_loss.item()

        # ── Fairness Loss ────────────────────────────────────────────────────
        fair_loss = self.fairness_loss(logits, labels, protected_attr, protected_groups)
        total_loss += self.beta * fair_loss
        metrics["fairness_loss"] = fair_loss.item()

        # ── Adversarial Debiasing ────────────────────────────────────────────
        adv_out = self.adversarial_debiasing(embedding, protected_attr)
        adv_loss = -adv_out["demo_loss"]  # negative: minimize ability to predict demographics
        total_loss += self.gamma * adv_loss
        metrics["adversarial_loss"] = adv_loss.item()

        metrics["total_loss"] = total_loss.item()

        return total_loss, metrics


# ── Fairness Audit ───────────────────────────────────────────────────────────

class FairnessAudit(nn.Module):
    """
    Comprehensive fairness audit of deployed face recognition model.

    Evaluates multiple fairness metrics and reports disparities.
    """

    def __init__(self):
        super().__init__()
        self.metrics_fn = FairnessMetrics()

    def audit(
        self,
        predictions: torch.Tensor,     # [N]  model predictions
        ground_truth: torch.Tensor,    # [N]  ground truth
        protected_attr: torch.Tensor,  # [N]  demographic attribute
        group_names: Optional[Dict[int, str]] = None,
    ) -> Dict[str, Dict]:
        """
        Run comprehensive fairness audit.

        Args:
            predictions : model predictions
            ground_truth: ground truth labels
            protected_attr: demographic attribute per sample
            group_names  : mapping from group id to name (e.g. {0: "Male", 1: "Female"})

        Returns:
            audit_report: nested dict with fairness metrics per group pair
        """
        if group_names is None:
            groups = sorted(torch.unique(protected_attr).tolist())
            group_names = {g: str(g) for g in groups}

        groups = list(group_names.keys())
        report = {}

        # Pairwise comparisons
        for i, group_a in enumerate(groups):
            for group_b in groups[i + 1:]:
                pair_name = f"{group_names[group_a]} vs {group_names[group_b]}"

                pair_report = {}

                # Demographic Parity
                dem_par = self.metrics_fn.demographic_parity(
                    predictions, protected_attr, group_a, group_b
                )
                pair_report["demographic_parity"] = dem_par

                # Equalized Odds
                eq_odds = self.metrics_fn.equalized_odds(
                    predictions, ground_truth, protected_attr, group_a, group_b
                )
                pair_report["equalized_odds"] = eq_odds

                # Accuracy Parity
                acc_par = self.metrics_fn.accuracy_parity(
                    predictions, ground_truth, protected_attr, group_a, group_b
                )
                pair_report["accuracy_parity"] = acc_par

                # Predictive Parity
                pred_par = self.metrics_fn.predictive_parity(
                    predictions, ground_truth, protected_attr, group_a, group_b
                )
                pair_report["predictive_parity"] = pred_par

                report[pair_name] = pair_report

        return report

    def print_report(self, report: Dict):
        """Pretty-print audit report."""
        print("\n" + "="*80)
        print("FAIRNESS AUDIT REPORT")
        print("="*80 + "\n")

        for pair_name, pair_metrics in report.items():
            print(f"📊 {pair_name}")
            print("-" * 80)

            # Demographic Parity
            dp = pair_metrics["demographic_parity"]
            print(f"  Demographic Parity:")
            print(f"    P(Y_pred=1 | Group A) = {dp['p_positive_a']:.4f}")
            print(f"    P(Y_pred=1 | Group B) = {dp['p_positive_b']:.4f}")
            print(f"    Disparity Ratio      = {dp['disparity_ratio']:.4f} " +
                  ("✅ FAIR" if 0.8 <= dp['disparity_ratio'] <= 1.25 else "❌ UNFAIR"))
            print()

            # Equalized Odds
            eo = pair_metrics["equalized_odds"]
            print(f"  Equalized Odds:")
            print(f"    TPR Gap (Group A-B)  = {eo['tpr_gap']:.4f} " +
                  ("✅" if eo['tpr_gap'] < 0.1 else "❌"))
            print(f"    FPR Gap (Group A-B)  = {eo['fpr_gap']:.4f} " +
                  ("✅" if eo['fpr_gap'] < 0.1 else "❌"))
            print()

            # Accuracy Parity
            ap = pair_metrics["accuracy_parity"]
            print(f"  Accuracy Parity:")
            print(f"    Accuracy Group A     = {ap['accuracy_a']:.4f}")
            print(f"    Accuracy Group B     = {ap['accuracy_b']:.4f}")
            print(f"    Accuracy Gap         = {ap['accuracy_gap']:.4f} " +
                  ("✅" if ap['accuracy_gap'] < 0.05 else "❌"))
            print()

            print()


# ── Fairness Gate (pipeline integration) ──────────────────────────────────────

class FairnessGate(nn.Module):
    """
    Drop-in fairness assurance gate for face recognition pipeline.

    Ensures model predictions meet fairness requirements before deployment.

    Workflow:
        1. Run fairness audit on test set
        2. Identify disparities
        3. Apply post-processing calibration if needed
        4. Gate deployment if disparities exceed thresholds
    """

    def __init__(
        self,
        fairness_type: str = "equalized_odds",  # "demographic_parity" or "equalized_odds"
        tolerance: float = 0.1,                  # max allowed gap
        calibrate: bool = True,
    ):
        super().__init__()
        self.fairness_type = fairness_type
        self.tolerance = tolerance
        self.calibrate = calibrate

        self.audit_fn = FairnessAudit()
        self.calibrator = FairnessCalibrator() if calibrate else None

    def check_fairness(
        self,
        predictions: torch.Tensor,
        ground_truth: torch.Tensor,
        protected_attr: torch.Tensor,
        group_names: Optional[Dict[int, str]] = None,
    ) -> Tuple[bool, Dict]:
        """
        Check if predictions meet fairness requirements.

        Args:
            predictions : model predictions
            ground_truth: ground truth labels
            protected_attr: demographic attribute
            group_names  : group names

        Returns:
            (passed: bool, report: dict)
        """
        report = self.audit_fn.audit(
            predictions, ground_truth, protected_attr, group_names
        )

        passed = True
        violations = []

        for pair_name, pair_metrics in report.items():
            if self.fairness_type == "demographic_parity":
                dp = pair_metrics["demographic_parity"]
                if not (0.8 <= dp["disparity_ratio"] <= 1.25):
                    passed = False
                    violations.append(f"{pair_name}: Disparity ratio = {dp['disparity_ratio']:.4f}")

            elif self.fairness_type == "equalized_odds":
                eo = pair_metrics["equalized_odds"]
                if eo["tpr_gap"] > self.tolerance or eo["fpr_gap"] > self.tolerance:
                    passed = False
                    violations.append(f"{pair_name}: TPR gap={eo['tpr_gap']:.4f}, FPR gap={eo['fpr_gap']:.4f}")

        return passed, {"report": report, "violations": violations}

    def calibrate_predictions(
        self,
        scores: torch.Tensor,
        labels: torch.Tensor,
        protected_attr: torch.Tensor,
        calibration_type: str = "equalized_odds",
    ) -> torch.Tensor:
        """
        Apply fairness calibration to predictions.

        Args:
            scores    : similarity scores or confidence
            labels    : ground truth (for equalized_odds)
            protected_attr: demographic attribute
            calibration_type: "equalized_odds" or "demographic_parity"

        Returns:
            calibrated_predictions: [N]
        """
        if calibration_type == "equalized_odds":
            self.calibrator.compute_thresholds_equalized_odds(
                scores, labels, protected_attr,
                list(torch.unique(protected_attr).tolist()),
                target_tpr=0.95,
            )
        elif calibration_type == "demographic_parity":
            self.calibrator.compute_thresholds_demographic_parity(
                scores, protected_attr,
                list(torch.unique(protected_attr).tolist()),
                target_positive_rate=0.1,
            )

        return self.calibrator.calibrate_predictions(scores, protected_attr)


# ── Bias Analysis Tools ──────────────────────────────────────────────────────

class BiasAnalyzer(nn.Module):
    """
    Analyzes model bias along multiple dimensions.

    Provides:
        • Intersectional fairness (fairness across multiple protected attributes)
        • Bias by age, gender, race, skin tone
        • Visualization-ready metrics
    """

    def __init__(self):
        super().__init__()
        self.metrics_fn = FairnessMetrics()

    def analyze_by_age_groups(
        self,
        predictions: torch.Tensor,
        ground_truth: torch.Tensor,
        ages: torch.Tensor,  # actual ages in years
    ) -> Dict[str, Dict]:
        """
        Analyze accuracy by age groups.
        """
        results = {}

        for age_label, (min_age, max_age) in Demographics.AGE_GROUPS.items():
            mask = (ages >= min_age) & (ages <= max_age)

            if mask.sum() > 0:
                acc = (predictions[mask] == ground_truth[mask]).float().mean().item()
                size = mask.sum().item()
                results[age_label] = {
                    "accuracy": acc,
                    "sample_size": size,
                }

        return results

    def analyze_intersectional(
        self,
        predictions: torch.Tensor,
        ground_truth: torch.Tensor,
        protected_attrs: Dict[str, torch.Tensor],  # {"gender": [B], "race": [B], ...}
    ) -> Dict[Tuple, Dict]:
        """
        Analyze fairness at intersections of multiple attributes.
        E.g. Black women, Asian men, etc.
        """
        results = {}

        # Get unique combinations
        attr_names = list(protected_attrs.keys())
        attr_values = [torch.unique(protected_attrs[name]).tolist() for name in attr_names]

        # Cartesian product
        import itertools
        combinations = list(itertools.product(*attr_values))

        for combo in combinations:
            mask = torch.ones(len(predictions), dtype=torch.bool)
            combo_name = ""

            for attr_name, attr_val in zip(attr_names, combo):
                mask &= protected_attrs[attr_name] == attr_val
                combo_name += f"{attr_name}={attr_val},"

            combo_name = combo_name.rstrip(",")

            if mask.sum() > 0:
                acc = (predictions[mask] == ground_truth[mask]).float().mean().item()
                size = mask.sum().item()
                results[combo] = {
                    "name": combo_name,
                    "accuracy": acc,
                    "sample_size": size,
                }

        return results
