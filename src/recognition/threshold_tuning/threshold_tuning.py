"""
Recognition — Threshold Tuning Module.

Finds the optimal decision threshold τ for face verification.

Methods:
    • EER threshold         : Equal Error Rate (threshold-free)
    • Youden J statistic    : maximize TPR − FPR
    • FAR-constrained TAR   : maximize TAR at target FAR
    • F1-score maximization : balance precision/recall
    • Operational tuning    : use case-specific (security vs convenience)

Outputs:
    • Optimal threshold value
    • Performance at optimal threshold
    • ROC curve data
    • Threshold sensitivity report
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np
from sklearn.metrics import roc_auc_score, roc_curve


@dataclass
class ThresholdResult:
    """Result from threshold optimization."""
    threshold   : float
    method      : str
    tpr         : float
    fpr         : float
    fnmr        : float
    fmr         : float
    auc         : float
    eer         : float
    f1          : float


class ThresholdTuner:
    """
    Finds optimal verification threshold from genuine/impostor scores.

    Usage:
        tuner = ThresholdTuner(scores, labels)
        result = tuner.find_optimal(method="eer")
        print(f"Optimal threshold: {result.threshold:.4f}")
    """

    def __init__(
        self,
        scores : List[float] | np.ndarray,
        labels : List[int]   | np.ndarray,
    ):
        self.scores = np.array(scores, dtype=np.float32)
        self.labels = np.array(labels, dtype=np.int32)

        assert len(self.scores) == len(self.labels), \
            "scores and labels must have same length"
        assert set(np.unique(self.labels)).issubset({0, 1}), \
            "labels must be binary (0=impostor, 1=genuine)"

        # Precompute ROC
        self.fpr_arr, self.tpr_arr, self.thresholds = roc_curve(
            self.labels, self.scores, pos_label=1
        )
        self.auc = float(roc_auc_score(self.labels, self.scores))
        self.fnr_arr = 1 - self.tpr_arr

    # ── EER Threshold ─────────────────────────────────────────────────────────

    def find_eer_threshold(self) -> ThresholdResult:
        """
        Find threshold where FPR ≈ FNR (Equal Error Rate).

        The EER is the standard threshold-free evaluation metric
        for face verification. Lower EER = better system.
        """
        eer_idx = np.nanargmin(np.abs(self.fnr_arr - self.fpr_arr))
        eer      = float((self.fpr_arr[eer_idx] + self.fnr_arr[eer_idx]) / 2)
        thresh   = float(self.thresholds[eer_idx])

        fmr, fnmr, f1 = self._metrics_at_threshold(thresh)
        return ThresholdResult(
            threshold = round(thresh, 4),
            method    = "eer",
            tpr       = round(float(self.tpr_arr[eer_idx]), 4),
            fpr       = round(float(self.fpr_arr[eer_idx]), 4),
            fnmr      = round(fnmr, 4),
            fmr       = round(fmr, 4),
            auc       = round(self.auc, 4),
            eer       = round(eer, 4),
            f1        = round(f1, 4),
        )

    # ── Youden Threshold ──────────────────────────────────────────────────────

    def find_youden_threshold(self) -> ThresholdResult:
        """
        Youden J statistic: maximizes TPR − FPR.

        Best threshold for maximizing the informedness of the classifier.
        """
        j_scores = self.tpr_arr - self.fpr_arr
        best_idx = np.argmax(j_scores)
        thresh   = float(self.thresholds[best_idx])

        fmr, fnmr, f1 = self._metrics_at_threshold(thresh)
        eer_result     = self.find_eer_threshold()

        return ThresholdResult(
            threshold = round(thresh, 4),
            method    = "youden",
            tpr       = round(float(self.tpr_arr[best_idx]), 4),
            fpr       = round(float(self.fpr_arr[best_idx]), 4),
            fnmr      = round(fnmr, 4),
            fmr       = round(fmr, 4),
            auc       = round(self.auc, 4),
            eer       = eer_result.eer,
            f1        = round(f1, 4),
        )

    # ── FAR-Constrained Threshold ─────────────────────────────────────────────

    def find_far_constrained_threshold(
        self,
        target_far : float = 0.001,   # TAR @ FAR=0.1%
    ) -> ThresholdResult:
        """
        Find threshold that achieves target FAR.

        Maximizes TAR (True Accept Rate) subject to FAR ≤ target_far.
        Standard operational metric for security systems.

        Args:
            target_far: maximum acceptable False Accept Rate
        """
        valid_indices = np.where(self.fpr_arr <= target_far)[0]

        if len(valid_indices) == 0:
            best_idx = 0
        else:
            best_idx = valid_indices[np.argmax(self.tpr_arr[valid_indices])]

        thresh        = float(self.thresholds[best_idx])
        fmr, fnmr, f1 = self._metrics_at_threshold(thresh)
        eer_result     = self.find_eer_threshold()

        return ThresholdResult(
            threshold = round(thresh, 4),
            method    = f"far_constrained_{target_far}",
            tpr       = round(float(self.tpr_arr[best_idx]), 4),
            fpr       = round(float(self.fpr_arr[best_idx]), 4),
            fnmr      = round(fnmr, 4),
            fmr       = round(fmr, 4),
            auc       = round(self.auc, 4),
            eer       = eer_result.eer,
            f1        = round(f1, 4),
        )

    # ── F1 Threshold ──────────────────────────────────────────────────────────

    def find_f1_threshold(self) -> ThresholdResult:
        """Find threshold that maximizes F1 score."""
        best_f1    = -1
        best_thresh = 0.5

        for thresh in np.linspace(0.0, 1.0, 200):
            _, _, f1 = self._metrics_at_threshold(thresh)
            if f1 > best_f1:
                best_f1     = f1
                best_thresh = thresh

        fmr, fnmr, f1 = self._metrics_at_threshold(best_thresh)
        eer_result     = self.find_eer_threshold()

        tpr_idx = np.searchsorted(self.thresholds[::-1], best_thresh)
        tpr_val = float(self.tpr_arr[-(tpr_idx + 1)]) if tpr_idx < len(self.tpr_arr) else 0.0

        return ThresholdResult(
            threshold = round(best_thresh, 4),
            method    = "f1_maximization",
            tpr       = round(tpr_val, 4),
            fpr       = round(fmr, 4),
            fnmr      = round(fnmr, 4),
            fmr       = round(fmr, 4),
            auc       = round(self.auc, 4),
            eer       = eer_result.eer,
            f1        = round(f1, 4),
        )

    # ── Find Optimal ──────────────────────────────────────────────────────────

    def find_optimal(
        self,
        method     : str   = "eer",
        target_far : float = 0.001,
    ) -> ThresholdResult:
        """
        Find optimal threshold using specified method.

        Args:
            method     : "eer" | "youden" | "far_constrained" | "f1"
            target_far : used only for method="far_constrained"

        Returns:
            ThresholdResult with optimal threshold and metrics
        """
        dispatch = {
            "eer"            : self.find_eer_threshold,
            "youden"         : self.find_youden_threshold,
            "far_constrained": lambda: self.find_far_constrained_threshold(target_far),
            "f1"             : self.find_f1_threshold,
        }
        if method not in dispatch:
            raise ValueError(f"Unknown method: {method}. Choose: {list(dispatch)}")
        return dispatch[method]()

    # ── Summary ───────────────────────────────────────────────────────────────

    def summary(self) -> dict:
        """Run all methods and return comparison dict."""
        results = {}
        for method in ["eer", "youden", "f1"]:
            r = self.find_optimal(method=method)
            results[method] = {
                "threshold": r.threshold,
                "tpr"      : r.tpr,
                "fpr"      : r.fpr,
                "auc"      : r.auc,
                "eer"      : r.eer,
                "f1"       : r.f1,
            }
        return results

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _metrics_at_threshold(
        self,
        threshold : float,
    ) -> Tuple[float, float, float]:
        """Return (FMR, FNMR, F1) at given threshold."""
        preds    = (self.scores >= threshold).astype(int)
        genuine  = self.labels == 1
        impostor = self.labels == 0

        fnmr = float(np.mean(preds[genuine] == 0)) if genuine.sum() > 0 else 0.0
        fmr  = float(np.mean(preds[impostor] == 1)) if impostor.sum() > 0 else 0.0

        tp = int(np.sum(preds[genuine] == 1))
        fp = int(np.sum(preds[impostor] == 1))
        fn = int(np.sum(preds[genuine] == 0))

        precision = tp / (tp + fp + 1e-8)
        recall    = tp / (tp + fn + 1e-8)
        f1        = 2 * precision * recall / (precision + recall + 1e-8)

        return fmr, fnmr, float(f1)
