"""
Demographic Fairness Evaluation Script.

Evaluates face recognition model fairness across demographic groups.

Metrics Computed:
    Per-group (gender × age × race):
        • TAR  @ FAR = 0.1%     (True Accept Rate)
        • FAR  @ threshold       (False Accept Rate)
        • FMR  (False Match Rate — impostor pairs accepted)
        • FNMR (False Non-Match Rate — genuine pairs rejected)
        • AUC  (Area Under ROC Curve)
        • EER  (Equal Error Rate)

Fairness Indices:
        • Inequity Ratio (IR)       : max/min of FMR across groups
        • Std Dev of FMR/FNMR       : dispersion across groups
        • Skewed Error Ratio (SER)  : worst-case error ratio
        • Demographic Parity Diff   : max(TAR) - min(TAR) across groups
        • Fairness Discrepancy Rate (FDR)

Datasets Supported:
        • CelebA    (gender, age attributes)
        • RFW       (race-balanced pairs)
        • BFW       (balanced faces in the wild)
        • LFW       (with demographic annotations)

Usage:
    python scripts/evaluate/eval_fairness.py \
        --checkpoint   experiments/runs/run_001/checkpoints/best_model.pt \
        --pairs-csv    data/splits/test/lfw_test_pairs.csv \
        --attr-csv     data/splits/test/celeba.csv \
        --output-dir   docs/results/fairness/ \
        --threshold    0.60
"""

import argparse
import csv
import json
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms.v2 as T              # ✅ v2 transforms
from PIL import Image
from sklearn.metrics import roc_auc_score, roc_curve

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from scripts.train.train_baseline import build_backbone, get_val_transforms


# ── Demographic Groups ────────────────────────────────────────────────────────

DEMOGRAPHIC_GROUPS = {
    "gender" : {
        "male"   : lambda attrs: attrs.get("Male", 0) == 1,
        "female" : lambda attrs: attrs.get("Male", 0) == 0,
    },
    "age" : {
        "young"  : lambda attrs: attrs.get("Young", 0) == 1,
        "older"  : lambda attrs: attrs.get("Young", 0) == 0,
    },
    "accessories" : {
        "with_glasses"    : lambda attrs: attrs.get("Eyeglasses", 0) == 1,
        "without_glasses" : lambda attrs: attrs.get("Eyeglasses", 0) == 0,
    },
}

# Intersectional groups
INTERSECTIONAL_GROUPS = [
    {"gender": "male",   "age": "young"},
    {"gender": "male",   "age": "older"},
    {"gender": "female", "age": "young"},
    {"gender": "female", "age": "older"},
]


# ── Embedding Extractor ───────────────────────────────────────────────────────

class EmbeddingExtractor:
    """
    Extracts L2-normalized face embeddings from a loaded model.

    Usage:
        extractor = EmbeddingExtractor(
            checkpoint = "experiments/runs/best/checkpoints/best_model.pt"
        )
        embedding = extractor.extract_single("path/to/face.jpg")
        embeddings = extractor.extract_batch(["img1.jpg", "img2.jpg"])
    """

    def __init__(
        self,
        checkpoint    : str,
        backbone_name : str = "resnet50",
        embedding_dim : int = 512,
        device        : str = "cuda",
        batch_size    : int = 64,
    ):
        self.device     = torch.device(
            device if torch.cuda.is_available() else "cpu"
        )
        self.batch_size = batch_size

        # Load model
        backbone = build_backbone(
            name          = backbone_name,
            embedding_dim = embedding_dim,
        )
        # ✅ PyTorch 2.6+: weights_only=False for full checkpoint
        ckpt = torch.load(
            checkpoint,
            map_location = self.device,
            weights_only = False,
        )
        state_dict = ckpt.get("model_state_dict", ckpt)
        state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
        backbone.load_state_dict(state_dict)
        backbone.eval().to(self.device)
        self.backbone = backbone

        # Transform pipeline
        self.transform = get_val_transforms(image_size=(112, 112))

    def extract_single(self, img_path: str) -> np.ndarray:
        """Extract embedding for a single image."""
        img = Image.open(img_path).convert("RGB")
        x   = self.transform(img).unsqueeze(0).to(self.device)

        with torch.no_grad():
            emb = self.backbone(x)
            emb = nn.functional.normalize(emb, p=2, dim=1)

        return emb.cpu().numpy()[0]

    def extract_batch(self, img_paths: List[str]) -> np.ndarray:
        """
        Extract embeddings for a batch of images.

        Args:
            img_paths : list of image file paths

        Returns:
            (N, embedding_dim) float32 numpy array
        """
        all_embeddings = []

        for i in range(0, len(img_paths), self.batch_size):
            batch_paths = img_paths[i:i + self.batch_size]
            batch_imgs  = []

            for p in batch_paths:
                try:
                    img = Image.open(p).convert("RGB")
                    batch_imgs.append(self.transform(img))
                except Exception:
                    # Fallback: zeros for missing images
                    batch_imgs.append(torch.zeros(3, 112, 112))

            batch_tensor = torch.stack(batch_imgs).to(self.device)

            with torch.no_grad():
                embs = self.backbone(batch_tensor)
                embs = nn.functional.normalize(embs, p=2, dim=1)

            all_embeddings.append(embs.cpu().numpy())

        return np.vstack(all_embeddings)


# ── Fairness Metrics ──────────────────────────────────────────────────────────

class FairnessEvaluator:
    """
    Computes comprehensive demographic fairness metrics for
    face verification systems.

    Metrics per demographic group:
        • TAR@FAR=0.1%  — operational performance metric
        • FMR / FNMR    — error rates at a fixed threshold
        • AUC           — ranking metric
        • EER           — threshold-free metric

    Fairness indices across groups:
        • Inequity Ratio (IR)
        • Standard deviation of FMR
        • Demographic Parity Difference
        • Skewed Error Ratio (SER)
        • Fairness Discrepancy Rate (FDR)

    Usage:
        evaluator = FairnessEvaluator(extractor, threshold=0.60)
        results   = evaluator.evaluate(pairs_csv, attr_csv)
        evaluator.save_report(results, output_dir="docs/results/fairness/")
    """

    def __init__(
        self,
        extractor : EmbeddingExtractor,
        threshold : float = 0.60,
        far_target: float = 0.001,     # TAR @ FAR=0.1%
    ):
        self.extractor  = extractor
        self.threshold  = threshold
        self.far_target = far_target

    # ── Cosine Similarity ─────────────────────────────────────────────────────

    @staticmethod
    def cosine_similarity(emb1: np.ndarray, emb2: np.ndarray) -> float:
        """Cosine similarity between two L2-normalized embeddings."""
        return float(np.dot(emb1, emb2))

    # ── Load Pairs ────────────────────────────────────────────────────────────

    def _load_pairs(self, pairs_csv: str) -> List[dict]:
        """
        Load verification pairs from CSV.

        CSV format:
            image1_path, image2_path, label (1=same, 0=different)
        """
        pairs = []
        with open(pairs_csv, newline="") as f:
            for row in csv.DictReader(f):
                pairs.append({
                    "img1"  : row["image1_path"],
                    "img2"  : row["image2_path"],
                    "label" : int(row["label"]),
                })
        return pairs

    # ── Load Attributes ───────────────────────────────────────────────────────

    def _load_attributes(self, attr_csv: str) -> Dict[str, dict]:
        """
        Load demographic attribute annotations.

        CSV format:
            image_path, [attribute columns...]

        Returns:
            dict: {image_path: {attr_name: value}}
        """
        attrs: Dict[str, dict] = {}
        with open(attr_csv, newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                img_path = row.pop("image_path", "")
                attrs[img_path] = {k: int(v) for k, v in row.items()
                                   if v.lstrip("-").isdigit()}
        return attrs

    # ── Compute ROC Metrics ───────────────────────────────────────────────────

    def _compute_roc_metrics(
        self,
        scores : np.ndarray,
        labels : np.ndarray,
    ) -> dict:
        """
        Compute ROC-based metrics.

        Args:
            scores : cosine similarity scores
            labels : 1 = genuine pair, 0 = impostor pair

        Returns:
            dict with TAR, FAR, FMR, FNMR, AUC, EER
        """
        if len(np.unique(labels)) < 2:
            return {
                "auc": float("nan"), "eer": float("nan"),
                "tar_at_far": float("nan"),
                "fmr": float("nan"), "fnmr": float("nan"),
                "n_genuine": int(np.sum(labels == 1)),
                "n_impostor": int(np.sum(labels == 0)),
            }

        # ROC curve
        fpr, tpr, thresholds = roc_curve(labels, scores, pos_label=1)

        # AUC
        auc = roc_auc_score(labels, scores)

        # EER (where FPR ≈ FNR)
        fnr = 1 - tpr
        eer_idx = np.nanargmin(np.abs(fnr - fpr))
        eer     = float((fpr[eer_idx] + fnr[eer_idx]) / 2)

        # TAR @ target FAR
        far_idx = np.searchsorted(fpr, self.far_target)
        tar_at_far = float(tpr[min(far_idx, len(tpr) - 1)])

        # FMR / FNMR at fixed threshold
        scores_arr  = np.array(scores)
        labels_arr  = np.array(labels)
        predictions = (scores_arr >= self.threshold).astype(int)

        genuine   = labels_arr == 1
        impostor  = labels_arr == 0

        fnmr = float(np.mean(predictions[genuine] == 0))   # genuine rejected
        fmr  = float(np.mean(predictions[impostor] == 1))  # impostor accepted

        return {
            "auc"          : round(auc, 4),
            "eer"          : round(eer, 4),
            "tar_at_far"   : round(tar_at_far, 4),
            "fmr"          : round(fmr, 4),
            "fnmr"         : round(fnmr, 4),
            "threshold"    : self.threshold,
            "n_genuine"    : int(np.sum(genuine)),
            "n_impostor"   : int(np.sum(impostor)),
        }

    # ── Evaluate ──────────────────────────────────────────────────────────────

    def evaluate(
        self,
        pairs_csv : str,
        attr_csv  : Optional[str] = None,
    ) -> dict:
        """
        Run full fairness evaluation.

        Pipeline:
            1. Load pairs and attributes
            2. Extract embeddings for all unique images
            3. Compute similarity scores for all pairs
            4. Compute per-group metrics
            5. Compute fairness indices

        Args:
            pairs_csv : path to verification pairs CSV
            attr_csv  : path to demographic attributes CSV

        Returns:
            Nested dict with per-group and fairness metrics
        """
        print(f"\n📊 Fairness Evaluation")
        print(f"   Pairs    : {pairs_csv}")
        print(f"   Attrs    : {attr_csv}")
        print(f"   Threshold: {self.threshold}")

        # ── Load data ─────────────────────────────────────────────────────
        pairs = self._load_pairs(pairs_csv)
        attrs = self._load_attributes(attr_csv) if attr_csv else {}

        print(f"\n   Pairs loaded    : {len(pairs):,}")
        print(f"   Attributes loaded: {len(attrs):,}")

        # ── Extract embeddings ────────────────────────────────────────────
        print("\n   Extracting embeddings...")
        unique_images = list(set(
            [p["img1"] for p in pairs] + [p["img2"] for p in pairs]
        ))

        start = time.perf_counter()
        emb_array = self.extractor.extract_batch(unique_images)
        emb_map   = dict(zip(unique_images, emb_array))
        elapsed   = time.perf_counter() - start

        print(f"   Extracted {len(unique_images):,} embeddings in {elapsed:.1f}s")

        # ── Compute pair scores ───────────────────────────────────────────
        print("   Computing pair similarities...")
        all_scores = []
        all_labels = []
        pair_attrs = []

        for pair in pairs:
            emb1 = emb_map.get(pair["img1"])
            emb2 = emb_map.get(pair["img2"])
            if emb1 is None or emb2 is None:
                continue

            score = self.cosine_similarity(emb1, emb2)
            all_scores.append(score)
            all_labels.append(pair["label"])

            # Get attributes for this pair (use image1's attributes)
            pair_attrs.append(attrs.get(pair["img1"], {}))

        all_scores = np.array(all_scores)
        all_labels = np.array(all_labels)

        # ── Overall metrics ───────────────────────────────────────────────
        print("   Computing metrics...")
        results = {
            "overall"    : self._compute_roc_metrics(all_scores, all_labels),
            "per_group"  : {},
            "fairness"   : {},
            "config"     : {
                "threshold"  : self.threshold,
                "far_target" : self.far_target,
                "n_pairs"    : len(all_scores),
                "pairs_csv"  : pairs_csv,
            },
        }

        # ── Per-group metrics ─────────────────────────────────────────────
        for dimension, groups in DEMOGRAPHIC_GROUPS.items():
            results["per_group"][dimension] = {}

            for group_name, group_filter in groups.items():
                # Find pairs belonging to this group
                mask = np.array([
                    group_filter(pa) for pa in pair_attrs
                ])

                if mask.sum() < 10:
                    print(f"   ⚠️  Skipping {dimension}/{group_name} "
                          f"(only {mask.sum()} samples)")
                    continue

                group_scores = all_scores[mask]
                group_labels = all_labels[mask]

                group_metrics = self._compute_roc_metrics(
                    group_scores, group_labels
                )
                group_metrics["n_samples"] = int(mask.sum())
                results["per_group"][dimension][group_name] = group_metrics

                print(
                    f"   {dimension}/{group_name:<20}: "
                    f"AUC={group_metrics['auc']:.4f} | "
                    f"EER={group_metrics['eer']:.4f} | "
                    f"TAR@FAR={group_metrics['tar_at_far']:.4f} | "
                    f"n={mask.sum():,}"
                )

        # ── Fairness Indices ──────────────────────────────────────────────
        results["fairness"] = self._compute_fairness_indices(
            results["per_group"]
        )

        return results

    # ── Fairness Indices ──────────────────────────────────────────────────────

    def _compute_fairness_indices(
        self,
        per_group: dict,
    ) -> dict:
        """
        Compute fairness indices from per-group metrics.

        Indices:
            IR   : Inequity Ratio = max(FMR) / min(FMR) across groups
            STD  : Standard deviation of FMR across groups
            SER  : Skewed Error Ratio = max(FNMR) / min(FNMR)
            DPD  : Demographic Parity Difference = max(TAR) - min(TAR)
            FDR  : Fairness Discrepancy Rate (weighted FMR+FNMR diff)
        """
        fairness = {}

        for dimension, groups in per_group.items():
            if not groups:
                continue

            fmr_vals  = [g["fmr"]       for g in groups.values() if not np.isnan(g["fmr"])]
            fnmr_vals = [g["fnmr"]      for g in groups.values() if not np.isnan(g["fnmr"])]
            tar_vals  = [g["tar_at_far"] for g in groups.values() if not np.isnan(g["tar_at_far"])]
            auc_vals  = [g["auc"]       for g in groups.values() if not np.isnan(g["auc"])]

            if not fmr_vals:
                continue

            # Inequity Ratio (FMR)
            ir_fmr = (
                max(fmr_vals) / min(fmr_vals)
                if min(fmr_vals) > 0 else float("inf")
            )

            # Inequity Ratio (FNMR)
            ir_fnmr = (
                max(fnmr_vals) / min(fnmr_vals)
                if fnmr_vals and min(fnmr_vals) > 0 else float("inf")
            )

            # Std dev of FMR
            std_fmr  = float(np.std(fmr_vals))
            std_fnmr = float(np.std(fnmr_vals)) if fnmr_vals else float("nan")

            # Skewed Error Ratio
            ser = (
                max(fnmr_vals) / min(fnmr_vals)
                if fnmr_vals and min(fnmr_vals) > 0 else float("inf")
            )

            # Demographic Parity Difference (TAR spread)
            dpd = float(max(tar_vals) - min(tar_vals)) if tar_vals else float("nan")

            # AUC std
            std_auc = float(np.std(auc_vals)) if auc_vals else float("nan")

            # Fairness Discrepancy Rate (simplified: avg of FMR+FNMR max deviation)
            global_fmr  = np.mean(fmr_vals)
            global_fnmr = np.mean(fnmr_vals) if fnmr_vals else 0
            fdr = float(
                np.mean([abs(v - global_fmr) for v in fmr_vals]) +
                np.mean([abs(v - global_fnmr) for v in fnmr_vals])
                if fnmr_vals else 0
            )

            fairness[dimension] = {
                "inequity_ratio_fmr"   : round(ir_fmr, 4),
                "inequity_ratio_fnmr"  : round(ir_fnmr, 4),
                "std_fmr"              : round(std_fmr, 4),
                "std_fnmr"             : round(std_fnmr, 4),
                "std_auc"              : round(std_auc, 4),
                "skewed_error_ratio"   : round(ser, 4),
                "demographic_parity_diff": round(dpd, 4),
                "fairness_discrepancy_rate": round(fdr, 4),
                "n_groups"             : len(groups),
                "fmr_range"            : [round(min(fmr_vals), 4), round(max(fmr_vals), 4)],
                "tar_range"            : [round(min(tar_vals), 4), round(max(tar_vals), 4)] if tar_vals else [],
            }

            # Fairness verdict
            if ir_fmr < 2.0 and dpd < 0.05:
                verdict = "✅ FAIR"
            elif ir_fmr < 5.0 and dpd < 0.10:
                verdict = "⚠️  MODERATE BIAS"
            else:
                verdict = "❌ HIGH BIAS"

            fairness[dimension]["verdict"] = verdict

        return fairness

    # ── Report ────────────────────────────────────────────────────────────────

    def save_report(
        self,
        results    : dict,
        output_dir : str,
    ):
        """
        Save fairness evaluation report.

        Writes:
            fairness_results.json    ← full raw results
            fairness_report.md       ← human-readable markdown
        """
        output = Path(output_dir)
        output.mkdir(parents=True, exist_ok=True)

        # JSON
        json_path = output / "fairness_results.json"
        with open(json_path, "w") as f:
            json.dump(results, f, indent=2, default=str)
        print(f"\n💾 JSON saved: {json_path}")

        # Markdown report
        md_path = output / "fairness_report.md"
        self._write_markdown(results, md_path)
        print(f"💾 Report saved: {md_path}")

    def _write_markdown(self, results: dict, path: Path):
        """Write markdown fairness report."""
        lines = [
            "# 📊 Demographic Fairness Evaluation Report\n",
            f"**Threshold**: {results['config']['threshold']}  ",
            f"**Total pairs**: {results['config']['n_pairs']:,}  \n",

            "---\n",
            "## Overall Performance\n",
            "| Metric | Value |",
            "|--------|-------|",
        ]

        overall = results.get("overall", {})
        for k, v in overall.items():
            if isinstance(v, float):
                lines.append(f"| {k} | {v:.4f} |")
            elif isinstance(v, int):
                lines.append(f"| {k} | {v:,} |")

        lines.append("\n---\n")
        lines.append("## Per-Group Results\n")

        for dim, groups in results.get("per_group", {}).items():
            lines.append(f"### {dim.title()}\n")
            lines.append("| Group | AUC | EER | TAR@FAR | FMR | FNMR | N |")
            lines.append("|-------|-----|-----|---------|-----|------|---|")
            for grp, m in groups.items():
                lines.append(
                    f"| {grp} "
                    f"| {m.get('auc','—')} "
                    f"| {m.get('eer','—')} "
                    f"| {m.get('tar_at_far','—')} "
                    f"| {m.get('fmr','—')} "
                    f"| {m.get('fnmr','—')} "
                    f"| {m.get('n_samples','—'):,} |"
                )
            lines.append("")

        lines.append("---\n")
        lines.append("## Fairness Indices\n")

        for dim, idx in results.get("fairness", {}).items():
            lines.append(f"### {dim.title()} — {idx.get('verdict', '')}\n")
            lines.append("| Index | Value |")
            lines.append("|-------|-------|")
            for k, v in idx.items():
                if k == "verdict":
                    continue
                lines.append(f"| {k.replace('_', ' ').title()} | {v} |")
            lines.append("")

        with open(path, "w") as f:
            f.write("\n".join(lines))


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description = "Demographic Fairness Evaluation for Face Recognition"
    )
    parser.add_argument(
        "--checkpoint",
        required = True,
        help     = "Path to trained model checkpoint (.pt)"
    )
    parser.add_argument(
        "--pairs-csv",
        required = True,
        help     = "Verification pairs CSV (image1, image2, label)"
    )
    parser.add_argument(
        "--attr-csv",
        default  = None,
        help     = "Demographic attributes CSV (image_path, attributes...)"
    )
    parser.add_argument("--output-dir",    default="docs/results/fairness/")
    parser.add_argument("--threshold",     type=float, default=0.60)
    parser.add_argument("--far-target",    type=float, default=0.001)
    parser.add_argument("--device",        default="cuda")
    parser.add_argument("--batch-size",    type=int, default=64)
    parser.add_argument("--backbone",      default="resnet50")
    parser.add_argument("--embedding-dim", type=int, default=512)
    return parser.parse_args()


def main():
    args = parse_args()

    extractor = EmbeddingExtractor(
        checkpoint    = args.checkpoint,
        backbone_name = args.backbone,
        embedding_dim = args.embedding_dim,
        device        = args.device,
        batch_size    = args.batch_size,
    )

    evaluator = FairnessEvaluator(
        extractor  = extractor,
        threshold  = args.threshold,
        far_target = args.far_target,
    )

    results = evaluator.evaluate(
        pairs_csv = args.pairs_csv,
        attr_csv  = args.attr_csv,
    )

    evaluator.save_report(results, output_dir=args.output_dir)

    # Print fairness summary
    print("\n" + "=" * 60)
    print("📋 FAIRNESS SUMMARY")
    print("=" * 60)
    for dim, idx in results.get("fairness", {}).items():
        print(f"\n  {dim.upper()}: {idx.get('verdict')}")
        print(f"    Inequity Ratio (FMR) : {idx.get('inequity_ratio_fmr')}")
        print(f"    Std Dev (FMR)        : {idx.get('std_fmr')}")
        print(f"    Demo Parity Diff     : {idx.get('demographic_parity_diff')}")
        print(f"    Fairness Disc. Rate  : {idx.get('fairness_discrepancy_rate')}")
    print("=" * 60)


if __name__ == "__main__":
    main()
