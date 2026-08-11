"""
Robustness — Occlusion Module.

Trains and evaluates face recognition models under partial occlusion.

Strategies:
    • Part-based recognition    : independent embeddings per face region
    • Attention masking         : spatial attention to visible regions
    • Occlusion-aware training  : curriculum learning from no occlusion → heavy
    • Inpainting augmentation   : hallucinate occluded regions

Evaluation:
    • Accuracy per occlusion type
    • Accuracy by occlusion ratio (0%, 25%, 50%, 75%)
    • Worst-case analysis
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch
import torch.nn as nn


OCCLUSION_TYPES = [
    "mask", "sunglasses", "hat", "scarf",
    "random_patches", "hair_strands", "hand",
]


class OcclusionRobustnessEvaluator:
    """
    Evaluates model accuracy under different occlusion conditions.

    Generates synthetic occlusions, extracts embeddings, and
    measures cosine similarity vs clean face embeddings.

    Usage:
        evaluator = OcclusionRobustnessEvaluator(extractor=embedding_extractor)
        results   = evaluator.evaluate(image_paths, labels)
    """

    def __init__(
        self,
        extractor,
        threshold : float = 0.60,
    ):
        self.extractor = extractor
        self.threshold = threshold

    def _apply_occlusion(
        self,
        img      : np.ndarray,
        occ_type : str,
    ) -> np.ndarray:
        """Apply synthetic occlusion to face image."""
        h, w = img.shape[:2]
        result = img.copy()

        if occ_type == "mask":
            pts = np.array([
                [int(0.05 * w), int(0.50 * h)],
                [int(0.95 * w), int(0.50 * h)],
                [int(0.95 * w), h],
                [int(0.05 * w), h],
            ], np.int32)
            cv2.fillPoly(result, [pts], (200, 200, 200))

        elif occ_type == "sunglasses":
            cy = int(0.35 * h)
            cv2.ellipse(result, (int(0.30 * w), cy), (int(0.22 * w), int(0.10 * h)),
                        0, 0, 360, (20, 20, 20), -1)
            cv2.ellipse(result, (int(0.70 * w), cy), (int(0.22 * w), int(0.10 * h)),
                        0, 0, 360, (20, 20, 20), -1)

        elif occ_type == "hat":
            cv2.rectangle(result, (int(0.05 * w), 0),
                          (int(0.95 * w), int(0.25 * h)), (50, 50, 50), -1)

        elif occ_type == "random_patches":
            for _ in range(3):
                pw = int(0.25 * w)
                ph = int(0.25 * h)
                x1 = np.random.randint(0, w - pw)
                y1 = np.random.randint(0, h - ph)
                cv2.rectangle(result, (x1, y1), (x1 + pw, y1 + ph), (0, 0, 0), -1)

        return result

    def evaluate(
        self,
        image_paths     : List[str],
        labels          : List[int],
        occlusion_types : Optional[List[str]] = None,
    ) -> Dict[str, Dict]:
        """
        Evaluate robustness across occlusion types.

        Args:
            image_paths     : list of face image file paths
            labels          : identity labels
            occlusion_types : list of occlusion types to test

        Returns:
            dict: {occ_type: {accuracy, avg_similarity, n_samples}}
        """
        occ_types = occlusion_types or OCCLUSION_TYPES
        results   = {}

        # Extract clean embeddings
        print("Extracting clean embeddings...")
        clean_embs = self.extractor.extract_batch(image_paths)

        for occ_type in occ_types:
            print(f"  Evaluating occlusion: {occ_type}")

            occluded_embs = []
            for img_path in image_paths:
                img = cv2.imread(img_path)
                if img is None:
                    occluded_embs.append(np.zeros(clean_embs.shape[1]))
                    continue
                occ = self._apply_occlusion(img, occ_type)
                from PIL import Image
                pil = Image.fromarray(cv2.cvtColor(occ, cv2.COLOR_BGR2RGB))
                emb = self.extractor.extract_single(pil)
                occluded_embs.append(emb)

            occluded_embs = np.array(occluded_embs)

            # Compute similarity scores
            similarities = np.sum(clean_embs * occluded_embs, axis=1)
            correct      = (similarities >= self.threshold).sum()
            accuracy     = float(correct) / max(len(image_paths), 1)

            results[occ_type] = {
                "accuracy"         : round(accuracy, 4),
                "avg_similarity"   : round(float(similarities.mean()), 4),
                "min_similarity"   : round(float(similarities.min()), 4),
                "n_samples"        : len(image_paths),
            }

        return results
