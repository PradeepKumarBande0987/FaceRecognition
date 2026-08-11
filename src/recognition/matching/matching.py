"""
Recognition — Matching Module.

Implements face matching strategies for verification and identification.

Matching Strategies:
    • cosine_similarity   : L2-normalized dot product (default)
    • euclidean_distance  : L2 distance in embedding space
    • faiss_flat          : exact FAISS search (GPU-accelerated)
    • faiss_ivf           : approximate FAISS IVF for large databases

Tasks:
    • 1:1 Verification  : are these the same person?
    • 1:N Identification: who is this person?
"""

from __future__ import annotations

from enum import Enum
from typing import Dict, List, Optional, Tuple

import numpy as np


# ── Metric Types ──────────────────────────────────────────────────────────────

class MatchMetric(str, Enum):
    COSINE      = "cosine"
    EUCLIDEAN   = "euclidean"
    FAISS_FLAT  = "faiss_flat"
    FAISS_IVF   = "faiss_ivf"


# ── Match Result ──────────────────────────────────────────────────────────────

class MatchResult:
    """Result from a face matching operation."""

    def __init__(
        self,
        identity_id      : Optional[str],
        similarity_score : float,
        is_match         : bool,
        threshold_used   : float,
        metric           : str,
        top_k_matches    : Optional[List[Dict]] = None,
    ):
        self.identity_id      = identity_id
        self.similarity_score = similarity_score
        self.is_match         = is_match
        self.threshold_used   = threshold_used
        self.metric           = metric
        self.top_k_matches    = top_k_matches or []

    def to_dict(self) -> dict:
        return {
            "identity_id"     : self.identity_id,
            "similarity_score": self.similarity_score,
            "is_match"        : self.is_match,
            "threshold_used"  : self.threshold_used,
            "metric"          : self.metric,
        }


# ── Face Matcher ──────────────────────────────────────────────────────────────

class FaceMatcher:
    """
    Implements face verification (1:1) and identification (1:N).

    Supports cosine similarity, Euclidean distance,
    and FAISS-accelerated approximate search.

    Usage:
        matcher = FaceMatcher(metric="cosine", threshold=0.60)

        # 1:1 Verification
        result = matcher.verify(emb_probe, emb_reference)

        # 1:N Identification
        results = matcher.identify(emb_probe, database, top_k=5)
    """

    def __init__(
        self,
        metric    : str   = "cosine",
        threshold : float = 0.60,
    ):
        self.metric    = MatchMetric(metric)
        self.threshold = threshold
        self._faiss_index = None

    # ── Similarity ────────────────────────────────────────────────────────────

    def cosine_similarity(
        self,
        emb1 : np.ndarray,
        emb2 : np.ndarray,
    ) -> float:
        """
        Cosine similarity between two L2-normalized embeddings.

        Both embeddings should be L2-normalized (unit vectors).
        Cosine similarity = dot product for unit vectors.
        """
        norm1 = np.linalg.norm(emb1)
        norm2 = np.linalg.norm(emb2)
        if norm1 < 1e-8 or norm2 < 1e-8:
            return 0.0
        return float(np.dot(emb1 / norm1, emb2 / norm2))

    def euclidean_distance(
        self,
        emb1 : np.ndarray,
        emb2 : np.ndarray,
    ) -> float:
        """L2 Euclidean distance (lower = more similar)."""
        return float(np.linalg.norm(emb1 - emb2))

    def similarity(self, emb1: np.ndarray, emb2: np.ndarray) -> float:
        """Compute similarity using configured metric."""
        if self.metric == MatchMetric.COSINE:
            return self.cosine_similarity(emb1, emb2)
        elif self.metric == MatchMetric.EUCLIDEAN:
            dist = self.euclidean_distance(emb1, emb2)
            return 1.0 / (1.0 + dist)
        return self.cosine_similarity(emb1, emb2)

    # ── 1:1 Verification ──────────────────────────────────────────────────────

    def verify(
        self,
        emb_probe     : np.ndarray,
        emb_reference : np.ndarray,
        threshold     : Optional[float] = None,
    ) -> MatchResult:
        """
        1:1 Face Verification.

        Args:
            emb_probe     : (512,) embedding of probe face
            emb_reference : (512,) embedding of reference face
            threshold     : override default threshold

        Returns:
            MatchResult with is_match decision
        """
        thresh = threshold or self.threshold
        score  = self.similarity(emb_probe, emb_reference)

        return MatchResult(
            identity_id      = None,
            similarity_score = round(score, 4),
            is_match         = score >= thresh,
            threshold_used   = thresh,
            metric           = self.metric.value,
        )

    # ── 1:N Identification ────────────────────────────────────────────────────

    def identify(
        self,
        emb_probe   : np.ndarray,
        database    : Dict[str, np.ndarray],
        top_k       : int   = 5,
        threshold   : Optional[float] = None,
    ) -> List[MatchResult]:
        """
        1:N Face Identification via brute-force cosine search.

        Args:
            emb_probe : (512,) probe embedding
            database  : {identity_id: embedding} dict
            top_k     : number of candidates to return
            threshold : minimum similarity score

        Returns:
            Top-K MatchResult sorted by similarity (descending)
        """
        thresh = threshold or self.threshold
        if not database:
            return []

        ids   = list(database.keys())
        embs  = np.stack(list(database.values()))          # (N, 512)
        probe = emb_probe / (np.linalg.norm(emb_probe) + 1e-8)
        scores = embs @ probe                              # cosine similarity

        top_idx = np.argsort(scores)[::-1][:top_k]
        results = []

        for idx in top_idx:
            score = float(scores[idx])
            if score < thresh:
                break
            results.append(MatchResult(
                identity_id      = ids[idx],
                similarity_score = round(score, 4),
                is_match         = score >= thresh,
                threshold_used   = thresh,
                metric           = self.metric.value,
            ))

        return results

    # ── FAISS Search ──────────────────────────────────────────────────────────

    def build_faiss_index(
        self,
        embeddings : np.ndarray,
        use_gpu    : bool = False,
    ):
        """
        Build FAISS flat index for fast approximate search.

        Args:
            embeddings : (N, D) float32 database embeddings
            use_gpu    : move index to GPU

        Note: requires pip install faiss-cpu (or faiss-gpu)
        """
        try:
            import faiss
        except ImportError:
            raise ImportError(
                "FAISS not installed.\n"
                "Install: pip install faiss-cpu  (or faiss-gpu)"
            )

        dim = embeddings.shape[1]
        index = faiss.IndexFlatIP(dim)    # Inner product = cosine for normalized embs

        # Normalize
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        embs_norm = embeddings / (norms + 1e-8)
        embs_norm = embs_norm.astype(np.float32)

        if use_gpu:
            res   = faiss.StandardGpuResources()
            index = faiss.index_cpu_to_gpu(res, 0, index)

        index.add(embs_norm)
        self._faiss_index = index
        print(f"✅ FAISS index built: {index.ntotal:,} vectors (dim={dim})")

    def faiss_search(
        self,
        query      : np.ndarray,
        top_k      : int = 5,
        ids        : Optional[List[str]] = None,
    ) -> List[dict]:
        """
        Search FAISS index for nearest neighbors.

        Args:
            query : (512,) float32 query embedding
            top_k : number of neighbors
            ids   : list of identity IDs in same order as index

        Returns:
            List of {identity_id, score} dicts
        """
        if self._faiss_index is None:
            raise RuntimeError("FAISS index not built. Call build_faiss_index() first.")

        q = query / (np.linalg.norm(query) + 1e-8)
        q = q.astype(np.float32).reshape(1, -1)

        scores, indices = self._faiss_index.search(q, top_k)
        scores  = scores[0]
        indices = indices[0]

        results = []
        for score, idx in zip(scores, indices):
            if idx < 0:
                continue
            identity_id = ids[idx] if ids and idx < len(ids) else str(idx)
            results.append({
                "identity_id"     : identity_id,
                "similarity_score": round(float(score), 4),
            })
        return results
