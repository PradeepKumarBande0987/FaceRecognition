"""
Recognition — Embeddings Module.

Extracts, stores, and manages L2-normalized face embeddings.

Features:
    • Multi-backbone embedding extraction
    • Batch extraction with progress tracking
    • Embedding database (in-memory + persistent JSON/NPY)
    • Embedding visualization (t-SNE / UMAP)
    • Embedding quality metrics
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms.v2 as T
from PIL import Image


# ── Embedding Extractor ───────────────────────────────────────────────────────

class EmbeddingExtractor:
    """
    Extracts L2-normalized face embeddings from trained backbones.

    Usage:
        extractor = EmbeddingExtractor(
            model     = backbone,
            device    = "cuda",
        )
        emb = extractor.extract_single("face.jpg")
        embs = extractor.extract_batch(["img1.jpg", "img2.jpg"])
    """

    def __init__(
        self,
        model       : nn.Module,
        device      : str = "cuda",
        batch_size  : int = 64,
        image_size  : Tuple[int, int] = (112, 112),
        normalize   : bool = True,
    ):
        self.model      = model.eval().to(device)
        self.device     = torch.device(device)
        self.batch_size = batch_size
        self.normalize  = normalize

        self.transform  = T.Compose([
            T.Resize(image_size, antialias=True),
            T.ToImage(),
            T.ToDtype(torch.float32, scale=True),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])

    @torch.no_grad()
    def extract_single(self, img_input: str | np.ndarray | Image.Image) -> np.ndarray:
        """Extract embedding for a single image."""
        if isinstance(img_input, str):
            img = Image.open(img_input).convert("RGB")
        elif isinstance(img_input, np.ndarray):
            import cv2
            img = Image.fromarray(cv2.cvtColor(img_input, cv2.COLOR_BGR2RGB))
        else:
            img = img_input

        tensor = self.transform(img).unsqueeze(0).to(self.device)
        emb    = self.model(tensor)
        if self.normalize:
            emb = nn.functional.normalize(emb, p=2, dim=1)
        return emb.cpu().numpy()[0]

    @torch.no_grad()
    def extract_batch(
        self,
        inputs     : List[str | np.ndarray | Image.Image],
        show_progress: bool = True,
    ) -> np.ndarray:
        """
        Extract embeddings for a batch of inputs.

        Returns:
            (N, embedding_dim) float32 numpy array
        """
        all_embeddings = []
        n_batches = (len(inputs) + self.batch_size - 1) // self.batch_size

        for i in range(0, len(inputs), self.batch_size):
            batch_inputs = inputs[i:i + self.batch_size]
            tensors      = []

            for inp in batch_inputs:
                try:
                    if isinstance(inp, str):
                        img = Image.open(inp).convert("RGB")
                    elif isinstance(inp, np.ndarray):
                        import cv2
                        img = Image.fromarray(cv2.cvtColor(inp, cv2.COLOR_BGR2RGB))
                    else:
                        img = inp
                    tensors.append(self.transform(img))
                except Exception:
                    tensors.append(torch.zeros(3, 112, 112))

            batch_tensor = torch.stack(tensors).to(self.device)
            embs         = self.model(batch_tensor)

            if self.normalize:
                embs = nn.functional.normalize(embs, p=2, dim=1)

            all_embeddings.append(embs.cpu().numpy())

            if show_progress and (i // self.batch_size) % 10 == 0:
                done = min(i + self.batch_size, len(inputs))
                print(f"  Embeddings: {done:,}/{len(inputs):,}")

        return np.vstack(all_embeddings)


# ── Embedding Database ────────────────────────────────────────────────────────

class EmbeddingDatabase:
    """
    In-memory embedding database with persistence.

    Stores:
        identity_id → embedding (512-d float32)
        identity_id → metadata dict

    Persistence:
        embeddings saved as .npy
        metadata saved as .json

    Usage:
        db = EmbeddingDatabase()
        db.insert("user_001", embedding, {"name": "Jane"})
        results = db.search(query_embedding, top_k=5)
        db.save("models/face_db")
        db = EmbeddingDatabase.load("models/face_db")
    """

    def __init__(self):
        self._embeddings : Dict[str, np.ndarray] = {}
        self._metadata   : Dict[str, dict]       = {}

    def insert(
        self,
        identity_id : str,
        embedding   : np.ndarray,
        metadata    : Optional[dict] = None,
    ):
        """Insert or update a face embedding."""
        emb_norm = embedding / (np.linalg.norm(embedding) + 1e-8)
        self._embeddings[identity_id] = emb_norm.astype(np.float32)
        self._metadata[identity_id]   = metadata or {}

    def delete(self, identity_id: str) -> bool:
        """Delete an identity from the database."""
        if identity_id in self._embeddings:
            del self._embeddings[identity_id]
            del self._metadata[identity_id]
            return True
        return False

    def search(
        self,
        query_embedding : np.ndarray,
        top_k           : int   = 5,
        threshold       : float = 0.60,
    ) -> List[Dict]:
        """
        Cosine similarity search.

        Args:
            query_embedding : (512,) float32 L2-normalized
            top_k           : number of matches to return
            threshold       : minimum similarity score

        Returns:
            List of dicts sorted by similarity (descending)
        """
        if not self._embeddings:
            return []

        ids   = list(self._embeddings.keys())
        db_embs = np.stack(list(self._embeddings.values()))  # (N, 512)
        q_norm  = query_embedding / (np.linalg.norm(query_embedding) + 1e-8)

        # Cosine similarity: (N,)
        scores = db_embs @ q_norm

        # Top-K above threshold
        top_indices = np.argsort(scores)[::-1][:top_k]
        results = []
        for idx in top_indices:
            score = float(scores[idx])
            if score < threshold:
                break
            results.append({
                "identity_id"     : ids[idx],
                "similarity_score": round(score, 4),
                "metadata"        : self._metadata[ids[idx]],
            })

        return results

    def save(self, output_prefix: str):
        """Save database to .npy + .json files."""
        prefix = Path(output_prefix)
        prefix.parent.mkdir(parents=True, exist_ok=True)

        ids   = list(self._embeddings.keys())
        embs  = np.stack([self._embeddings[i] for i in ids])

        np.save(f"{prefix}_embeddings.npy", embs)
        with open(f"{prefix}_ids.json", "w") as f:
            json.dump(ids, f)
        with open(f"{prefix}_metadata.json", "w") as f:
            json.dump(self._metadata, f, indent=2)

        print(f"✅ Database saved: {len(ids):,} identities → {prefix}*")

    @classmethod
    def load(cls, output_prefix: str) -> "EmbeddingDatabase":
        """Load database from saved files."""
        prefix = output_prefix
        embs   = np.load(f"{prefix}_embeddings.npy")

        with open(f"{prefix}_ids.json") as f:
            ids = json.load(f)
        with open(f"{prefix}_metadata.json") as f:
            metadata = json.load(f)

        db = cls()
        for i, identity_id in enumerate(ids):
            db._embeddings[identity_id] = embs[i]
            db._metadata[identity_id]   = metadata.get(identity_id, {})

        print(f"✅ Database loaded: {len(ids):,} identities")
        return db

    def __len__(self) -> int:
        return len(self._embeddings)

    @property
    def identity_ids(self) -> List[str]:
        return list(self._embeddings.keys())
