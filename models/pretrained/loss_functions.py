"""
Loss functions for face recognition pre-training.

Supported losses:
    • ArcFace (Additive Angular Margin)
    • CosFace (Large Margin Cosine Loss)
    • SphereFace (Angular Softmax)
    • Triplet Loss (Distance-based)
    • Combined losses (ArcFace + Triplet)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional
import math


# ── ArcFace Loss ──────────────────────────────────────────────────────────────

class ArcFaceLoss(nn.Module):
    """
    ArcFace: Additive Angular Margin Loss for Deep Face Recognition.

    Paper: ArcFace: Additive Angular Margin Loss for Deep Face Recognition
    Link: https://arxiv.org/abs/1801.07698

    Key idea: Add angular margin in embedding space instead of feature space.

    Loss = -log(
        exp(s * cos(θ + m))
        ────────────────────────────────────────────
        exp(s * cos(θ + m)) + Σ exp(s * cos(θ_j))
    )

    where:
        θ   = angle between embedding and weight vector
        m   = angular margin (typically 0.5 radians ≈ 28.6°)
        s   = feature scale (typically 64)
        θ_j = angles for other classes

    Effect:
        • Enforces large angular margin between identity classes
        • Embeddings become highly separated
        • Improves generalization and robustness

    Hyperparameters:
        • margin (m): 0.4-0.6 (default 0.5)
        • scale (s): 32-64 (default 64)
            - Smaller s: softer decision boundaries
            - Larger s: sharper boundaries (may cause training instability)
    """

    def __init__(
        self,
        num_classes: int,
        embedding_dim: int = 512,
        margin: float = 0.5,
        scale: float = 64.0,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.embedding_dim = embedding_dim
        self.margin = margin
        self.scale = scale

        # Weight matrix: [num_classes, embedding_dim]
        # Stores normalized weight vectors for each class
        self.weight = nn.Parameter(
            torch.FloatTensor(num_classes, embedding_dim)
        )
        nn.init.xavier_uniform_(self.weight)

    def forward(
        self,
        embeddings: torch.Tensor,  # [B, embedding_dim] normalized
        labels: torch.Tensor,      # [B]
    ) -> torch.Tensor:
        """
        Args:
            embeddings: [B, embedding_dim] L2-normalized embeddings
            labels    : [B] class labels (identity)

        Returns:
            loss: scalar
        """
        # Normalize weight vectors
        W = F.normalize(self.weight, p=2, dim=1)

        # Compute cosine similarity: [B, num_classes]
        logits = torch.matmul(embeddings, W.t())

        # Compute angles
        theta = torch.acos(torch.clamp(logits, -1.0, 1.0))

        # Add angular margin only to target class
        target_logit = logits.scatter(
            1, labels.view(-1, 1), 0
        )  # zero out target

        # Compute target with margin
        theta_m = theta.scatter(
            1, labels.view(-1, 1),
            torch.acos(
                torch.clamp(
                    torch.cos(
                        torch.acos(
                            torch.clamp(
                                logits.gather(1, labels.view(-1, 1)),
                                -1.0, 1.0
                            )
                        ) + self.margin
                    ),
                    -1.0, 1.0
                )
            )
        )

        # Compute output logits: s * cos(θ + m)
        output = self.scale * torch.cos(theta_m)

        # Cross-entropy loss
        loss = F.cross_entropy(output, labels)

        return loss


# ── CosFace Loss ──────────────────────────────────────────────────────────────

class CosFaceLoss(nn.Module):
    """
    CosFace: Large Margin Cosine Loss for Deep Face Recognition.

    Paper: CosFace: Large Margin Cosine Loss for Deep Face Recognition
    Link: https://arxiv.org/abs/1801.09414

    Key idea: Subtract margin from cosine similarity (large margin cosine loss).

    Loss = -log(
        exp(s * (cos(θ) - m))
        ─────────────────────────────────────────────
        exp(s * (cos(θ) - m)) + Σ exp(s * cos(θ_j))
    )

    Difference from ArcFace:
        • ArcFace: margin in angular space (θ + m)
        • CosFace: margin in cosine space (cos(θ) - m)
        • Both achieve similar results, CosFace is slightly simpler

    Hyperparameters:
        • margin (m): 0.25-0.35 (default 0.35)
        • scale (s): 64 (default)
    """

    def __init__(
        self,
        num_classes: int,
        embedding_dim: int = 512,
        margin: float = 0.35,
        scale: float = 64.0,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.embedding_dim = embedding_dim
        self.margin = margin
        self.scale = scale

        self.weight = nn.Parameter(
            torch.FloatTensor(num_classes, embedding_dim)
        )
        nn.init.xavier_uniform_(self.weight)

    def forward(
        self,
        embeddings: torch.Tensor,
        labels: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            embeddings: [B, embedding_dim] normalized
            labels    : [B]

        Returns:
            loss: scalar
        """
        W = F.normalize(self.weight, p=2, dim=1)
        logits = torch.matmul(embeddings, W.t())

        # Subtract margin from target class
        target_logit = logits.gather(1, labels.view(-1, 1))
        target_logit = target_logit - self.margin

        # Replace target logit
        output = logits.scatter(1, labels.view(-1, 1), target_logit)

        # Scale
        output = self.scale * output

        # Cross-entropy loss
        loss = F.cross_entropy(output, labels)

        return loss


# ── SphereFace Loss ───────────────────────────────────────────────────────────

class SphereFaceLoss(nn.Module):
    """
    SphereFace: Deep Hypersphere Embedding for Face Recognition.

    Paper: SphereFace: Deep Hypersphere Embedding for Face Recognition
    Link: https://arxiv.org/abs/1704.08063

    Key idea: Angular softmax - use angles instead of feature distances.

    Loss = -log(
        exp(m * θ_yi)
        ──────────────────────────────────
        exp(m * θ_yi) + Σ exp(θ_j)
    )

    where:
        θ    = angle between embedding and weight
        m    = angular margin multiplier (typically 4)
        θ_yi = angle for target class
        θ_j  = angles for other classes

    Effect:
        • Enforces angular margin through increasing constraint on target
        • Simpler than ArcFace but effective

    Hyperparameters:
        • margin (m): 1, 2, 3, 4 (integer, typically 4)
        • scale (s): 64
    """

    def __init__(
        self,
        num_classes: int,
        embedding_dim: int = 512,
        margin: int = 4,
        scale: float = 64.0,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.embedding_dim = embedding_dim
        self.margin = margin
        self.scale = scale

        self.weight = nn.Parameter(
            torch.FloatTensor(num_classes, embedding_dim)
        )
        nn.init.xavier_uniform_(self.weight)

    def forward(
        self,
        embeddings: torch.Tensor,
        labels: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            embeddings: [B, embedding_dim] normalized
            labels    : [B]

        Returns:
            loss: scalar
        """
        W = F.normalize(self.weight, p=2, dim=1)
        logits = torch.matmul(embeddings, W.t())

        # Compute angles
        theta = torch.acos(torch.clamp(logits, -1.0, 1.0))

        # Apply margin multiplier to target
        # φ = m * θ for target, θ for others
        phi = theta.clone()
        phi.scatter_(
            1, labels.view(-1, 1),
            self.margin * theta.gather(1, labels.view(-1, 1))
        )

        # Convert back to cosine
        cos_phi = torch.cos(phi)

        # Scale and apply cross-entropy
        output = self.scale * cos_phi
        loss = F.cross_entropy(output, labels)

        return loss


# ── Triplet Loss ──────────────────────────────────────────────────────────────

class TripletLoss(nn.Module):
    """
    Triplet Loss for metric learning in face recognition.

    Paper: FaceNet: A Unified Embedding for Face Recognition and Clustering
    Link: https://arxiv.org/abs/1503.03832

    Key idea: Learn embeddings where same identity is close, different are far.

    Loss = max(d(a, p) - d(a, n) + margin, 0)

    where:
        a = anchor (reference face)
        p = positive (same identity as anchor)
        n = negative (different identity)
        d = Euclidean distance
        margin = minimum required separation (default 0.2)

    Triplet selection strategies:
        1. Random: randomly sample triplets (slow convergence)
        2. Hard: select hardest negatives (fastest convergence)
        3. Semi-hard: select semi-hard negatives (balanced)

    The "hardness" of a triplet:
        - Hard negative: d(a, n) < d(a, p)  (misclassified)
        - Semi-hard: d(a, p) < d(a, n) < d(a, p) + margin

    Hard triplets are critical for convergence!
    """

    def __init__(self, margin: float = 0.2):
        super().__init__()
        self.margin = margin

    def forward(
        self,
        embeddings: torch.Tensor,  # [B, embedding_dim]
        labels: torch.Tensor,      # [B]
    ) -> torch.Tensor:
        """
        Args:
            embeddings: [B, embedding_dim]
            labels    : [B]

        Returns:
            loss: scalar
        """
        # Compute pairwise distances: [B, B]
        dist = torch.cdist(embeddings, embeddings, p=2)

        # Create mask for same identity
        mask = (labels.unsqueeze(0) == labels.unsqueeze(1)).float()

        # For each anchor, find hardest positive and negative
        B = embeddings.shape[0]
        loss = 0.0

        for i in range(B):
            # Positives: same identity as anchor i
            pos_mask = (mask[i] == 1) & (torch.arange(B, device=labels.device) != i)
            if pos_mask.sum() == 0:
                continue

            # Negatives: different identity
            neg_mask = (mask[i] == 0)
            if neg_mask.sum() == 0:
                continue

            # Hardest positive: maximum distance among positives
            pos_dist = dist[i].masked_fill(~pos_mask.bool(), -float('inf'))
            hardest_pos = pos_dist.max()

            # Hardest negative: minimum distance among negatives
            neg_dist = dist[i].masked_fill(~neg_mask.bool(), float('inf'))
            hardest_neg = neg_dist.min()

            # Triplet loss
            triplet_loss = F.relu(hardest_pos - hardest_neg + self.margin)
            loss += triplet_loss

        return loss / max(B, 1)


# ── Batch Hard Triplet Loss ───────────────────────────────────────────────────

class BatchHardTripletLoss(nn.Module):
    """
    Batch Hard Triplet Loss: optimized triplet sampling within batch.

    Key optimization: Instead of computing all triplets O(n³),
    compute hard triplets within each batch O(n²).

    Better convergence than random triplet selection!
    """

    def __init__(self, margin: float = 0.2):
        super().__init__()
        self.margin = margin

    def forward(
        self,
        embeddings: torch.Tensor,
        labels: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            embeddings: [B, embedding_dim]
            labels    : [B]

        Returns:
            loss: scalar
        """
        B = embeddings.shape[0]

        # Pairwise distances
        dist = torch.cdist(embeddings, embeddings, p=2)
        dist = dist.clamp(min=1e-12).sqrt()

        # Same identity mask
        same = (labels.unsqueeze(0) == labels.unsqueeze(1)).float()

        # For each sample, find hardest positive and hardest negative
        # Hardest positive: max distance among positives
        pos_dist = dist.clone()
        pos_dist.masked_fill_(same == 0, -1)
        hardest_pos_dist = pos_dist.max(dim=1)[0]

        # Hardest negative: min distance among negatives
        neg_dist = dist.clone()
        neg_dist.masked_fill_(same == 1, float('inf'))
        hardest_neg_dist = neg_dist.min(dim=1)[0]

        # Triplet loss
        loss = F.relu(hardest_pos_dist - hardest_neg_dist + self.margin).mean()

        return loss


# ── Combined Loss ─────────────────────────────────────────────────────────────

class CombinedLoss(nn.Module):
    """
    Combined loss: ArcFace + Triplet for better convergence.

    Combines benefits of:
        • ArcFace: angular margin, discriminative embeddings
        • Triplet: distance-based learning

    Loss = λ₁ * L_arcface + λ₂ * L_triplet

    Typical weights:
        • λ₁ = 1.0 (ArcFace)
        • λ₂ = 0.5 (Triplet)
    """

    def __init__(
        self,
        num_classes: int,
        embedding_dim: int = 512,
        arcface_weight: float = 1.0,
        triplet_weight: float = 0.5,
        arcface_margin: float = 0.5,
        arcface_scale: float = 64.0,
        triplet_margin: float = 0.2,
    ):
        super().__init__()
        self.arcface_weight = arcface_weight
        self.triplet_weight = triplet_weight

        self.arcface_loss = ArcFaceLoss(
            num_classes=num_classes,
            embedding_dim=embedding_dim,
            margin=arcface_margin,
            scale=arcface_scale,
        )

        self.triplet_loss = BatchHardTripletLoss(margin=triplet_margin)

    def forward(
        self,
        embeddings: torch.Tensor,
        labels: torch.Tensor,
    ) -> Tuple[torch.Tensor, dict]:
        """
        Args:
            embeddings: [B, embedding_dim]
            labels    : [B]

        Returns:
            total_loss, loss_dict
        """
        arcface = self.arcface_loss(embeddings, labels)
        triplet = self.triplet_loss(embeddings, labels)

        total_loss = self.arcface_weight * arcface + self.triplet_weight * triplet

        return total_loss, {
            "arcface": arcface.item(),
            "triplet": triplet.item(),
            "total"  : total_loss.item(),
        }
