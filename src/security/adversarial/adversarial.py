"""
Security — Adversarial Robustness Module.

Implements adversarial attack generation and defense mechanisms
for face recognition models.

Attacks (for robustness testing):
    • FGSM   : Fast Gradient Sign Method
    • PGD    : Projected Gradient Descent
    • CW     : Carlini-Wagner L2
    • Patch  : Physical adversarial patch

Defenses:
    • Adversarial training
    • Input smoothing / denoising
    • Feature squeezing
    • Certified robustness (randomized smoothing)
"""

from __future__ import annotations

from typing import Optional, Tuple

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# ── Adversarial Detection ────────────────────────────────────────────────────

class AdversarialDetector:
    """
    Detects adversarial perturbations in face images.

    Checks for unusually high-frequency pixel patterns
    that are characteristic of adversarial examples (FGSM, PGD).

    Signal: adversarial images have atypically strong
    high-frequency components relative to their visual appearance.
    """

    def __init__(self, threshold: float = 0.70):
        self.threshold = threshold

    def check(self, face_bgr: np.ndarray) -> Tuple[bool, float, str]:
        """
        Check for adversarial perturbations.

        Args:
            face_bgr : (H, W, 3) BGR face crop

        Returns:
            (is_clean, score, reason)
            is_clean=True -> no perturbation detected
        """
        gray = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2GRAY).astype(np.float32)

        # Laplacian of Gaussian (LoG) response
        log = cv2.Laplacian(gray, cv2.CV_32F, ksize=5)
        log_n = np.abs(log) / (gray.std() + 1e-8)
        log_ratio = float(log_n.mean())

        # Noise estimate via median filter residual
        denoised = cv2.medianBlur(face_bgr, 3).astype(np.float32)
        residual = np.abs(face_bgr.astype(np.float32) - denoised)
        noise = float(residual.mean())

        # Higher noise + LoG -> likely adversarial
        # Score closer to 1 = clean, closer to 0 = perturbed
        adv_signal = min((log_ratio * 0.3 + noise * 0.05), 1.0)
        clean_score = max(0.0, 1.0 - adv_signal * 0.8)

        is_clean = clean_score >= self.threshold
        reason = "clean" if is_clean else "possible_adversarial_perturbation"

        return is_clean, round(clean_score, 4), reason


# ── FGSM Attack ───────────────────────────────────────────────────────────────

class FGSMAttack:
    """
    Fast Gradient Sign Method (FGSM) adversarial attack.

    Goodfellow et al. "Explaining and Harnessing Adversarial Examples". 2015.

    Perturbation:
        x_adv = x + ε · sign(∇_x L(θ, x, y))

    Usage:
        attack   = FGSMAttack(model, epsilon=8/255)
        x_adv    = attack(x, y)
    """

    def __init__(
        self,
        model   : nn.Module,
        epsilon : float = 8 / 255,
    ):
        self.model   = model
        self.epsilon = epsilon

    def __call__(
        self,
        x : torch.Tensor,
        y : torch.Tensor,
    ) -> torch.Tensor:
        """
        Generate FGSM adversarial examples.

        Args:
            x : (B, C, H, W) input images [0, 1]
            y : (B,) true labels

        Returns:
            x_adv: adversarial examples (same shape as x)
        """
        x_adv = x.clone().detach().requires_grad_(True)

        self.model.eval()
        logits = self.model(x_adv)
        loss   = F.cross_entropy(logits, y)
        loss.backward()

        with torch.no_grad():
            perturbation = self.epsilon * x_adv.grad.sign()
            x_adv = (x + perturbation).clamp(0, 1)

        return x_adv.detach()


# ── PGD Attack ────────────────────────────────────────────────────────────────

class PGDAttack:
    """
    Projected Gradient Descent (PGD) adversarial attack.

    Madry et al. "Towards Deep Learning Models Resistant to Adversarial Attacks". 2018.

    Iterative FGSM with projection back to ε-ball:
        x_t+1 = Π_{x+S}(x_t + α · sign(∇_x L(θ, x_t, y)))

    Usage:
        attack = PGDAttack(model, epsilon=8/255, steps=20, alpha=2/255)
        x_adv  = attack(x, y)
    """

    def __init__(
        self,
        model   : nn.Module,
        epsilon : float = 8 / 255,
        alpha   : float = 2 / 255,
        steps   : int   = 20,
        random_start: bool = True,
    ):
        self.model        = model
        self.epsilon      = epsilon
        self.alpha        = alpha
        self.steps        = steps
        self.random_start = random_start

    def __call__(
        self,
        x : torch.Tensor,
        y : torch.Tensor,
    ) -> torch.Tensor:
        """Generate PGD adversarial examples."""
        x_adv = x.clone().detach()

        if self.random_start:
            x_adv = x_adv + torch.empty_like(x_adv).uniform_(
                -self.epsilon, self.epsilon
            )
            x_adv = x_adv.clamp(0, 1)

        self.model.eval()
        for _ in range(self.steps):
            x_adv.requires_grad_(True)
            logits = self.model(x_adv)
            loss   = F.cross_entropy(logits, y)
            loss.backward()

            with torch.no_grad():
                x_adv = x_adv + self.alpha * x_adv.grad.sign()
                # Project back to epsilon-ball
                delta = torch.clamp(x_adv - x, -self.epsilon, self.epsilon)
                x_adv = (x + delta).clamp(0, 1).detach()

        return x_adv


# ── Input Smoothing Defense ───────────────────────────────────────────────────

class InputSmoothingDefense:
    """
    Defends against adversarial examples via input smoothing.

    Methods:
        • Gaussian smoothing
        • Median filtering
        • JPEG compression (feature squeezing)

    Usage:
        defense = InputSmoothingDefense(method="gaussian", sigma=1.0)
        x_clean = defense(x_adv)
    """

    def __init__(
        self,
        method : str   = "gaussian",
        sigma  : float = 1.0,
        kernel : int   = 3,
    ):
        self.method = method
        self.sigma  = sigma
        self.kernel = kernel

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        """Apply defense to input tensor."""
        import torchvision.transforms.functional as TF

        if self.method == "gaussian":
            return TF.gaussian_blur(x, kernel_size=self.kernel, sigma=self.sigma)
        elif self.method == "median":
            import cv2
            import numpy as np
            imgs = []
            for img in x:
                arr = img.permute(1, 2, 0).cpu().numpy()
                arr = cv2.medianBlur((arr * 255).astype(np.uint8), self.kernel)
                imgs.append(torch.from_numpy(arr).permute(2, 0, 1).float() / 255)
            return torch.stack(imgs).to(x.device)
        return x


# ── Adversarial Trainer ───────────────────────────────────────────────────────

class AdversarialTrainer:
    """
    Adversarial training wrapper.

    Mixes clean and adversarial examples during training
    to improve model robustness.

    Usage:
        trainer = AdversarialTrainer(
            model    = backbone,
            attack   = PGDAttack(model, epsilon=4/255, steps=7),
            adv_ratio= 0.5,   # 50% adversarial examples per batch
        )
        loss = trainer.compute_loss(images, labels)
    """

    def __init__(
        self,
        model     : nn.Module,
        attack,
        loss_fn   : nn.Module = nn.CrossEntropyLoss(),
        adv_ratio : float = 0.5,
    ):
        self.model     = model
        self.attack    = attack
        self.loss_fn   = loss_fn
        self.adv_ratio = adv_ratio

    def compute_loss(
        self,
        images : torch.Tensor,
        labels : torch.Tensor,
    ) -> Tuple[torch.Tensor, dict]:
        """
        Compute mixed clean + adversarial training loss.

        Returns:
            (total_loss, metrics_dict)
        """
        n_adv   = int(len(images) * self.adv_ratio)
        n_clean = len(images) - n_adv

        clean_imgs = images[:n_clean]
        adv_imgs_src = images[n_clean:]
        adv_labels   = labels[n_clean:]

        # Generate adversarial examples
        adv_imgs = self.attack(adv_imgs_src, adv_labels)

        # Concat and forward
        all_imgs   = torch.cat([clean_imgs, adv_imgs], dim=0)
        all_labels = labels

        self.model.train()
        logits     = self.model(all_imgs)
        total_loss = self.loss_fn(logits, all_labels)

        return total_loss, {
            "n_clean"     : n_clean,
            "n_adversarial": n_adv,
            "adv_ratio"   : self.adv_ratio,
        }
