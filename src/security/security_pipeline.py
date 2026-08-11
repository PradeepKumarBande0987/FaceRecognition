from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional


@dataclass
class SecurityResult:
    """Aggregated security result for one face crop."""

    liveness_ok: bool = True
    liveness_score: float = 1.0
    liveness_reason: str = "live"

    antispoof_ok: bool = True
    antispoof_score: float = 1.0
    antispoof_reason: str = "genuine"

    adversarial_ok: bool = True
    adversarial_score: float = 1.0
    adversarial_reason: str = "clean"

    @property
    def all_passed(self) -> bool:
        return self.liveness_ok and self.antispoof_ok and self.adversarial_ok

    @property
    def overall_score(self) -> float:
        return round(
            (self.liveness_score + self.antispoof_score + self.adversarial_score) / 3.0,
            4,
        )

    def summary(self) -> str:
        parts = []
        if not self.liveness_ok:
            parts.append(f"SPOOF({self.liveness_reason})")
        if not self.antispoof_ok:
            parts.append(f"FAKE({self.antispoof_reason})")
        if not self.adversarial_ok:
            parts.append(f"ADV({self.adversarial_reason})")
        return " | ".join(parts) if parts else "SECURE"


class SecurityPipeline:
    """Composes liveness, anti-spoofing, and adversarial checks."""

    def __init__(
        self,
        liveness_threshold: float = 0.30,
        antispoof_threshold: float = 0.30,
        adversarial_threshold: float = 0.50,
        enabled: bool = True,
        liveness_checker_cls: Optional[Any] = None,
        antispoof_checker_cls: Optional[Any] = None,
        adversarial_checker_cls: Optional[Any] = None,
    ):
        self.enabled = enabled

        if liveness_checker_cls is None or antispoof_checker_cls is None or adversarial_checker_cls is None:
            raise ValueError(
                "SecurityPipeline requires checker classes: "
                "liveness_checker_cls, antispoof_checker_cls, adversarial_checker_cls"
            )

        self.liveness = liveness_checker_cls(liveness_threshold)
        self.antispoof = antispoof_checker_cls(antispoof_threshold)
        self.adversarial = adversarial_checker_cls(adversarial_threshold)

    def check(self, face_bgr: Any) -> SecurityResult:
        if not self.enabled:
            return SecurityResult()

        liv_ok, liv_s, liv_r = self.liveness.check(face_bgr)
        asp_ok, asp_s, asp_r = self.antispoof.check(face_bgr)
        adv_ok, adv_s, adv_r = self.adversarial.check(face_bgr)

        return SecurityResult(
            liveness_ok=liv_ok,
            liveness_score=liv_s,
            liveness_reason=liv_r,
            antispoof_ok=asp_ok,
            antispoof_score=asp_s,
            antispoof_reason=asp_r,
            adversarial_ok=adv_ok,
            adversarial_score=adv_s,
            adversarial_reason=adv_r,
        )
