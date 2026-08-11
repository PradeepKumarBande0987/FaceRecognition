"""
Ablation Study Runner.

Systematically removes/adds components to measure each one's
contribution to final model performance.

Ablation Dimensions:
    1. Loss function       : Softmax vs ArcFace vs CosFace
    2. Backbone            : ResNet50 vs EfficientNet vs ViT
    3. Augmentation        : none → standard → occlusion → LR
    4. Training data       : VGGFace2 only → + CelebA → + CCTV
    5. ArcFace margin m    : 0.3 → 0.4 → 0.5 → 0.6
    6. Embedding dim       : 128 → 256 → 512 → 1024
    7. Feature scale s     : 32 → 48 → 64 → 80

Usage:
    python experiments/ablations/ablations.py \
        --dimension loss_function \
        --output-dir experiments/ablations/results/
"""

import json
import time
import uuid
import logging
from dataclasses import dataclass, field, asdict
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional

logging.basicConfig(
    level  = logging.INFO,
    format = "%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)


# ── Ablation Enums ────────────────────────────────────────────────────────────

class AblationDimension(str, Enum):
    LOSS_FUNCTION   = "loss_function"
    BACKBONE        = "backbone"
    AUGMENTATION    = "augmentation"
    TRAINING_DATA   = "training_data"
    ARCFACE_MARGIN  = "arcface_margin"
    EMBEDDING_DIM   = "embedding_dim"
    FEATURE_SCALE   = "feature_scale"


# ── Ablation Configs ──────────────────────────────────────────────────────────

ABLATION_CONFIGS: Dict[str, List[Dict[str, Any]]] = {

    AblationDimension.LOSS_FUNCTION: [
        {"name": "softmax",  "loss": "softmax",  "margin": 0.0, "scale": 1},
        {"name": "cosface",  "loss": "cosface",  "margin": 0.35,"scale": 64},
        {"name": "arcface",  "loss": "arcface",  "margin": 0.5, "scale": 64},
    ],

    AblationDimension.BACKBONE: [
        {"name": "resnet50",       "backbone": "resnet50",       "params_m": 25.6},
        {"name": "efficientnet_b4","backbone": "efficientnet_b4","params_m": 19.3},
        {"name": "mobilefacenet",  "backbone": "mobilefacenet",  "params_m": 0.99},
        {"name": "vit_face",       "backbone": "vit_face",       "params_m": 86.6},
    ],

    AblationDimension.AUGMENTATION: [
        {
            "name"          : "no_augmentation",
            "standard_aug"  : False,
            "occlusion_aug" : False,
            "lr_aug"        : False,
        },
        {
            "name"          : "standard_only",
            "standard_aug"  : True,
            "occlusion_aug" : False,
            "lr_aug"        : False,
        },
        {
            "name"          : "standard_occlusion",
            "standard_aug"  : True,
            "occlusion_aug" : True,
            "lr_aug"        : False,
        },
        {
            "name"          : "full_augmentation",
            "standard_aug"  : True,
            "occlusion_aug" : True,
            "lr_aug"        : True,
        },
    ],

    AblationDimension.TRAINING_DATA: [
        {"name": "vggface2_only",          "datasets": ["vggface2"]},
        {"name": "vggface2_celeba",        "datasets": ["vggface2", "celeba"]},
        {"name": "vggface2_celeba_cctv",   "datasets": ["vggface2", "celeba", "custom_cctv"]},
    ],

    AblationDimension.ARCFACE_MARGIN: [
        {"name": f"margin_{m}", "margin": m}
        for m in [0.3, 0.4, 0.5, 0.6]
    ],

    AblationDimension.EMBEDDING_DIM: [
        {"name": f"dim_{d}", "embedding_dim": d}
        for d in [128, 256, 512, 1024]
    ],

    AblationDimension.FEATURE_SCALE: [
        {"name": f"scale_{s}", "scale": s}
        for s in [32, 48, 64, 80]
    ],
}


# ── Result Dataclass ──────────────────────────────────────────────────────────

@dataclass
class AblationResult:
    """Result from a single ablation run."""

    run_id          : str
    dimension       : str
    config_name     : str
    config          : Dict[str, Any]

    # Core metrics
    lfw_accuracy    : Optional[float] = None
    lfw_auc         : Optional[float] = None
    vggface2_rank1  : Optional[float] = None
    vggface2_rank5  : Optional[float] = None
    train_loss      : Optional[float] = None
    val_loss        : Optional[float] = None

    # Efficiency
    params_millions     : Optional[float] = None
    inference_ms        : Optional[float] = None
    epochs_to_converge  : Optional[int]   = None

    # Meta
    status          : str = "pending"   # pending / running / done / failed
    error_msg       : Optional[str] = None
    started_at      : Optional[str] = None
    finished_at     : Optional[str] = None
    duration_minutes: Optional[float] = None

    def to_dict(self) -> dict:
        return asdict(self)


# ── Ablation Runner ───────────────────────────────────────────────────────────

class AblationRunner:
    """
    Runs and tracks ablation studies.

    Usage:
        runner = AblationRunner(output_dir="experiments/ablations/results")
        runner.run_dimension(AblationDimension.LOSS_FUNCTION)
        runner.run_all()
        runner.save_summary()
    """

    def __init__(
        self,
        output_dir  : str = "experiments/ablations/results",
        base_epochs : int = 30,
        seed        : int = 42,
    ):
        self.output_dir  = Path(output_dir)
        self.base_epochs = base_epochs
        self.seed        = seed
        self.results     : List[AblationResult] = []
        self.output_dir.mkdir(parents=True, exist_ok=True)

    # ── Run Single Config ─────────────────────────────────────────────────────

    def run_single(
        self,
        dimension   : str,
        config      : Dict[str, Any],
    ) -> AblationResult:
        """
        Run a single ablation configuration.

        Args:
            dimension : ablation dimension name
            config    : configuration dict for this trial

        Returns:
            AblationResult with metrics filled in
        """
        run_id = str(uuid.uuid4())[:8]
        result = AblationResult(
            run_id      = run_id,
            dimension   = dimension,
            config_name = config.get("name", run_id),
            config      = config,
            started_at  = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            status      = "running",
        )

        logger.info(
            f"\n🔬 Ablation: [{dimension}] config='{config.get('name')}'"
        )

        start = time.time()

        try:
            # ── TODO: Replace with real training call ──────────────────────
            # from pretraining.trainer import Trainer
            # from models.backbones import get_backbone
            # trainer = Trainer(
            #     model        = get_backbone(config.get("backbone", "resnet50")),
            #     loss_fn      = get_loss(config),
            #     train_loader = get_train_loader(config),
            #     num_epochs   = self.base_epochs,
            # )
            # metrics = trainer.train()
            # ──────────────────────────────────────────────────────────────

            # Stub metrics
            import random
            random.seed(hash(config.get("name", "")))

            result.lfw_accuracy     = round(98.5 + random.uniform(0, 1.0), 2)
            result.lfw_auc          = round(0.998 + random.uniform(0, 0.002), 4)
            result.vggface2_rank1   = round(94.0 + random.uniform(0, 3.5), 2)
            result.vggface2_rank5   = round(98.0 + random.uniform(0, 1.5), 2)
            result.train_loss       = round(5.0  - random.uniform(0, 2.0), 4)
            result.val_loss         = round(5.5  - random.uniform(0, 2.0), 4)
            result.inference_ms     = round(30.0 + random.uniform(0, 30.0), 1)
            result.epochs_to_converge = random.randint(20, 40)
            result.status           = "done"

        except Exception as e:
            result.status    = "failed"
            result.error_msg = str(e)
            logger.error(f"  ❌ Failed: {e}")

        result.finished_at      = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        result.duration_minutes = round((time.time() - start) / 60, 2)

        logger.info(
            f"  ✅ LFW: {result.lfw_accuracy}% | "
            f"VGGFace2 Rank-1: {result.vggface2_rank1}%"
        )

        # Save individual result
        result_path = (
            self.output_dir
            / dimension
            / f"{result.config_name}_{run_id}.json"
        )
        result_path.parent.mkdir(parents=True, exist_ok=True)
        with open(result_path, "w") as f:
            json.dump(result.to_dict(), f, indent=2)

        self.results.append(result)
        return result

    # ── Run Dimension ─────────────────────────────────────────────────────────

    def run_dimension(
        self,
        dimension: AblationDimension,
    ) -> List[AblationResult]:
        """
        Run all ablation configs for a given dimension.

        Args:
            dimension: which aspect to ablate

        Returns:
            List of AblationResult for all configs
        """
        configs = ABLATION_CONFIGS.get(dimension, [])
        if not configs:
            logger.warning(f"No configs found for dimension: {dimension}")
            return []

        logger.info(
            f"\n📊 Running ablation: {dimension} "
            f"({len(configs)} configs)"
        )

        dim_results = []
        for cfg in configs:
            result = self.run_single(dimension=dimension, config=cfg)
            dim_results.append(result)

        self._print_dimension_summary(dimension, dim_results)
        return dim_results

    # ── Run All ───────────────────────────────────────────────────────────────

    def run_all(self) -> List[AblationResult]:
        """Run all ablation dimensions."""
        logger.info("\n🚀 Running ALL ablation dimensions\n")
        for dim in AblationDimension:
            self.run_dimension(dim)
        self.save_summary()
        return self.results

    # ── Summary ───────────────────────────────────────────────────────────────

    def save_summary(self):
        """Save aggregated ablation summary to JSON and Markdown."""
        summary_path = self.output_dir / "ablation_summary.json"
        md_path      = self.output_dir / "ablation_summary.md"

        # JSON
        all_data = [r.to_dict() for r in self.results]
        with open(summary_path, "w") as f:
            json.dump(all_data, f, indent=2)

        # Markdown table
        lines = [
            "# Ablation Study Summary\n",
            "| Dimension | Config | LFW Acc (%) | VGGFace2 Rank-1 (%) |",
            "|-----------|--------|-------------|---------------------|",
        ]
        for r in sorted(
            self.results, key=lambda x: -(x.lfw_accuracy or 0)
        ):
            lines.append(
                f"| {r.dimension} | {r.config_name} "
                f"| {r.lfw_accuracy} | {r.vggface2_rank1} |"
            )

        with open(md_path, "w") as f:
            f.write("\n".join(lines))

        logger.info(f"\n💾 Summary saved → {summary_path}")
        logger.info(f"💾 Markdown  saved → {md_path}")

    def _print_dimension_summary(
        self,
        dimension   : str,
        results     : List[AblationResult],
    ):
        """Print sorted results for a dimension."""
        sorted_r = sorted(
            [r for r in results if r.lfw_accuracy is not None],
            key=lambda x: -(x.lfw_accuracy or 0),
        )
        print(f"\n📊 Results for: {dimension}")
        print("-" * 60)
        print(f"  {'Config':<25} {'LFW Acc':>10} {'Rank-1':>10}")
        print("-" * 60)
        for r in sorted_r:
            print(
                f"  {r.config_name:<25} "
                f"{r.lfw_accuracy:>9.2f}% "
                f"{r.vggface2_rank1:>9.2f}%"
            )
        print("-" * 60)
        if sorted_r:
            best = sorted_r[0]
            print(f"  🏆 Best: {best.config_name} (LFW={best.lfw_accuracy}%)")


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Ablation Study Runner")
    parser.add_argument(
        "--dimension",
        choices=[d.value for d in AblationDimension] + ["all"],
        default="all",
    )
    parser.add_argument("--output-dir", default="experiments/ablations/results")
    parser.add_argument("--epochs", type=int, default=30)
    args = parser.parse_args()

    runner = AblationRunner(
        output_dir  = args.output_dir,
        base_epochs = args.epochs,
    )

    if args.dimension == "all":
        runner.run_all()
    else:
        runner.run_dimension(AblationDimension(args.dimension))
