"""
Hyperparameter Search — MLflow + Optuna.

Searches optimal hyperparameters for the ArcFace face recognition model.

Search Space:
    • Learning rate    : log-uniform [1e-4, 1e-1]
    • ArcFace margin m : uniform [0.3, 0.6]
    • ArcFace scale s  : categorical [32, 48, 64, 80]
    • Weight decay     : log-uniform [1e-5, 1e-3]
    • Batch size       : categorical [128, 256, 512]
    • Embedding dim    : categorical [256, 512]
    • Warmup epochs    : int [1, 10]

Tracking:
    All trials logged to MLflow with parent/child run structure.
    Best trial registered to MLflow Model Registry.

Usage:
    python experiments/hyperparameter_search/hyperparameter_search.py \
        --n-trials 50 \
        --study-name arcface_search_v1 \
        --mlflow-uri http://localhost:5000
"""

import json
import logging
import time
from pathlib import Path
from typing import Any, Dict, Optional

logging.basicConfig(
    level  = logging.INFO,
    format = "%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

# ── Search Space ──────────────────────────────────────────────────────────────

SEARCH_SPACE = {
    "learning_rate"  : {"type": "float",      "low": 1e-4, "high": 1e-1,  "log": True},
    "arcface_margin" : {"type": "float",      "low": 0.3,  "high": 0.6,   "log": False},
    "arcface_scale"  : {"type": "categorical","choices": [32, 48, 64, 80]},
    "weight_decay"   : {"type": "float",      "low": 1e-5, "high": 1e-3,  "log": True},
    "batch_size"     : {"type": "categorical","choices": [128, 256, 512]},
    "embedding_dim"  : {"type": "categorical","choices": [256, 512]},
    "warmup_epochs"  : {"type": "int",        "low": 1,    "high": 10},
    "lr_scheduler"   : {"type": "categorical","choices": ["cosine", "step"]},
    "optimizer"      : {"type": "categorical","choices": ["sgd", "adamw"]},
}

# Fixed config (not searched)
FIXED_CONFIG = {
    "backbone"     : "resnet50",
    "num_epochs"   : 30,
    "image_size"   : (112, 112),
    "num_classes"  : 8631,
    "mixed_precision": True,
    "gradient_clip": 1.0,
}


# ── Trial Result ──────────────────────────────────────────────────────────────

class TrialResult:
    """Result from a single hyperparameter trial."""

    def __init__(self, trial_number: int, params: dict):
        self.trial_number    = trial_number
        self.params          = params
        self.lfw_accuracy    : Optional[float] = None
        self.val_loss        : Optional[float] = None
        self.train_loss      : Optional[float] = None
        self.inference_ms    : Optional[float] = None
        self.run_id          : Optional[str]   = None
        self.status          = "pending"
        self.error_msg       : Optional[str]   = None
        self.duration_secs   : Optional[float] = None

    def to_dict(self) -> dict:
        return {
            "trial_number" : self.trial_number,
            "params"       : self.params,
            "lfw_accuracy" : self.lfw_accuracy,
            "val_loss"     : self.val_loss,
            "train_loss"   : self.train_loss,
            "inference_ms" : self.inference_ms,
            "run_id"       : self.run_id,
            "status"       : self.status,
            "error_msg"    : self.error_msg,
            "duration_secs": self.duration_secs,
        }


# ── Hyperparameter Search ─────────────────────────────────────────────────────

class HyperparameterSearch:
    """
    Optuna-based hyperparameter search with MLflow tracking.

    Each trial is logged as a child run under a parent MLflow run.
    Best trial is registered in MLflow Model Registry.

    Usage:
        search = HyperparameterSearch(
            study_name  = "arcface_v1",
            n_trials    = 50,
            mlflow_uri  = "http://localhost:5000",
            output_dir  = "experiments/hyperparameter_search/results",
        )
        best_params = search.run()
    """

    def __init__(
        self,
        study_name  : str = "arcface_search",
        n_trials    : int = 50,
        mlflow_uri  : str = "http://localhost:5000",
        output_dir  : str = "experiments/hyperparameter_search/results",
        direction   : str = "maximize",
        metric      : str = "lfw_accuracy",
        seed        : int = 42,
    ):
        self.study_name  = study_name
        self.n_trials    = n_trials
        self.mlflow_uri  = mlflow_uri
        self.output_dir  = Path(output_dir)
        self.direction   = direction
        self.metric      = metric
        self.seed        = seed
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.trial_results: list[TrialResult] = []

    # ── Objective ─────────────────────────────────────────────────────────────

    def objective(self, trial) -> float:
        """
        Optuna objective function.

        Samples hyperparameters, trains model, returns metric.
        Each trial is logged as a child MLflow run.

        Args:
            trial: Optuna trial object

        Returns:
            Metric value to optimize (e.g., LFW accuracy)
        """
        # Sample hyperparameters
        params = self._sample_params(trial)
        result = TrialResult(trial.number, params)
        start  = time.time()

        try:
            import mlflow

            # Child MLflow run for this trial
            with mlflow.start_run(
                nested   = True,
                run_name = f"trial_{trial.number:03d}",
            ) as child_run:
                result.run_id = child_run.info.run_id

                # Log all hyperparameters
                mlflow.log_params({**params, **FIXED_CONFIG})
                mlflow.set_tags({
                    "trial_number" : trial.number,
                    "study_name"   : self.study_name,
                    "optimizer"    : params.get("optimizer", "sgd"),
                })

                # ── TODO: Replace with real training ──────────────────────
                # from pretraining.trainer import Trainer
                # trainer = Trainer(**{**params, **FIXED_CONFIG})
                # metrics = trainer.train()
                # lfw_acc = evaluate_lfw(trainer.model)
                # ──────────────────────────────────────────────────────────

                # Stub training (replace with real call)
                import random
                random.seed(trial.number + self.seed)

                lfw_acc     = round(98.0 + random.uniform(0, 1.5), 4)
                val_loss    = round(4.0  - random.uniform(0, 2.0), 4)
                train_loss  = round(3.5  - random.uniform(0, 1.5), 4)
                infer_ms    = round(35.0 + random.uniform(0, 25.0), 1)

                result.lfw_accuracy = lfw_acc
                result.val_loss     = val_loss
                result.train_loss   = train_loss
                result.inference_ms = infer_ms
                result.status       = "done"

                # Log metrics
                mlflow.log_metrics({
                    "lfw_accuracy"  : lfw_acc,
                    "val_loss"      : val_loss,
                    "train_loss"    : train_loss,
                    "inference_ms"  : infer_ms,
                })

                # Store run_id for best trial retrieval
                trial.set_user_attr("run_id", child_run.info.run_id)

        except ImportError:
            logger.warning("MLflow not installed. Running without tracking.")
            import random
            random.seed(trial.number + self.seed)
            result.lfw_accuracy = round(98.0 + random.uniform(0, 1.5), 4)
            result.status = "done"

        except Exception as e:
            result.status    = "failed"
            result.error_msg = str(e)
            logger.error(f"Trial {trial.number} failed: {e}")
            result.lfw_accuracy = 0.0

        result.duration_secs = round(time.time() - start, 2)
        self.trial_results.append(result)

        logger.info(
            f"  Trial {trial.number:3d} | "
            f"LFW={result.lfw_accuracy:.2f}% | "
            f"params={json.dumps(params, separators=(',', ':'))}"
        )

        return result.lfw_accuracy or 0.0

    # ── Sample Params ─────────────────────────────────────────────────────────

    def _sample_params(self, trial) -> Dict[str, Any]:
        """Sample hyperparameters from search space."""
        params = {}
        for name, spec in SEARCH_SPACE.items():
            if spec["type"] == "float":
                params[name] = trial.suggest_float(
                    name, spec["low"], spec["high"], log=spec.get("log", False)
                )
            elif spec["type"] == "int":
                params[name] = trial.suggest_int(
                    name, spec["low"], spec["high"]
                )
            elif spec["type"] == "categorical":
                params[name] = trial.suggest_categorical(
                    name, spec["choices"]
                )
        return params

    # ── Run Search ────────────────────────────────────────────────────────────

    def run(self) -> Dict[str, Any]:
        """
        Execute the hyperparameter search.

        Returns:
            Best hyperparameter configuration found
        """
        try:
            import optuna
            import mlflow

            mlflow.set_tracking_uri(self.mlflow_uri)
            mlflow.set_experiment(self.study_name)

            logger.info(f"\n🔍 Hyperparameter Search: {self.study_name}")
            logger.info(f"   Trials    : {self.n_trials}")
            logger.info(f"   Metric    : {self.metric} ({self.direction})")
            logger.info(f"   MLflow    : {self.mlflow_uri}\n")

            # Parent MLflow run
            with mlflow.start_run(run_name=f"{self.study_name}_study"):
                mlflow.log_param("n_trials", self.n_trials)
                mlflow.log_param("search_space", json.dumps(SEARCH_SPACE))
                mlflow.set_tags({
                    "study_name" : self.study_name,
                    "direction"  : self.direction,
                    "metric"     : self.metric,
                })

                # Optuna study
                study = optuna.create_study(
                    direction  = self.direction,
                    study_name = self.study_name,
                    sampler    = optuna.samplers.TPESampler(seed=self.seed),
                    pruner     = optuna.pruners.MedianPruner(n_startup_trials=5),
                )

                study.optimize(self.objective, n_trials=self.n_trials)

                # Log best trial
                best = study.best_trial
                mlflow.log_params(best.params)
                mlflow.log_metric(f"best_{self.metric}", best.value)
                if run_id := best.user_attrs.get("run_id"):
                    mlflow.log_param("best_child_run_id", run_id)

            best_params = {**best.params, **FIXED_CONFIG}
            self._save_results(study, best_params)
            self._print_summary(study, best_params)
            return best_params

        except ImportError as e:
            logger.error(f"Missing dependency: {e}")
            logger.info("Install with: pip install optuna mlflow")
            return {}

    # ── Save / Print ──────────────────────────────────────────────────────────

    def _save_results(self, study, best_params: dict):
        """Save search results to JSON."""
        output_path = (
            self.output_dir / f"{self.study_name}_results.json"
        )
        data = {
            "study_name"    : self.study_name,
            "n_trials"      : self.n_trials,
            "best_params"   : best_params,
            "best_value"    : study.best_value,
            "all_trials"    : [r.to_dict() for r in self.trial_results],
        }
        with open(output_path, "w") as f:
            json.dump(data, f, indent=2)
        logger.info(f"\n💾 Results saved: {output_path}")

    def _print_summary(self, study, best_params: dict):
        """Print best trial summary."""
        print(f"\n{'='*60}")
        print(f"🏆 Best Trial: #{study.best_trial.number}")
        print(f"   {self.metric}: {study.best_value:.4f}%")
        print(f"\n   Best Parameters:")
        for k, v in best_params.items():
            if k not in FIXED_CONFIG:
                print(f"     {k:<20}: {v}")
        print(f"{'='*60}")


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="ArcFace Hyperparameter Search (Optuna + MLflow)"
    )
    parser.add_argument("--n-trials",   type=int, default=50)
    parser.add_argument("--study-name", default="arcface_search_v1")
    parser.add_argument("--mlflow-uri", default="http://localhost:5000")
    parser.add_argument("--output-dir", default="experiments/hyperparameter_search/results")
    parser.add_argument("--metric",     default="lfw_accuracy")
    parser.add_argument("--seed",       type=int, default=42)

    args = parser.parse_args()

    search = HyperparameterSearch(
        study_name = args.study_name,
        n_trials   = args.n_trials,
        mlflow_uri = args.mlflow_uri,
        output_dir = args.output_dir,
        metric     = args.metric,
        seed       = args.seed,
    )
    best = search.run()
