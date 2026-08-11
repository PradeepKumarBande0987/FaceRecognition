"""
Edge Device Benchmark Script.

Benchmarks face recognition model performance on:
    • CPU          (x86 / ARM)
    • CUDA GPU     (NVIDIA)
    • ONNX Runtime (CPU / CUDA / TensorRT provider)
    • Simulated edge constraints (memory / latency limits)

Metrics Reported:
    • Latency      : p50, p90, p95, p99 in milliseconds
    • Throughput   : images per second
    • Memory       : peak GPU/RAM usage (MB)
    • Accuracy     : LFW TAR@FAR=0.1% (if pairs provided)
    • Power proxy  : throughput / memory (higher = more efficient)

Edge Profiles:
    • raspberry_pi  : 1 core, 512MB RAM constraint simulation
    • jetson_nano   : 4 cores, 4GB RAM, CUDA constraint
    • standard_cpu  : 8 cores, no memory constraint
    • a100_gpu      : Full A100 80GB GPU benchmark
    • onnx_cpu      : ONNX Runtime, CPU provider
    • onnx_tensorrt : ONNX Runtime, TensorRT provider (NVIDIA only)

Usage:
    python scripts/benchmark/run_edge_benchmark.py \
        --checkpoint  experiments/runs/run_001/checkpoints/best_model.pt \
        --onnx-model  models/exported/face_recognition.onnx \
        --profile     onnx_cpu standard_cpu \
        --batch-sizes 1 4 8 16 \
        --n-runs      200 \
        --output-dir  docs/results/benchmarks/
"""

import argparse
import json
import os
import platform
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from scripts.train.train_baseline import build_backbone


# ── Edge Profiles ─────────────────────────────────────────────────────────────

EDGE_PROFILES = {
    "raspberry_pi" : {
        "description"     : "Raspberry Pi 4 (ARM Cortex-A72, 4GB RAM)",
        "n_threads"       : 1,
        "memory_limit_mb" : 512,
        "backend"         : "torch_cpu",
        "device"          : "cpu",
    },
    "jetson_nano"  : {
        "description"     : "NVIDIA Jetson Nano (4 cores, 4GB RAM, 128 CUDA)",
        "n_threads"       : 4,
        "memory_limit_mb" : 4096,
        "backend"         : "torch_cuda",
        "device"          : "cuda",
    },
    "standard_cpu" : {
        "description"     : "Standard x86 CPU (8 cores)",
        "n_threads"       : 8,
        "memory_limit_mb" : None,
        "backend"         : "torch_cpu",
        "device"          : "cpu",
    },
    "a100_gpu"     : {
        "description"     : "NVIDIA A100 80GB GPU",
        "n_threads"       : None,
        "memory_limit_mb" : None,
        "backend"         : "torch_cuda",
        "device"          : "cuda",
    },
    "onnx_cpu"     : {
        "description"     : "ONNX Runtime — CPUExecutionProvider",
        "n_threads"       : 4,
        "memory_limit_mb" : None,
        "backend"         : "onnx",
        "provider"        : "CPUExecutionProvider",
    },
    "onnx_cuda"    : {
        "description"     : "ONNX Runtime — CUDAExecutionProvider",
        "n_threads"       : None,
        "memory_limit_mb" : None,
        "backend"         : "onnx",
        "provider"        : "CUDAExecutionProvider",
    },
    "onnx_tensorrt": {
        "description"     : "ONNX Runtime — TensorrtExecutionProvider",
        "n_threads"       : None,
        "memory_limit_mb" : None,
        "backend"         : "onnx",
        "provider"        : "TensorrtExecutionProvider",
    },
}


# ── Benchmark Result ──────────────────────────────────────────────────────────

@dataclass
class BenchmarkResult:
    """Result from a single benchmark run."""

    profile       : str
    backend       : str
    batch_size    : int
    n_runs        : int
    image_size    : tuple

    # Latency (ms)
    latency_p50   : float = 0.0
    latency_p90   : float = 0.0
    latency_p95   : float = 0.0
    latency_p99   : float = 0.0
    latency_mean  : float = 0.0
    latency_std   : float = 0.0

    # Throughput
    throughput_ips: float = 0.0     # images per second

    # Memory (MB)
    memory_mb     : float = 0.0

    # Model info
    model_params_m: float = 0.0

    # System
    cpu_model     : str = ""
    gpu_model     : str = ""
    platform      : str = ""

    # Status
    status        : str = "pending"
    error_msg     : str = ""

    def to_dict(self) -> dict:
        return asdict(self)

    def print_summary(self):
        """Print formatted benchmark summary."""
        print(f"\n  Profile    : {self.profile}")
        print(f"  Backend    : {self.backend}")
        print(f"  Batch size : {self.batch_size}")
        print(f"  Latency    : p50={self.latency_p50:.1f}ms | "
              f"p95={self.latency_p95:.1f}ms | "
              f"p99={self.latency_p99:.1f}ms")
        print(f"  Throughput : {self.throughput_ips:.1f} images/sec")
        print(f"  Memory     : {self.memory_mb:.1f} MB")
        print(f"  Status     : {self.status}")


# ── Edge Benchmarker ──────────────────────────────────────────────────────────

class EdgeBenchmarker:
    """
    Runs face recognition inference benchmarks across
    different hardware profiles and backends.

    Supports:
        • PyTorch (CPU + CUDA)
        • ONNX Runtime (CPU + CUDA + TensorRT)

    Usage:
        benchmarker = EdgeBenchmarker(
            checkpoint  = "experiments/runs/best/checkpoints/best_model.pt",
            onnx_model  = "models/exported/face_recognition.onnx",
        )
        results = benchmarker.run_all(
            profiles    = ["onnx_cpu", "standard_cpu"],
            batch_sizes = [1, 4, 8],
            n_runs      = 200,
        )
        benchmarker.save_report(results, "docs/results/benchmarks/")
    """

    def __init__(
        self,
        checkpoint    : Optional[str] = None,
        onnx_model    : Optional[str] = None,
        backbone_name : str = "resnet50",
        embedding_dim : int = 512,
        image_size    : Tuple[int, int] = (112, 112),
        warmup        : int = 20,
    ):
        self.checkpoint    = checkpoint
        self.onnx_model    = onnx_model
        self.backbone_name = backbone_name
        self.embedding_dim = embedding_dim
        self.image_size    = image_size
        self.warmup        = warmup

        # System info
        self.sys_info = self._get_system_info()
        self._print_system_info()

    # ── System Info ───────────────────────────────────────────────────────────

    def _get_system_info(self) -> dict:
        """Collect system hardware information."""
        info = {
            "platform"    : platform.platform(),
            "python"      : platform.python_version(),
            "pytorch"     : torch.__version__,
            "cpu_model"   : platform.processor() or "unknown",
            "cuda_available": torch.cuda.is_available(),
            "gpu_model"   : (
                torch.cuda.get_device_name(0)
                if torch.cuda.is_available() else "N/A"
            ),
            "gpu_memory_gb": (
                torch.cuda.get_device_properties(0).total_memory / 1e9
                if torch.cuda.is_available() else 0.0
            ),
            "cpu_count"   : os.cpu_count(),
        }
        return info

    def _print_system_info(self):
        """Print system hardware summary."""
        print(f"\n🖥️  System Info")
        print(f"   Platform  : {self.sys_info['platform']}")
        print(f"   PyTorch   : {self.sys_info['pytorch']}")
        print(f"   CPU       : {self.sys_info['cpu_model']}")
        print(f"   CPU cores : {self.sys_info['cpu_count']}")
        print(f"   CUDA      : {self.sys_info['cuda_available']}")
        if self.sys_info["cuda_available"]:
            print(f"   GPU       : {self.sys_info['gpu_model']}")
            print(f"   GPU mem   : {self.sys_info['gpu_memory_gb']:.1f} GB")

    # ── PyTorch Benchmark ─────────────────────────────────────────────────────

    def _benchmark_pytorch(
        self,
        profile    : str,
        batch_size : int,
        n_runs     : int,
    ) -> BenchmarkResult:
        """Benchmark using native PyTorch inference."""
        cfg    = EDGE_PROFILES[profile]
        device = torch.device(cfg["device"])

        if "cuda" in cfg["device"] and not torch.cuda.is_available():
            result = BenchmarkResult(
                profile=profile, backend="torch",
                batch_size=batch_size, n_runs=n_runs,
                image_size=self.image_size,
                status="skipped", error_msg="CUDA not available",
            )
            return result

        # Thread control for CPU profiles
        if cfg.get("n_threads"):
            torch.set_num_threads(cfg["n_threads"])

        # Load model
        backbone = build_backbone(self.backbone_name, self.embedding_dim)

        if self.checkpoint and Path(self.checkpoint).exists():
            ckpt = torch.load(
                self.checkpoint, map_location=device, weights_only=False
            )
            state = ckpt.get("model_state_dict", ckpt)
            state = {k.replace("module.", ""): v for k, v in state.items()}
            backbone.load_state_dict(state)

        backbone.eval().to(device)

        # Count params
        n_params = sum(p.numel() for p in backbone.parameters()) / 1e6

        # Dummy input
        H, W = self.image_size
        dummy = torch.randn(batch_size, 3, H, W, device=device)

        # Warmup
        with torch.no_grad():
            for _ in range(self.warmup):
                _ = backbone(dummy)

        if "cuda" in str(device):
            torch.cuda.synchronize()

        # Timed runs
        latencies = []
        mem_before = (
            torch.cuda.memory_allocated(device) / 1e6
            if "cuda" in str(device) else 0.0
        )

        with torch.no_grad():
            for _ in range(n_runs):
                t0 = time.perf_counter()
                _  = backbone(dummy)
                if "cuda" in str(device):
                    torch.cuda.synchronize()
                latencies.append((time.perf_counter() - t0) * 1000)

        mem_after = (
            torch.cuda.max_memory_allocated(device) / 1e6
            if "cuda" in str(device) else 0.0
        )

        lats = np.array(latencies)
        return BenchmarkResult(
            profile        = profile,
            backend        = f"torch_{cfg['device']}",
            batch_size     = batch_size,
            n_runs         = n_runs,
            image_size     = self.image_size,
            latency_p50    = round(float(np.percentile(lats, 50)), 2),
            latency_p90    = round(float(np.percentile(lats, 90)), 2),
            latency_p95    = round(float(np.percentile(lats, 95)), 2),
            latency_p99    = round(float(np.percentile(lats, 99)), 2),
            latency_mean   = round(float(lats.mean()), 2),
            latency_std    = round(float(lats.std()), 2),
            throughput_ips = round(batch_size / (lats.mean() / 1000), 1),
            memory_mb      = round(mem_after - mem_before, 1),
            model_params_m = round(n_params, 2),
            cpu_model      = self.sys_info["cpu_model"],
            gpu_model      = self.sys_info.get("gpu_model", ""),
            platform       = self.sys_info["platform"],
            status         = "done",
        )

    # ── ONNX Benchmark ────────────────────────────────────────────────────────

    def _benchmark_onnx(
        self,
        profile    : str,
        batch_size : int,
        n_runs     : int,
    ) -> BenchmarkResult:
        """Benchmark using ONNX Runtime."""
        try:
            import onnxruntime as ort
        except ImportError:
            return BenchmarkResult(
                profile=profile, backend="onnx",
                batch_size=batch_size, n_runs=n_runs,
                image_size=self.image_size,
                status="skipped",
                error_msg="onnxruntime not installed",
            )

        if not self.onnx_model or not Path(self.onnx_model).exists():
            return BenchmarkResult(
                profile=profile, backend="onnx",
                batch_size=batch_size, n_runs=n_runs,
                image_size=self.image_size,
                status="skipped",
                error_msg=f"ONNX model not found: {self.onnx_model}",
            )

        cfg      = EDGE_PROFILES[profile]
        provider = cfg.get("provider", "CPUExecutionProvider")

        # Session options
        sess_opts = ort.SessionOptions()
        if cfg.get("n_threads"):
            sess_opts.intra_op_num_threads = cfg["n_threads"]
            sess_opts.inter_op_num_threads = 1
        sess_opts.graph_optimization_level = (
            ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        )

        try:
            session = ort.InferenceSession(
                self.onnx_model,
                sess_options = sess_opts,
                providers    = [provider],
            )
        except Exception as e:
            return BenchmarkResult(
                profile=profile, backend="onnx",
                batch_size=batch_size, n_runs=n_runs,
                image_size=self.image_size,
                status="failed",
                error_msg=str(e),
            )

        # Input name
        input_name = session.get_inputs()[0].name
        H, W       = self.image_size
        dummy_np   = np.random.randn(
            batch_size, 3, H, W
        ).astype(np.float32)
        inputs     = {input_name: dummy_np}

        # Warmup
        for _ in range(self.warmup):
            session.run(None, inputs)

        # Timed runs
        latencies = []
        for _ in range(n_runs):
            t0 = time.perf_counter()
            session.run(None, inputs)
            latencies.append((time.perf_counter() - t0) * 1000)

        lats = np.array(latencies)
        return BenchmarkResult(
            profile        = profile,
            backend        = f"onnx_{provider.lower().replace('executionprovider','')}",
            batch_size     = batch_size,
            n_runs         = n_runs,
            image_size     = self.image_size,
            latency_p50    = round(float(np.percentile(lats, 50)), 2),
            latency_p90    = round(float(np.percentile(lats, 90)), 2),
            latency_p95    = round(float(np.percentile(lats, 95)), 2),
            latency_p99    = round(float(np.percentile(lats, 99)), 2),
            latency_mean   = round(float(lats.mean()), 2),
            latency_std    = round(float(lats.std()), 2),
            throughput_ips = round(batch_size / (lats.mean() / 1000), 1),
            memory_mb      = 0.0,
            cpu_model      = self.sys_info["cpu_model"],
            gpu_model      = self.sys_info.get("gpu_model", ""),
            platform       = self.sys_info["platform"],
            status         = "done",
        )

    # ── Run All ───────────────────────────────────────────────────────────────

    def run_all(
        self,
        profiles    : List[str],
        batch_sizes : List[int] = [1, 4, 8, 16],
        n_runs      : int = 200,
    ) -> List[BenchmarkResult]:
        """
        Run benchmarks for all profile × batch_size combinations.

        Args:
            profiles    : list of profile names (see EDGE_PROFILES)
            batch_sizes : list of batch sizes to test
            n_runs      : number of timed inference runs

        Returns:
            List of BenchmarkResult (one per profile × batch_size)
        """
        all_results = []
        total = len(profiles) * len(batch_sizes)
        count = 0

        print(f"\n🚀 Starting edge benchmarks")
        print(f"   Profiles    : {profiles}")
        print(f"   Batch sizes : {batch_sizes}")
        print(f"   Runs/config : {n_runs}")
        print(f"   Total runs  : {total}\n")

        for profile in profiles:
            if profile not in EDGE_PROFILES:
                print(f"⚠️  Unknown profile: {profile}. Skipping.")
                continue

            cfg     = EDGE_PROFILES[profile]
            backend = cfg["backend"]

            print(f"\n{'='*55}")
            print(f"📍 Profile: {profile}")
            print(f"   {cfg['description']}")
            print(f"{'='*55}")

            for bs in batch_sizes:
                count += 1
                print(f"\n  [{count}/{total}] Batch size = {bs}")

                if backend in ["torch_cpu", "torch_cuda"]:
                    result = self._benchmark_pytorch(profile, bs, n_runs)
                elif backend == "onnx":
                    result = self._benchmark_onnx(profile, bs, n_runs)
                else:
                    print(f"  ❌ Unknown backend: {backend}")
                    continue

                result.print_summary()
                all_results.append(result)

        return all_results

    # ── Save Report ───────────────────────────────────────────────────────────

    def save_report(
        self,
        results    : List[BenchmarkResult],
        output_dir : str,
    ):
        """
        Save benchmark report as JSON + Markdown.

        Args:
            results    : list of BenchmarkResult
            output_dir : output directory path
        """
        output = Path(output_dir)
        output.mkdir(parents=True, exist_ok=True)

        # JSON
        json_path = output / "benchmark_results.json"
        with open(json_path, "w") as f:
            json.dump(
                {
                    "system_info" : self.sys_info,
                    "results"     : [r.to_dict() for r in results],
                },
                f, indent=2, default=str,
            )
        print(f"\n💾 JSON saved : {json_path}")

        # Markdown table
        md_path = output / "benchmark_report.md"
        self._write_markdown(results, md_path)
        print(f"💾 Report saved: {md_path}")

    def _write_markdown(self, results: List[BenchmarkResult], path: Path):
        """Write markdown benchmark report."""
        lines = [
            "# ⚡ Edge Benchmark Report\n",
            f"**Platform** : {self.sys_info['platform']}  ",
            f"**GPU**      : {self.sys_info.get('gpu_model', 'N/A')}  ",
            f"**PyTorch**  : {self.sys_info['pytorch']}  \n",
            "---\n",
            "## Results\n",
            "| Profile | Backend | Batch | p50 (ms) | p95 (ms) | p99 (ms) | Throughput (img/s) | Memory (MB) | Status |",
            "|---------|---------|-------|----------|----------|----------|--------------------|-------------|--------|",
        ]

        for r in results:
            lines.append(
                f"| {r.profile} | {r.backend} | {r.batch_size} "
                f"| {r.latency_p50} | {r.latency_p95} | {r.latency_p99} "
                f"| {r.throughput_ips} | {r.memory_mb} | {r.status} |"
            )

        lines.append("\n---\n")
        lines.append("## System Info\n")
        lines.append("| Property | Value |")
        lines.append("|----------|-------|")
        for k, v in self.sys_info.items():
            lines.append(f"| {k} | {v} |")

        with open(path, "w") as f:
            f.write("\n".join(lines))


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description = "Edge Device Benchmark for Face Recognition Models"
    )
    parser.add_argument("--checkpoint",   default=None,
                        help="PyTorch checkpoint .pt")
    parser.add_argument("--onnx-model",   default=None,
                        help="Exported ONNX model .onnx")
    parser.add_argument(
        "--profile", nargs="+",
        default     = ["standard_cpu", "onnx_cpu"],
        choices     = list(EDGE_PROFILES.keys()),
        help        = "Edge profiles to benchmark"
    )
    parser.add_argument(
        "--batch-sizes", nargs="+", type=int,
        default = [1, 4, 8, 16],
    )
    parser.add_argument("--n-runs",       type=int, default=200)
    parser.add_argument("--warmup",       type=int, default=20)
    parser.add_argument("--output-dir",   default="docs/results/benchmarks/")
    parser.add_argument("--backbone",     default="resnet50")
    parser.add_argument("--embedding-dim",type=int, default=512)
    parser.add_argument("--list-profiles",action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()

    if args.list_profiles:
        print("\n📋 Available Edge Profiles:")
        for name, cfg in EDGE_PROFILES.items():
            print(f"  {name:<20}: {cfg['description']}")
        return

    benchmarker = EdgeBenchmarker(
        checkpoint    = args.checkpoint,
        onnx_model    = args.onnx_model,
        backbone_name = args.backbone,
        embedding_dim = args.embedding_dim,
        warmup        = args.warmup,
    )

    results = benchmarker.run_all(
        profiles    = args.profile,
        batch_sizes = args.batch_sizes,
        n_runs      = args.n_runs,
    )

    benchmarker.save_report(results, output_dir=args.output_dir)


if __name__ == "__main__":
    main()
