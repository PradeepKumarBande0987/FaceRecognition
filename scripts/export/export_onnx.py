"""
ONNX Model Export Script.

Exports a trained PyTorch face recognition model to ONNX format.

Uses the NEW PyTorch 2.5+ dynamo-based exporter:
    torch.onnx.export(..., dynamo=True)    ← recommended
    torch.onnx.export(..., dynamo=False)   ← legacy TorchScript fallback

Features:
    • Dynamic batch size support
    • ONNX Runtime verification
    • Model optimization (constant folding)
    • Metadata embedding (model name, version, input specs)
    • Export report generation (dynamo=True)

Targets:
    • ONNX Runtime (CPU / CUDA)
    • TensorRT (via ONNX → TRT conversion)
    • OpenVINO (via ONNX → OV conversion)
    • Edge devices (via ONNX → CoreML / TFLite)

Usage:
    python scripts/export/export_onnx.py \
        --checkpoint  experiments/runs/run_001/checkpoints/best_model.pt \
        --output      models/exported/face_recognition.onnx \
        --batch-size  1 \
        --dynamo \
        --verify \
        --opset       20
"""

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from scripts.train.train_baseline import build_backbone


# ── Config ────────────────────────────────────────────────────────────────────

EXPORT_CONFIG = {
    "image_size"   : (112, 112),
    "embedding_dim": 512,
    "opset_version": 20,              # ONNX opset 20 (latest stable 2025)
    "input_name"   : "face_image",
    "output_name"  : "embedding",
}


# ── Model Wrapper ─────────────────────────────────────────────────────────────

class FaceEmbeddingModel(nn.Module):
    """
    Thin wrapper for the backbone that:
        1. Accepts (B, C, H, W) float32 input
        2. Returns L2-normalized (B, 512) embedding

    This is the exported model — no ArcFace head included.
    """

    def __init__(self, backbone: nn.Module):
        super().__init__()
        self.backbone = backbone

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, 3, 112, 112) float32, normalized

        Returns:
            embedding: (B, 512) float32, L2-normalized
        """
        embedding = self.backbone(x)
        # L2 normalize
        embedding = nn.functional.normalize(embedding, p=2, dim=1)
        return embedding


# ── Checkpoint Loader ─────────────────────────────────────────────────────────

def load_checkpoint(
    checkpoint_path : str,
    backbone_name   : str = "resnet50",
    embedding_dim   : int = 512,
    device          : str = "cpu",
) -> nn.Module:
    """
    Load backbone weights from training checkpoint.

    Args:
        checkpoint_path : path to .pt checkpoint file
        backbone_name   : backbone architecture name
        embedding_dim   : embedding vector size
        device          : device to load on

    Returns:
        Backbone nn.Module in eval mode
    """
    print(f"📂 Loading checkpoint: {checkpoint_path}")

    # ✅ PyTorch 2.6+: weights_only=True is now default
    # Use weights_only=False for full checkpoint (optimizer + config)
    ckpt = torch.load(
        checkpoint_path,
        map_location    = device,
        weights_only    = False,    # checkpoint contains non-tensor state
    )

    # Extract config if available
    ckpt_config = ckpt.get("config", {})
    backbone_name  = ckpt_config.get("backbone", backbone_name)
    embedding_dim  = ckpt_config.get("embedding_dim", embedding_dim)

    print(f"   Backbone     : {backbone_name}")
    print(f"   Embedding dim: {embedding_dim}")
    print(f"   Epoch        : {ckpt.get('epoch', 'unknown')}")

    # Build backbone
    backbone = build_backbone(
        name          = backbone_name,
        embedding_dim = embedding_dim,
    )

    # Load state dict (handle DDP prefix if needed)
    state_dict = ckpt.get("model_state_dict", ckpt)
    # Strip "module." prefix from DDP-trained models
    state_dict = {
        k.replace("module.", ""): v
        for k, v in state_dict.items()
    }

    backbone.load_state_dict(state_dict)
    backbone.eval()

    print("   ✅ Weights loaded")
    return backbone


# ── ONNX Exporter ─────────────────────────────────────────────────────────────

class ONNXExporter:
    """
    Exports a face recognition model to ONNX format.

    Supports:
        • dynamo=True  : new torch.export-based exporter (PyTorch 2.5+)
        • dynamo=False : legacy TorchScript exporter (fallback)

    Usage:
        exporter = ONNXExporter(
            checkpoint   = "experiments/runs/best/checkpoints/best_model.pt",
            output_path  = "models/exported/face_recognition.onnx",
        )
        exporter.export()
        exporter.verify()
        exporter.benchmark()
    """

    def __init__(
        self,
        checkpoint   : str,
        output_path  : str,
        batch_size   : int  = 1,
        dynamic_batch: bool = True,
        use_dynamo   : bool = True,
        opset        : int  = 20,
        verify       : bool = True,
        optimize     : bool = True,
        device       : str  = "cpu",
        backbone     : str  = "resnet50",
        embedding_dim: int  = 512,
    ):
        self.checkpoint    = checkpoint
        self.output_path   = Path(output_path)
        self.batch_size    = batch_size
        self.dynamic_batch = dynamic_batch
        self.use_dynamo    = use_dynamo
        self.opset         = opset
        self.do_verify     = verify
        self.optimize      = optimize
        self.device        = device
        self.backbone_name = backbone
        self.embedding_dim = embedding_dim

        self.output_path.parent.mkdir(parents=True, exist_ok=True)

    # ── Export ────────────────────────────────────────────────────────────────

    def export(self) -> str:
        """
        Run ONNX export.

        Returns:
            Path to exported .onnx file
        """
        print(f"\n🚀 ONNX Export")
        print(f"   Checkpoint : {self.checkpoint}")
        print(f"   Output     : {self.output_path}")
        print(f"   Dynamo     : {self.use_dynamo}")
        print(f"   Opset      : {self.opset}")
        print(f"   Batch size : {self.batch_size}")

        # ── Load model ────────────────────────────────────────────────────
        backbone = load_checkpoint(
            checkpoint_path = self.checkpoint,
            backbone_name   = self.backbone_name,
            embedding_dim   = self.embedding_dim,
            device          = self.device,
        )
        model = FaceEmbeddingModel(backbone).to(self.device)
        model.eval()

        # ── Dummy input ───────────────────────────────────────────────────
        H, W = EXPORT_CONFIG["image_size"]
        dummy_input = torch.randn(
            self.batch_size, 3, H, W,
            device = self.device,
        )

        print(f"\n   Input shape : {list(dummy_input.shape)}")
        start = time.perf_counter()

        # ── Export ────────────────────────────────────────────────────────
        if self.use_dynamo:
            self._export_dynamo(model, dummy_input)
        else:
            self._export_torchscript(model, dummy_input)

        elapsed = time.perf_counter() - start
        size_mb = self.output_path.stat().st_size / 1e6

        print(f"\n✅ Export complete:")
        print(f"   File       : {self.output_path}")
        print(f"   Size       : {size_mb:.1f} MB")
        print(f"   Time       : {elapsed:.1f}s")

        # ── Verify ────────────────────────────────────────────────────────
        if self.do_verify:
            self.verify(model, dummy_input)

        return str(self.output_path)

    # ── Dynamo Exporter (recommended PyTorch 2.5+) ────────────────────────────

    def _export_dynamo(self, model: nn.Module, dummy_input: torch.Tensor):
        """
        Export using torch.onnx.export(..., dynamo=True).

        This is the recommended approach in PyTorch 2.5+.
        Uses torch.export under the hood for accurate graph capture.
        """
        print("\n  📦 Using dynamo=True exporter (PyTorch 2.5+ recommended)")

        H, W = EXPORT_CONFIG["image_size"]

        # Dynamic shapes for batch dimension
        dynamic_shapes = None
        if self.dynamic_batch:
            dynamic_shapes = {
                "face_image": {0: torch.export.Dim("batch_size", min=1, max=512)}
            }

        # ✅ New API: torch.onnx.export(..., dynamo=True)
        onnx_program = torch.onnx.export(
            model,
            args            = (dummy_input,),
            f               = None,                     # return ONNXProgram first
            input_names     = [EXPORT_CONFIG["input_name"]],
            output_names    = [EXPORT_CONFIG["output_name"]],
            opset_version   = self.opset,
            dynamo          = True,                     # ✅ new exporter
            optimize        = self.optimize,
            verify          = False,                    # we verify manually
            dynamic_shapes  = dynamic_shapes,
            report          = True,                     # generates markdown report
            artifacts_dir   = str(self.output_path.parent),
        )

        # Save to file
        onnx_program.save(str(self.output_path))
        print(f"  ✅ Saved: {self.output_path}")

    # ── TorchScript Exporter (legacy fallback) ────────────────────────────────

    def _export_torchscript(self, model: nn.Module, dummy_input: torch.Tensor):
        """
        Fallback: Export using legacy TorchScript path (dynamo=False).

        Use when dynamo=True encounters unsupported operators.
        """
        print("\n  📦 Using dynamo=False exporter (TorchScript legacy)")

        H, W = EXPORT_CONFIG["image_size"]

        dynamic_axes = None
        if self.dynamic_batch:
            dynamic_axes = {
                EXPORT_CONFIG["input_name"] : {0: "batch_size"},
                EXPORT_CONFIG["output_name"]: {0: "batch_size"},
            }

        torch.onnx.export(
            model,
            dummy_input,
            str(self.output_path),
            input_names         = [EXPORT_CONFIG["input_name"]],
            output_names        = [EXPORT_CONFIG["output_name"]],
            opset_version       = self.opset,
            dynamo              = False,             # legacy path
            do_constant_folding = self.optimize,
            dynamic_axes        = dynamic_axes,
            export_params       = True,
        )
        print(f"  ✅ Saved: {self.output_path}")

    # ── Verify ────────────────────────────────────────────────────────────────

    def verify(
        self,
        model       : nn.Module,
        dummy_input : torch.Tensor,
    ) -> bool:
        """
        Verify ONNX model output matches PyTorch output.

        Runs both models on identical input and checks:
            • Output shape matches
            • Max absolute difference < 1e-4

        Args:
            model       : original PyTorch model
            dummy_input : example input tensor

        Returns:
            True if verification passed
        """
        try:
            import onnx
            import onnxruntime as ort
        except ImportError:
            print("\n⚠️  onnx / onnxruntime not installed. Skipping verify.")
            print("   pip install onnx onnxruntime")
            return False

        print("\n🔍 Verifying ONNX model...")

        # Check ONNX model validity
        onnx_model = onnx.load(str(self.output_path))
        onnx.checker.check_model(onnx_model)
        print("  ✅ ONNX model is well-formed")

        # PyTorch output
        with torch.no_grad():
            pt_output = model(dummy_input).cpu().numpy()

        # ONNX Runtime output
        ort_session = ort.InferenceSession(
            str(self.output_path),
            providers = ["CPUExecutionProvider"],
        )
        ort_inputs  = {
            EXPORT_CONFIG["input_name"]: dummy_input.cpu().numpy()
        }
        ort_output  = ort_session.run(None, ort_inputs)[0]

        # Compare
        max_diff = np.max(np.abs(pt_output - ort_output))
        shape_match = pt_output.shape == ort_output.shape

        print(f"  PyTorch output shape : {pt_output.shape}")
        print(f"  ONNX RT output shape : {ort_output.shape}")
        print(f"  Shape match          : {'✅' if shape_match else '❌'}")
        print(f"  Max abs difference   : {max_diff:.2e}")

        if max_diff < 1e-4 and shape_match:
            print("  ✅ Verification PASSED")
            return True
        else:
            print("  ❌ Verification FAILED — outputs diverge")
            return False

    # ── Benchmark ─────────────────────────────────────────────────────────────

    def benchmark(self, n_runs: int = 100, warmup: int = 10):
        """
        Benchmark ONNX Runtime inference latency.

        Args:
            n_runs : number of timed runs
            warmup : warmup runs (excluded from timing)

        Prints:
            p50, p90, p95, p99 latency in milliseconds
        """
        try:
            import onnxruntime as ort
        except ImportError:
            print("⚠️  onnxruntime not installed. Skipping benchmark.")
            return

        print(f"\n⏱️  ONNX Runtime Benchmark ({n_runs} runs)")

        H, W   = EXPORT_CONFIG["image_size"]
        dummy  = np.random.randn(
            self.batch_size, 3, H, W
        ).astype(np.float32)

        session = ort.InferenceSession(
            str(self.output_path),
            providers = ["CPUExecutionProvider"],
        )
        inputs = {EXPORT_CONFIG["input_name"]: dummy}

        # Warmup
        for _ in range(warmup):
            session.run(None, inputs)

        # Timed runs
        latencies = []
        for _ in range(n_runs):
            t0 = time.perf_counter()
            session.run(None, inputs)
            latencies.append((time.perf_counter() - t0) * 1000)

        latencies = np.array(latencies)
        print(f"   Batch size : {self.batch_size}")
        print(f"   p50 latency: {np.percentile(latencies, 50):.2f} ms")
        print(f"   p90 latency: {np.percentile(latencies, 90):.2f} ms")
        print(f"   p95 latency: {np.percentile(latencies, 95):.2f} ms")
        print(f"   p99 latency: {np.percentile(latencies, 99):.2f} ms")
        print(f"   Mean       : {latencies.mean():.2f} ms")


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description = "Export Face Recognition Model to ONNX"
    )
    parser.add_argument(
        "--checkpoint",
        required = True,
        help     = "Path to .pt training checkpoint"
    )
    parser.add_argument(
        "--output",
        default = "models/exported/face_recognition.onnx",
        help    = "Output .onnx file path"
    )
    parser.add_argument("--batch-size",    type=int,  default=1)
    parser.add_argument("--dynamic-batch", action="store_true", default=True)
    parser.add_argument("--dynamo",        action="store_true", default=True,
                        help="Use dynamo=True exporter (PyTorch 2.5+ recommended)")
    parser.add_argument("--no-dynamo",     action="store_true",
                        help="Use legacy TorchScript exporter")
    parser.add_argument("--opset",         type=int,  default=20)
    parser.add_argument("--verify",        action="store_true", default=True)
    parser.add_argument("--no-verify",     action="store_true")
    parser.add_argument("--benchmark",     action="store_true")
    parser.add_argument("--n-benchmark",   type=int, default=100)
    parser.add_argument("--device",        default="cpu")
    parser.add_argument("--backbone",      default="resnet50")
    parser.add_argument("--embedding-dim", type=int, default=512)
    return parser.parse_args()


def main():
    args = parse_args()

    exporter = ONNXExporter(
        checkpoint    = args.checkpoint,
        output_path   = args.output,
        batch_size    = args.batch_size,
        dynamic_batch = args.dynamic_batch,
        use_dynamo    = not args.no_dynamo,
        opset         = args.opset,
        verify        = not args.no_verify,
        device        = args.device,
        backbone      = args.backbone,
        embedding_dim = args.embedding_dim,
    )

    exporter.export()

    if args.benchmark:
        exporter.benchmark(n_runs=args.n_benchmark)


if __name__ == "__main__":
    main()
