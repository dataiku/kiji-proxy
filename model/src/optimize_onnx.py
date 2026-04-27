"""Serialize an ONNX Runtime optimized FP32 model without quantization."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

project_root = Path(__file__).resolve().parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

try:
    from model.src.onnx_quantization import (
        copy_runtime_artifacts,
        resolve_source_onnx_path,
    )
except ImportError:
    from onnx_quantization import copy_runtime_artifacts, resolve_source_onnx_path


OPTIMIZATION_LEVELS = {
    "disable": "ORT_DISABLE_ALL",
    "basic": "ORT_ENABLE_BASIC",
    "extended": "ORT_ENABLE_EXTENDED",
    "all": "ORT_ENABLE_ALL",
    "layout": "ORT_ENABLE_LAYOUT",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Write an ONNX Runtime optimized FP32 model artifact."
    )
    parser.add_argument(
        "onnx",
        help="Path to an ONNX file, or a directory containing model.onnx.",
    )
    parser.add_argument(
        "--output",
        default="./model/quantization_experiments/ort_optimized_fp32",
        help="Directory where the optimized model.onnx will be written.",
    )
    parser.add_argument(
        "--level",
        choices=sorted(OPTIMIZATION_LEVELS),
        default="all",
        help="ONNX Runtime graph optimization level.",
    )
    parser.add_argument(
        "--provider",
        choices=["cpu", "cuda"],
        default="cpu",
        help="Provider used while serializing the optimized graph.",
    )
    parser.add_argument(
        "--copy-artifacts",
        action="store_true",
        help="Copy tokenizer, label mappings, config, and CRF JSON files too.",
    )
    parser.add_argument(
        "--report",
        default=None,
        help="Optional JSON report path. Defaults to <output>/optimization_report.json.",
    )
    return parser.parse_args()


def optimize_onnx_model(
    onnx_path: str | Path,
    output_path: str | Path,
    *,
    level: str = "all",
    provider: str = "cpu",
    copy_artifacts: bool = False,
) -> dict:
    import onnx
    import onnxruntime as ort

    source_model = resolve_source_onnx_path(onnx_path)
    model_dir = source_model.parent
    output_dir = Path(output_path)
    output_dir.mkdir(parents=True, exist_ok=True)

    if copy_artifacts:
        copy_runtime_artifacts(model_dir, output_dir)

    output_model = output_dir / "model.onnx"
    if output_model.exists():
        output_model.unlink()

    opts = ort.SessionOptions()
    opts.graph_optimization_level = getattr(
        ort.GraphOptimizationLevel,
        OPTIMIZATION_LEVELS[level],
    )
    opts.optimized_model_filepath = str(output_model)

    providers = (
        ["CPUExecutionProvider"]
        if provider == "cpu"
        else ["CUDAExecutionProvider", "CPUExecutionProvider"]
    )
    session = ort.InferenceSession(str(source_model), sess_options=opts, providers=providers)

    if not output_model.exists():
        raise FileNotFoundError(f"ONNX Runtime did not write {output_model}")

    model_onnx = onnx.load(str(output_model))
    input_names = [input_tensor.name for input_tensor in model_onnx.graph.input]
    output_names = [output_tensor.name for output_tensor in model_onnx.graph.output]
    if not input_names or not output_names:
        raise ValueError(f"Invalid optimized ONNX graph: {output_model}")

    return {
        "method": "ort_graph_optimization",
        "level": level,
        "provider": provider,
        "source_model": str(source_model),
        "output_model": str(output_model),
        "session_providers": session.get_providers(),
        "source_size_mb": source_model.stat().st_size / (1024 * 1024),
        "output_size_mb": output_model.stat().st_size / (1024 * 1024),
    }


def main() -> int:
    args = parse_args()
    report = optimize_onnx_model(
        args.onnx,
        args.output,
        level=args.level,
        provider=args.provider,
        copy_artifacts=args.copy_artifacts,
    )

    report_path = (
        Path(args.report) if args.report else Path(args.output) / "optimization_report.json"
    )
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with report_path.open("w") as f:
        json.dump(report, f, indent=2)

    print(f"Optimized model: {report['output_model']}")
    print(f"Level: {report['level']}")
    print(f"Provider: {report['provider']}")
    print(f"Session providers: {report['session_providers']}")
    print(f"Size: {report['source_size_mb']:.2f} MB -> {report['output_size_mb']:.2f} MB")
    print(f"Report: {report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
