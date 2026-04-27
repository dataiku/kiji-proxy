"""Post-training quantization for an existing ONNX model."""

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
        SUPPORTED_QUANTIZATION_MODES,
        quantize_onnx_model,
        quantize_onnx_model_dynamic,
    )
except ImportError:
    from onnx_quantization import (
        SUPPORTED_QUANTIZATION_MODES,
        quantize_onnx_model,
        quantize_onnx_model_dynamic,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Quantize an existing ONNX model without running training."
    )
    parser.add_argument(
        "onnx",
        help="Path to an ONNX file, or a directory containing model.onnx.",
    )
    parser.add_argument(
        "--output",
        default="./model/quantization_experiments/post_quantized",
        help="Directory where model_quantized.onnx will be written.",
    )
    parser.add_argument(
        "--method",
        choices=["optimum", "ort-dynamic"],
        default="optimum",
        help="Quantization backend to use.",
    )
    parser.add_argument(
        "--mode",
        choices=SUPPORTED_QUANTIZATION_MODES,
        default="avx512_vnni",
        help="Optimum quantization config when --method=optimum.",
    )
    parser.add_argument(
        "--dynamic-mode-name",
        default=None,
        help="Label used in reports when --method=ort-dynamic.",
    )
    parser.add_argument(
        "--op-types",
        nargs="+",
        default=["MatMul"],
        help="Operator types to quantize when --method=ort-dynamic.",
    )
    parser.add_argument(
        "--weight-type",
        choices=["QInt8", "QUInt8", "QInt16", "QUInt16", "QInt4", "QUInt4"],
        default="QInt8",
        help="ONNX Runtime weight type when --method=ort-dynamic.",
    )
    parser.add_argument(
        "--per-channel",
        action="store_true",
        help="Enable per-channel dynamic quantization when --method=ort-dynamic.",
    )
    parser.add_argument(
        "--reduce-range",
        action="store_true",
        help="Enable reduced-range dynamic quantization when --method=ort-dynamic.",
    )
    parser.add_argument(
        "--copy-artifacts",
        action="store_true",
        help="Copy tokenizer, label mappings, config, and CRF JSON files too.",
    )
    parser.add_argument(
        "--report",
        default=None,
        help="Optional JSON report path. Defaults to <output>/quantization_report.json.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.method == "optimum":
        result = quantize_onnx_model(
            args.onnx,
            args.output,
            quantization_mode=args.mode,
            copy_artifacts=args.copy_artifacts,
        )
    else:
        dynamic_mode_name = args.dynamic_mode_name or (
            "ort_dynamic_"
            f"{'-'.join(args.op_types)}_"
            f"{args.weight_type}_"
            f"per_channel_{args.per_channel}_"
            f"reduce_range_{args.reduce_range}"
        )
        result = quantize_onnx_model_dynamic(
            args.onnx,
            args.output,
            mode=dynamic_mode_name,
            op_types_to_quantize=args.op_types,
            weight_type=args.weight_type,
            per_channel=args.per_channel,
            reduce_range=args.reduce_range,
            copy_artifacts=args.copy_artifacts,
        )

    report_path = (
        Path(args.report) if args.report else Path(args.output) / "quantization_report.json"
    )
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with report_path.open("w") as f:
        json.dump(result.to_dict(), f, indent=2)

    print(f"Quantized model: {result.output_model}")
    print(f"Method: {result.method}")
    print(f"Mode: {result.mode}")
    print(f"Size: {result.size_mb:.2f} MB")
    print(f"Report: {report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
