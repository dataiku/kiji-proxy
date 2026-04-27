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
        quantize_onnx_model_static,
    )
except ImportError:
    from onnx_quantization import (
        SUPPORTED_QUANTIZATION_MODES,
        quantize_onnx_model,
        quantize_onnx_model_dynamic,
        quantize_onnx_model_static,
    )


def load_calibration_texts(
    *,
    num_samples: int,
    seed: int,
    language: str | None,
) -> list[str]:
    """Load representative benchmark texts for static quantization calibration."""
    from tests.benchmark.run import load_ai4privacy_samples

    samples = load_ai4privacy_samples(num_samples, seed=seed, language=language)
    return [sample["text"] for sample in samples]


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
        choices=["optimum", "ort-dynamic", "ort-static"],
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
        help="Label used in reports for ONNX Runtime quantization variants.",
    )
    parser.add_argument(
        "--op-types",
        nargs="+",
        default=["MatMul"],
        help="Operator types to quantize for ONNX Runtime quantization.",
    )
    parser.add_argument(
        "--weight-type",
        choices=["QInt8", "QUInt8", "QInt16", "QUInt16", "QInt4", "QUInt4"],
        default="QInt8",
        help="ONNX Runtime weight type when --method is ort-dynamic or ort-static.",
    )
    parser.add_argument(
        "--activation-type",
        choices=["QInt8", "QUInt8", "QInt16", "QUInt16"],
        default="QUInt8",
        help="ONNX Runtime activation type when --method=ort-static.",
    )
    parser.add_argument(
        "--quant-format",
        choices=["QDQ", "QOperator"],
        default="QDQ",
        help="ONNX Runtime quantization format when --method=ort-static.",
    )
    parser.add_argument(
        "--calibration-method",
        choices=["MinMax", "Entropy", "Percentile", "Distribution"],
        default="MinMax",
        help="Calibration method when --method=ort-static.",
    )
    parser.add_argument(
        "--calibration-samples",
        type=int,
        default=50,
        help="Number of ai4privacy samples to use when --method=ort-static.",
    )
    parser.add_argument(
        "--calibration-seed",
        type=int,
        default=123,
        help="Shuffle seed for static quantization calibration samples.",
    )
    parser.add_argument(
        "--calibration-language",
        default=None,
        help="Optional ai4privacy language filter for static calibration.",
    )
    parser.add_argument(
        "--per-channel",
        action="store_true",
        help="Enable per-channel quantization for ONNX Runtime quantization.",
    )
    parser.add_argument(
        "--reduce-range",
        action="store_true",
        help="Enable reduced-range quantization for ONNX Runtime quantization.",
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
    elif args.method == "ort-dynamic":
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
    else:
        calibration_texts = load_calibration_texts(
            num_samples=args.calibration_samples,
            seed=args.calibration_seed,
            language=args.calibration_language,
        )
        static_mode_name = args.dynamic_mode_name or (
            "ort_static_"
            f"{args.quant_format}_"
            f"{args.calibration_method}_"
            f"{'-'.join(args.op_types)}_"
            f"{args.activation_type}_{args.weight_type}_"
            f"per_channel_{args.per_channel}_"
            f"reduce_range_{args.reduce_range}"
        )
        result = quantize_onnx_model_static(
            args.onnx,
            args.output,
            mode=static_mode_name,
            calibration_texts=calibration_texts,
            quant_format=args.quant_format,
            calibration_method=args.calibration_method,
            activation_type=args.activation_type,
            weight_type=args.weight_type,
            op_types_to_quantize=args.op_types,
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
