"""Run multiple ONNX quantization variants without touching production artifacts."""

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
        QuantizationResult,
        quantize_onnx_model,
    )
except ImportError:
    from onnx_quantization import (
        SUPPORTED_QUANTIZATION_MODES,
        QuantizationResult,
        quantize_onnx_model,
    )


def _load_parity_tools():
    try:
        from model.src.parity_benchmark import (
            format_parity_report,
            run_parity_benchmark,
        )
    except ImportError:
        from parity_benchmark import format_parity_report, run_parity_benchmark

    return format_parity_report, run_parity_benchmark


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Quantize one exported model.onnx with several ONNX Runtime "
            "quantization configs."
        )
    )
    parser.add_argument(
        "--onnx-model",
        default="./model/quantized",
        help="Directory containing model.onnx and runtime artifacts.",
    )
    parser.add_argument(
        "--output-root",
        default="./model/quantization_experiments",
        help="Directory where per-mode experiment folders are written.",
    )
    parser.add_argument(
        "--modes",
        nargs="+",
        choices=SUPPORTED_QUANTIZATION_MODES,
        default=list(SUPPORTED_QUANTIZATION_MODES),
        help="Quantization modes to test.",
    )
    parser.add_argument(
        "--checkpoint",
        default=None,
        help="Optional PyTorch checkpoint directory for parity checks.",
    )
    parser.add_argument(
        "--num-ai4privacy",
        type=int,
        default=0,
        help="Additional ai4privacy samples to include in parity checks.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--language", default=None)
    parser.add_argument(
        "--confidence-threshold",
        type=float,
        default=0.0,
        help="ONNX entity confidence threshold for parity checks.",
    )
    parser.add_argument(
        "--fail-on-parity",
        action="store_true",
        help="Exit non-zero if any requested parity check fails.",
    )
    parser.add_argument(
        "--report",
        default=None,
        help="Optional JSON report path. Defaults to <output-root>/summary.json.",
    )
    return parser.parse_args()


def _mode_output_dir(output_root: Path, mode: str) -> Path:
    return output_root / mode


def _result_record(
    result: QuantizationResult,
    *,
    parity: dict | None,
) -> dict:
    record = result.to_dict()
    if parity is not None:
        record["parity"] = parity
    return record


def run_sweep(args: argparse.Namespace) -> tuple[list[dict], bool]:
    onnx_model = Path(args.onnx_model)
    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    records: list[dict] = []
    all_parity_passed = True

    for mode in args.modes:
        variant_dir = _mode_output_dir(output_root, mode)
        print(f"\n[{mode}] writing {variant_dir}")
        result = quantize_onnx_model(
            onnx_model,
            variant_dir,
            quantization_mode=mode,
            copy_artifacts=True,
        )
        print(f"[{mode}] {result.output_model} ({result.size_mb:.2f} MB)")

        parity_dict = None
        if args.checkpoint:
            format_parity_report, run_parity_benchmark = _load_parity_tools()
            parity_report = run_parity_benchmark(
                args.checkpoint,
                str(variant_dir),
                onnx_file="model_quantized.onnx",
                num_ai4privacy=args.num_ai4privacy,
                seed=args.seed,
                language=args.language,
                confidence_threshold=args.confidence_threshold,
            )
            print(format_parity_report(parity_report))
            parity_dict = parity_report.to_dict()
            all_parity_passed = all_parity_passed and parity_report.passed

        records.append(_result_record(result, parity=parity_dict))

    report_path = Path(args.report) if args.report else output_root / "summary.json"
    with report_path.open("w") as f:
        json.dump(
            {
                "source_onnx_model": str(onnx_model),
                "output_root": str(output_root),
                "results": records,
            },
            f,
            indent=2,
        )
    print(f"\nWrote report: {report_path}")

    return records, all_parity_passed


def main() -> int:
    args = parse_args()
    _, all_parity_passed = run_sweep(args)
    if args.fail_on_parity and not all_parity_passed:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
