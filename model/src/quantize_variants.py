"""Build multiple quantized variants of the PII detection ONNX model.

Each variant is written to ``<output>/<variant>/`` with the same supporting
files as ``model/quantized/`` (tokenizer, label_mappings, crf_transitions,
config) so that ``tests/benchmark/run.py`` can be pointed at any variant
directory via ``--model-path``.

Usage:
    uv run python -m model.src.quantize_variants \\
        --source ./model/quantized \\
        --output ./model/quant_variants \\
        --variants fp32,fp16,int8_dyn_default,int8_dyn_avx512_vnni,\\
                   int8_dyn_avx2,int8_dyn_q8,int8_static

Variants:
    fp32                  Reference, no quantization (copy of source).
    fp16                  Float16 conversion via ORT's converter.
    int8_dyn_default      onnxruntime.quantize_dynamic with library defaults.
    int8_dyn_avx512_vnni  Optimum AVX-512 VNNI dynamic INT8 (current prod).
    int8_dyn_avx2         Optimum AVX2 dynamic INT8.
    int8_dyn_avx512       Optimum AVX-512 (non-VNNI) dynamic INT8.
    int8_static           Optimum static INT8 calibrated on ai4privacy samples.
    int4_rtn_block32      MatMul 4-bit weight-only RTN, block_size=32 (higher accuracy).
    int4_rtn_block128     MatMul 4-bit weight-only RTN, block_size=128 (smaller).
    int4_hqq_block64      MatMul 4-bit weight-only HQQ, block_size=64.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
from pathlib import Path

ALL_VARIANTS = [
    "fp32",
    "fp16",
    "int8_dyn_default",
    "int8_dyn_avx512_vnni",
    "int8_dyn_avx2",
    "int8_dyn_avx512",
    "int8_static",
    "int4_rtn_block32",
    "int4_rtn_block128",
    "int4_hqq_block64",
]

SUPPORT_FILES = [
    "tokenizer.json",
    "tokenizer_config.json",
    "vocab.txt",
    "spm.model",
    "special_tokens_map.json",
    "added_tokens.json",
    "label_mappings.json",
    "crf_transitions.json",
    "config.json",
    "ort_config.json",
    "model_manifest.json",
]


def copy_support_files(source: Path, dest: Path) -> None:
    """Copy tokenizer + label + config files alongside the variant ONNX."""
    dest.mkdir(parents=True, exist_ok=True)
    for name in SUPPORT_FILES:
        src = source / name
        if src.exists():
            shutil.copy2(src, dest / name)


def report_size(onnx_path: Path) -> float:
    size_mb = onnx_path.stat().st_size / (1024 * 1024)
    print(f"   size: {size_mb:.1f} MB")
    return size_mb


def build_fp32(source: Path, dest: Path) -> None:
    print(f"[fp32] copying {source}/model.onnx -> {dest}")
    copy_support_files(source, dest)
    shutil.copy2(source / "model.onnx", dest / "model.onnx")
    report_size(dest / "model.onnx")


def build_fp16(source: Path, dest: Path) -> None:
    print(f"[fp16] converting {source}/model.onnx to float16 -> {dest}")
    import onnx
    from onnxruntime.transformers.float16 import convert_float_to_float16

    copy_support_files(source, dest)
    model = onnx.load(str(source / "model.onnx"))
    fp16_model = convert_float_to_float16(model, keep_io_types=True)
    onnx.save(fp16_model, str(dest / "model.onnx"))
    report_size(dest / "model.onnx")


def build_int8_dyn_default(source: Path, dest: Path) -> None:
    print(f"[int8_dyn_default] quantize_dynamic defaults -> {dest}")
    from onnxruntime.quantization import QuantType, quantize_dynamic

    copy_support_files(source, dest)
    quantize_dynamic(
        model_input=str(source / "model.onnx"),
        model_output=str(dest / "model.onnx"),
        weight_type=QuantType.QInt8,
        per_channel=False,
    )
    report_size(dest / "model.onnx")


def _optimum_dynamic(source: Path, dest: Path, mode: str) -> None:
    """Run an Optimum ``AutoQuantizationConfig`` preset."""
    from optimum.onnxruntime import ORTQuantizer
    from optimum.onnxruntime.configuration import AutoQuantizationConfig

    copy_support_files(source, dest)
    quantizer = ORTQuantizer.from_pretrained(str(source), file_name="model.onnx")

    if mode == "avx512_vnni":
        qconfig = AutoQuantizationConfig.avx512_vnni(is_static=False)
    elif mode == "avx2":
        qconfig = AutoQuantizationConfig.avx2(is_static=False)
    elif mode == "avx512":
        qconfig = AutoQuantizationConfig.avx512(is_static=False)
    else:
        raise ValueError(f"unknown optimum mode: {mode}")

    quantizer.quantize(save_dir=str(dest), quantization_config=qconfig)
    quantized = dest / "model_quantized.onnx"
    if quantized.exists():
        quantized.rename(dest / "model.onnx")
    report_size(dest / "model.onnx")


def build_int8_dyn_avx512_vnni(source: Path, dest: Path) -> None:
    print(f"[int8_dyn_avx512_vnni] optimum avx512_vnni -> {dest}")
    _optimum_dynamic(source, dest, "avx512_vnni")


def build_int8_dyn_avx2(source: Path, dest: Path) -> None:
    print(f"[int8_dyn_avx2] optimum avx2 -> {dest}")
    _optimum_dynamic(source, dest, "avx2")


def build_int8_dyn_avx512(source: Path, dest: Path) -> None:
    print(f"[int8_dyn_avx512] optimum avx512 (non-VNNI) -> {dest}")
    _optimum_dynamic(source, dest, "avx512")


def build_int8_static(source: Path, dest: Path, num_calibration: int = 200) -> None:
    print(f"[int8_static] optimum static INT8 with {num_calibration} calibration samples -> {dest}")
    from optimum.onnxruntime import ORTQuantizer
    from optimum.onnxruntime.configuration import AutoCalibrationConfig, AutoQuantizationConfig
    from transformers import AutoTokenizer

    sys.path.insert(0, str(Path(__file__).parent.parent.parent))
    from tests.benchmark.run import load_ai4privacy_samples  # noqa: E402

    copy_support_files(source, dest)

    tokenizer = AutoTokenizer.from_pretrained(str(source))
    samples = load_ai4privacy_samples(num_calibration, seed=42, language=None)
    print(f"   loaded {len(samples)} calibration samples")

    quantizer = ORTQuantizer.from_pretrained(str(source), file_name="model.onnx")

    def preprocess(example):
        out = tokenizer(
            example["text"],
            truncation=True,
            max_length=512,
            padding="max_length",
        )
        # The DeBERTa-v3 ONNX export uses only input_ids + attention_mask;
        # ``token_type_ids`` (added by the tokenizer) would break ORT validation.
        return {"input_ids": out["input_ids"], "attention_mask": out["attention_mask"]}

    from datasets import Dataset

    calibration_dataset = Dataset.from_list(
        [{"text": s["text"]} for s in samples]
    ).map(preprocess, remove_columns=["text"])

    calibration_config = AutoCalibrationConfig.minmax(calibration_dataset)
    qconfig = AutoQuantizationConfig.avx512_vnni(is_static=True, per_channel=True)

    print("   running calibration ...")
    ranges = quantizer.fit(
        dataset=calibration_dataset,
        calibration_config=calibration_config,
        operators_to_quantize=qconfig.operators_to_quantize,
    )
    quantizer.quantize(
        save_dir=str(dest),
        quantization_config=qconfig,
        calibration_tensors_range=ranges,
    )
    quantized = dest / "model_quantized.onnx"
    if quantized.exists():
        quantized.rename(dest / "model.onnx")
    report_size(dest / "model.onnx")


def _matmul_nbits_quantize(
    source: Path,
    dest: Path,
    *,
    block_size: int,
    algorithm: str,
    is_symmetric: bool = True,
) -> None:
    """Run ONNX Runtime's MatMulNBitsQuantizer (weight-only 4-bit on MatMul ops)."""
    import onnx
    from onnxruntime.quantization.matmul_nbits_quantizer import (
        HQQWeightOnlyQuantConfig,
        MatMulNBitsQuantizer,
        RTNWeightOnlyQuantConfig,
    )

    copy_support_files(source, dest)

    if algorithm == "rtn":
        algo_config = RTNWeightOnlyQuantConfig()
    elif algorithm == "hqq":
        algo_config = HQQWeightOnlyQuantConfig(block_size=block_size)
    else:
        raise ValueError(f"unknown nbits algorithm: {algorithm}")

    quantizer = MatMulNBitsQuantizer(
        model=str(source / "model.onnx"),
        bits=4,
        block_size=block_size,
        is_symmetric=is_symmetric,
        algo_config=algo_config,
    )
    quantizer.process()
    onnx.save(
        quantizer.model.model,
        str(dest / "model.onnx"),
        save_as_external_data=False,
    )
    report_size(dest / "model.onnx")


def build_int4_rtn_block32(source: Path, dest: Path) -> None:
    print(f"[int4_rtn_block32] MatMulNBits RTN block_size=32 -> {dest}")
    _matmul_nbits_quantize(source, dest, block_size=32, algorithm="rtn")


def build_int4_rtn_block128(source: Path, dest: Path) -> None:
    print(f"[int4_rtn_block128] MatMulNBits RTN block_size=128 -> {dest}")
    _matmul_nbits_quantize(source, dest, block_size=128, algorithm="rtn")


def build_int4_hqq_block64(source: Path, dest: Path) -> None:
    print(f"[int4_hqq_block64] MatMulNBits HQQ block_size=64 -> {dest}")
    _matmul_nbits_quantize(source, dest, block_size=64, algorithm="hqq")


VARIANT_BUILDERS = {
    "fp32": build_fp32,
    "fp16": build_fp16,
    "int8_dyn_default": build_int8_dyn_default,
    "int8_dyn_avx512_vnni": build_int8_dyn_avx512_vnni,
    "int8_dyn_avx2": build_int8_dyn_avx2,
    "int8_dyn_avx512": build_int8_dyn_avx512,
    "int8_static": build_int8_static,
    "int4_rtn_block32": build_int4_rtn_block32,
    "int4_rtn_block128": build_int4_rtn_block128,
    "int4_hqq_block64": build_int4_hqq_block64,
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source", default="./model/quantized")
    p.add_argument("--output", default="./model/quant_variants")
    p.add_argument(
        "--variants",
        default="all",
        help="Comma-separated list of variant names, or 'all'. "
        f"Available: {','.join(ALL_VARIANTS)}",
    )
    p.add_argument(
        "--calibration-samples",
        type=int,
        default=200,
        help="Number of calibration samples for int8_static (default: 200)",
    )
    return p.parse_args()


def main() -> int:
    args = parse_args()
    source = Path(args.source).resolve()
    output = Path(args.output).resolve()

    if not (source / "model.onnx").exists():
        print(f"ERROR: {source}/model.onnx does not exist", file=sys.stderr)
        return 1

    requested = (
        ALL_VARIANTS
        if args.variants == "all"
        else [v.strip() for v in args.variants.split(",") if v.strip()]
    )
    unknown = [v for v in requested if v not in VARIANT_BUILDERS]
    if unknown:
        print(f"ERROR: unknown variants: {unknown}", file=sys.stderr)
        print(f"Known: {ALL_VARIANTS}", file=sys.stderr)
        return 1

    output.mkdir(parents=True, exist_ok=True)
    summary = []
    for name in requested:
        dest = output / name
        if (dest / "model.onnx").exists():
            print(f"\n=== {name}: already exists at {dest}, skipping ===")
            size_mb = (dest / "model.onnx").stat().st_size / (1024 * 1024)
            summary.append({"variant": name, "status": "skipped", "size_mb": round(size_mb, 1)})
            continue
        print(f"\n=== {name} -> {dest} ===")
        t0 = time.perf_counter()
        try:
            if name == "int8_static":
                VARIANT_BUILDERS[name](source, dest, args.calibration_samples)
            else:
                VARIANT_BUILDERS[name](source, dest)
            elapsed = time.perf_counter() - t0
            size_mb = (dest / "model.onnx").stat().st_size / (1024 * 1024)
            summary.append(
                {"variant": name, "status": "ok", "size_mb": round(size_mb, 1), "build_seconds": round(elapsed, 1)}
            )
        except Exception as e:
            print(f"FAILED: {e}")
            import traceback

            traceback.print_exc()
            summary.append({"variant": name, "status": "failed", "error": str(e)})
            shutil.rmtree(dest, ignore_errors=True)

    print("\n" + "=" * 60)
    print("Variant build summary:")
    print("=" * 60)
    for row in summary:
        print(json.dumps(row))

    failed = [r for r in summary if r.get("status") == "failed"]
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
