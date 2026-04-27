"""Utilities for quantizing exported ONNX PII models."""

from __future__ import annotations

import json
import shutil
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Literal

QuantizationMode = Literal["avx512_vnni", "avx512", "avx2"]

SUPPORTED_QUANTIZATION_MODES: tuple[QuantizationMode, ...] = (
    "avx512_vnni",
    "avx512",
    "avx2",
)

RUNTIME_ARTIFACT_SUFFIXES = {
    ".json",
    ".model",
    ".txt",
}


@dataclass(frozen=True)
class QuantizationResult:
    """Summary of one quantized ONNX artifact."""

    mode: str
    source_model: str
    output_model: str
    size_mb: float
    method: str = "optimum"

    def to_dict(self) -> dict:
        return asdict(self)


def resolve_onnx_model_dir(onnx_path: str | Path) -> Path:
    """Return the directory containing the source ONNX model."""
    path = Path(onnx_path)
    return path.parent if path.is_file() else path


def resolve_source_onnx_path(onnx_path: str | Path) -> Path:
    """Return the source ONNX file from either a file path or model directory."""
    path = Path(onnx_path)
    if path.is_file():
        return path

    model_path = path / "model.onnx"
    if model_path.exists():
        return model_path

    raise FileNotFoundError(
        f"Expected an ONNX file or a directory containing model.onnx: {path}"
    )


def build_quantization_config(mode: str):
    """Build an Optimum ONNX Runtime quantization config for a named mode."""
    from optimum.onnxruntime.configuration import AutoQuantizationConfig

    if mode == "avx512_vnni":
        return AutoQuantizationConfig.avx512_vnni(is_static=False)
    if mode == "avx512":
        return AutoQuantizationConfig.avx512(is_static=False)
    if mode == "avx2":
        return AutoQuantizationConfig.avx2(is_static=False)

    supported = ", ".join(SUPPORTED_QUANTIZATION_MODES)
    raise ValueError(f"Unknown quantization mode: {mode}. Supported: {supported}")


def copy_runtime_artifacts(source_dir: str | Path, output_dir: str | Path) -> None:
    """Copy tokenizer, labels, config, and CRF metadata needed for inference."""
    source = Path(source_dir)
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)

    for artifact in source.iterdir():
        if artifact.is_dir() or artifact.suffix not in RUNTIME_ARTIFACT_SUFFIXES:
            continue
        shutil.copy2(artifact, output / artifact.name)


def quantize_onnx_model(
    onnx_path: str | Path,
    output_path: str | Path,
    *,
    quantization_mode: str = "avx512_vnni",
    copy_artifacts: bool = False,
) -> QuantizationResult:
    """Quantize ``model.onnx`` into ``output_path`` as ``model_quantized.onnx``."""
    import onnx
    from optimum.onnxruntime import ORTQuantizer

    source_model = resolve_source_onnx_path(onnx_path)
    model_dir = source_model.parent
    output_dir = Path(output_path)
    output_dir.mkdir(parents=True, exist_ok=True)

    if copy_artifacts:
        copy_runtime_artifacts(model_dir, output_dir)

    old_quantized = output_dir / "model_quantized.onnx"
    if old_quantized.exists():
        old_quantized.unlink()

    qconfig = build_quantization_config(quantization_mode)
    quantizer = ORTQuantizer.from_pretrained(
        str(model_dir),
        file_name=source_model.name,
    )
    quantizer.quantize(save_dir=str(output_dir), quantization_config=qconfig)

    quantized_model_path = output_dir / "model_quantized.onnx"
    if not quantized_model_path.exists():
        generated_files = sorted(
            path
            for path in output_dir.glob("*quantized*.onnx")
            if path.name != "model.onnx"
        )
        if not generated_files:
            raise FileNotFoundError(
                "Quantization completed but model_quantized.onnx was not written "
                f"to {output_dir}"
            )
        generated_files[0].rename(quantized_model_path)

    model_onnx = onnx.load(str(quantized_model_path))
    input_names = [input_tensor.name for input_tensor in model_onnx.graph.input]
    output_names = [output_tensor.name for output_tensor in model_onnx.graph.output]
    if not input_names or not output_names:
        raise ValueError(f"Invalid quantized ONNX graph: {quantized_model_path}")

    size_mb = quantized_model_path.stat().st_size / (1024 * 1024)
    return QuantizationResult(
        method="optimum",
        mode=quantization_mode,
        source_model=str(source_model),
        output_model=str(quantized_model_path),
        size_mb=size_mb,
    )


def quantize_onnx_model_dynamic(
    onnx_path: str | Path,
    output_path: str | Path,
    *,
    mode: str,
    op_types_to_quantize: list[str],
    weight_type: str = "QInt8",
    per_channel: bool = False,
    reduce_range: bool = False,
    copy_artifacts: bool = False,
) -> QuantizationResult:
    """Quantize with raw ONNX Runtime dynamic quantization controls."""
    import onnx
    from onnxruntime.quantization import QuantType, quantize_dynamic

    source_model = resolve_source_onnx_path(onnx_path)
    model_dir = source_model.parent
    output_dir = Path(output_path)
    output_dir.mkdir(parents=True, exist_ok=True)

    if copy_artifacts:
        copy_runtime_artifacts(model_dir, output_dir)

    quantized_model_path = output_dir / "model_quantized.onnx"
    if quantized_model_path.exists():
        quantized_model_path.unlink()

    try:
        resolved_weight_type = getattr(QuantType, weight_type)
    except AttributeError as exc:
        supported = ", ".join(name for name in QuantType.__members__)
        raise ValueError(
            f"Unknown ONNX Runtime QuantType: {weight_type}. Supported: {supported}"
        ) from exc

    quantize_dynamic(
        model_input=str(source_model),
        model_output=str(quantized_model_path),
        op_types_to_quantize=op_types_to_quantize,
        per_channel=per_channel,
        reduce_range=reduce_range,
        weight_type=resolved_weight_type,
    )

    model_onnx = onnx.load(str(quantized_model_path))
    input_names = [input_tensor.name for input_tensor in model_onnx.graph.input]
    output_names = [output_tensor.name for output_tensor in model_onnx.graph.output]
    if not input_names or not output_names:
        raise ValueError(f"Invalid quantized ONNX graph: {quantized_model_path}")

    size_mb = quantized_model_path.stat().st_size / (1024 * 1024)
    return QuantizationResult(
        method="ort_dynamic",
        mode=mode,
        source_model=str(source_model),
        output_model=str(quantized_model_path),
        size_mb=size_mb,
    )


def write_quantization_report(
    report_path: str | Path,
    results: list[QuantizationResult],
    *,
    extra: dict | None = None,
) -> None:
    """Write a JSON report for quantization experiments."""
    report = {
        "results": [result.to_dict() for result in results],
    }
    if extra:
        report.update(extra)

    path = Path(report_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        json.dump(report, f, indent=2)
