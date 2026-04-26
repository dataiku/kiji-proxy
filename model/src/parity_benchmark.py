"""Parity checks between PyTorch checkpoints and exported ONNX artifacts."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

# ``tests.benchmark.run`` lives at repo root when this module is imported from
# the Metaflow ``src`` package.
project_root = Path(__file__).resolve().parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

try:
    from .eval_model import PIIModelLoader
    from .span_decoder import Span
except ImportError:
    from eval_model import PIIModelLoader
    from span_decoder import Span

from tests.benchmark.run import OnnxPIIModel, load_ai4privacy_samples


PARITY_TEXTS = [
    "My name is John Smith and my email is john.smith@email.com.",
    "Call Sarah Johnson at 555-123-4567. She was born on March 15, 1985.",
    "I live at 123 Main Street, Springfield, IL 62701.",
    "SSN: 123-45-6789, DOB: 01/15/1990, passport X1234567.",
    "Please send the reset link to alex.martinez+test@company.org.",
    "Fatima Khaled resides at 2114 Cedar Crescent in Marseille, France.",
    "Driver license F23098719 and zip code 13008 are on the form.",
    "Username jlee_42 logged in with token api_6f9d8c7b2a1e4f0d.",
]


@dataclass
class ParityMismatch:
    """One text whose PyTorch and ONNX spans differ."""

    index: int
    text: str
    pytorch_spans: list[Span]
    onnx_spans: list[Span]


@dataclass
class ParityReport:
    """Summary of a parity check."""

    checkpoint_path: str
    onnx_model_path: str
    onnx_file: str | None
    total: int
    matches: int
    mismatches: list[ParityMismatch]

    @property
    def passed(self) -> bool:
        return not self.mismatches

    def to_dict(self) -> dict:
        data = asdict(self)
        data["passed"] = self.passed
        data["mismatch_count"] = len(self.mismatches)
        return data


def _format_span(text: str, span: Span) -> str:
    start, end, label = span
    return f"{label}:{text[start:end]!r}@{start}:{end}"


def format_parity_report(report: ParityReport, *, max_examples: int = 5) -> str:
    """Return a human-readable parity report."""
    status = "PASS" if report.passed else "FAIL"
    lines = [
        (
            f"Parity {status}: {report.matches}/{report.total} matched "
            f"({report.onnx_file or 'default ONNX'})"
        )
    ]
    for mismatch in report.mismatches[:max_examples]:
        lines.append(f"  Sample {mismatch.index}: {mismatch.text[:120]!r}")
        pytorch = [_format_span(mismatch.text, span) for span in mismatch.pytorch_spans]
        onnx = [_format_span(mismatch.text, span) for span in mismatch.onnx_spans]
        lines.append(f"    PyTorch: {pytorch}")
        lines.append(f"    ONNX:    {onnx}")
    if len(report.mismatches) > max_examples:
        lines.append(f"  ... {len(report.mismatches) - max_examples} more mismatch(es)")
    return "\n".join(lines)


def _load_texts(
    *,
    num_ai4privacy: int,
    seed: int,
    language: str | None,
) -> list[str]:
    texts = list(PARITY_TEXTS)
    if num_ai4privacy <= 0:
        return texts

    samples = load_ai4privacy_samples(num_ai4privacy, seed=seed, language=language)
    texts.extend(sample["text"] for sample in samples)
    return texts


def run_parity_benchmark(
    checkpoint_path: str,
    onnx_model_path: str,
    *,
    onnx_file: str | None = None,
    texts: list[str] | None = None,
    num_ai4privacy: int = 0,
    seed: int = 42,
    language: str | None = None,
    confidence_threshold: float = 0.0,
) -> ParityReport:
    """Compare PyTorch checkpoint spans with ONNX spans on the same texts."""
    sample_texts = texts or _load_texts(
        num_ai4privacy=num_ai4privacy,
        seed=seed,
        language=language,
    )
    if not sample_texts:
        raise ValueError("No parity texts were provided")

    pytorch_model = PIIModelLoader(checkpoint_path)
    pytorch_model.load_model()
    onnx_model = OnnxPIIModel(
        onnx_model_path,
        entity_confidence_threshold=confidence_threshold,
        onnx_filename=onnx_file,
    )

    mismatches: list[ParityMismatch] = []
    matches = 0
    for index, text in enumerate(sample_texts):
        pytorch_spans, _ = pytorch_model.predict_spans(text)
        onnx_spans = onnx_model.predict(text)
        if pytorch_spans == onnx_spans:
            matches += 1
        else:
            mismatches.append(
                ParityMismatch(
                    index=index,
                    text=text,
                    pytorch_spans=pytorch_spans,
                    onnx_spans=onnx_spans,
                )
            )

    return ParityReport(
        checkpoint_path=str(checkpoint_path),
        onnx_model_path=str(onnx_model_path),
        onnx_file=onnx_file,
        total=len(sample_texts),
        matches=matches,
        mismatches=mismatches,
    )


def assert_parity(report: ParityReport) -> None:
    """Raise if a parity report contains mismatches."""
    if not report.passed:
        raise RuntimeError(format_parity_report(report))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare PyTorch checkpoint and ONNX PII span outputs"
    )
    parser.add_argument("--checkpoint", default="./model/trained")
    parser.add_argument("--onnx-model", default="./model/quantized")
    parser.add_argument(
        "--onnx-file",
        default=None,
        help="Specific ONNX file name inside --onnx-model, e.g. model.onnx.",
    )
    parser.add_argument(
        "--num-ai4privacy",
        type=int,
        default=0,
        help="Additional ai4privacy samples to include (0 = fixed canaries only).",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--language", default=None)
    parser.add_argument(
        "--confidence-threshold",
        type=float,
        default=0.0,
        help="ONNX entity confidence threshold. Use 0 for raw decoder parity.",
    )
    parser.add_argument("--report", default=None, help="Optional JSON report path.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    report = run_parity_benchmark(
        args.checkpoint,
        args.onnx_model,
        onnx_file=args.onnx_file,
        num_ai4privacy=args.num_ai4privacy,
        seed=args.seed,
        language=args.language,
        confidence_threshold=args.confidence_threshold,
    )

    print(format_parity_report(report))
    if args.report:
        report_path = Path(args.report)
        report_path.parent.mkdir(parents=True, exist_ok=True)
        with report_path.open("w") as f:
            json.dump(report.to_dict(), f, indent=2)

    return 0 if report.passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
