"""Benchmark the ONNX PII model against ai4privacy/pii-masking-300k.

Runs inference directly against the quantized ONNX model (no backend needed)
and reports span-based precision, recall, and F1 per label.

Usage:
    uv run python -m tests.benchmark.run
    uv run python -m tests.benchmark.run --num 5000 --model-path ./model/quantized
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import onnxruntime as ort
from datasets import load_dataset
from transformers import AutoTokenizer

# ---------------------------------------------------------------------------
# Label mapping: ai4privacy labels -> kiji model labels
# Only labels that the kiji model supports are mapped; the rest are skipped.
# ---------------------------------------------------------------------------
AI4PRIVACY_TO_KIJI: dict[str, str] = {
    "GIVENNAME1": "FIRSTNAME",
    "GIVENNAME2": "FIRSTNAME",
    "LASTNAME1": "SURNAME",
    "LASTNAME2": "SURNAME",
    "LASTNAME3": "SURNAME",
    "EMAIL": "EMAIL",
    "TEL": "PHONENUMBER",
    "BOD": "DATEOFBIRTH",
    "SOCIALNUMBER": "SSN",
    "STREET": "STREET",
    "BUILDING": "BUILDINGNUM",
    "CITY": "CITY",
    "STATE": "STATE",
    "POSTCODE": "ZIP",
    "COUNTRY": "COUNTRY",
    "DRIVERLICENSE": "DRIVERLICENSENUM",
    "PASSPORT": "PASSPORTID",
    "IDCARD": "NATIONALID",
    "USERNAME": "USERNAME",
    "PASS": "PASSWORD",
}

# Labels in the ai4privacy dataset that we intentionally skip because the
# kiji model has no equivalent: TIME, DATE, TITLE, SEX, GEOCOORD, IP,
# CARDISSUER, SECADDRESS


# ---------------------------------------------------------------------------
# ONNX model wrapper
# ---------------------------------------------------------------------------

def viterbi_decode(
    logits: np.ndarray,
    transitions: np.ndarray,
    start_transitions: np.ndarray,
    end_transitions: np.ndarray,
) -> list[int]:
    """Viterbi decoding using CRF transition parameters.

    Args:
        logits: Emission scores of shape (seq_len, num_labels).
        transitions: Transition matrix (num_labels, num_labels).
                     transitions[i][j] = score for transitioning from tag i to tag j.
        start_transitions: Start scores (num_labels,).
        end_transitions: End scores (num_labels,).

    Returns:
        Best label sequence as list of label IDs.
    """
    seq_len, num_labels = logits.shape
    # viterbi[t][j] = best score ending in label j at position t
    viterbi = np.full((seq_len, num_labels), -1e9)
    backpointers = np.zeros((seq_len, num_labels), dtype=int)

    # Initialization
    viterbi[0] = start_transitions + logits[0]

    # Forward pass
    for t in range(1, seq_len):
        for j in range(num_labels):
            # Score of arriving at label j from each previous label
            scores = viterbi[t - 1] + transitions[:, j] + logits[t, j]
            backpointers[t, j] = int(np.argmax(scores))
            viterbi[t, j] = scores[backpointers[t, j]]

    # Add end transitions
    final_scores = viterbi[seq_len - 1] + end_transitions
    best_last = int(np.argmax(final_scores))

    # Backtrace
    best_path = [best_last]
    for t in range(seq_len - 1, 0, -1):
        best_path.append(backpointers[t, best_path[-1]])
    best_path.reverse()
    return best_path


class OnnxPIIModel:
    """Thin wrapper around the quantized ONNX model for inference."""

    def __init__(self, model_dir: str):
        model_dir = Path(model_dir)
        onnx_file = model_dir / "model_quantized.onnx"
        if not onnx_file.exists():
            onnx_file = model_dir / "model.onnx"
        if not onnx_file.exists():
            raise FileNotFoundError(f"No ONNX model found in {model_dir}")

        mappings_path = model_dir / "label_mappings.json"
        with mappings_path.open() as f:
            mappings = json.load(f)
        self.id2label: dict[int, str] = {
            int(k): v for k, v in mappings["pii"]["id2label"].items()
        }

        self.tokenizer = AutoTokenizer.from_pretrained(str(model_dir))

        opts = ort.SessionOptions()
        opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        self.session = ort.InferenceSession(
            str(onnx_file),
            sess_options=opts,
            providers=["CPUExecutionProvider"],
        )

        # Load CRF transition parameters for Viterbi decoding
        crf_path = model_dir / "crf_transitions.json"
        if crf_path.exists():
            with crf_path.open() as f:
                crf = json.load(f)
            self.transitions = np.array(crf["transitions"], dtype=np.float32)
            self.start_transitions = np.array(crf["start_transitions"], dtype=np.float32)
            self.end_transitions = np.array(crf["end_transitions"], dtype=np.float32)
        else:
            self.transitions = None
            self.start_transitions = None
            self.end_transitions = None

    def _trim_span(self, text: str, start: int, end: int) -> tuple[int, int]:
        """Trim leading/trailing whitespace and trailing punctuation from a span.

        SentencePiece includes the preceding space in token offsets.
        The model may also extend spans to include trailing sentence
        punctuation (e.g. "April 12, 1988,") which should be stripped.
        This mirrors the Go backend's finalizeEntity trimming.
        """
        while start < end and text[start] in " \t\n\r":
            start += 1
        while end > start and text[end - 1] in " \t\n\r":
            end -= 1
        # Strip trailing sentence punctuation only when it's followed by
        # whitespace or end-of-string (so "yahoo.com" keeps the dot but
        # "1988," at end of phrase loses the comma).
        while end > start and text[end - 1] in ",.;:!?":
            if end < len(text) and text[end] not in " \t\n\r":
                break
            end -= 1
        return start, end

    def predict(self, text: str) -> list[tuple[int, int, str]]:
        """Return list of (start, end, label) spans detected in text."""
        inputs = self.tokenizer(
            text,
            return_tensors="np",
            truncation=True,
            max_length=512,
            return_offsets_mapping=True,
        )
        offset_mapping = inputs.pop("offset_mapping")[0]
        ort_inputs = {
            "input_ids": inputs["input_ids"],
            "attention_mask": inputs["attention_mask"],
        }
        logits = self.session.run(None, ort_inputs)[0]  # [1, seq, labels]

        if self.transitions is not None:
            # Use Viterbi decoding with CRF transition constraints
            seq_logits = logits[0]  # (seq_len, num_labels)
            predictions = viterbi_decode(
                seq_logits,
                self.transitions,
                self.start_transitions,
                self.end_transitions,
            )
        else:
            predictions = np.argmax(logits, axis=-1)[0]

        tokens = self.tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])
        entities: list[tuple[int, int, str]] = []
        cur_label: str | None = None
        cur_start = 0
        cur_end = 0

        for token, pred, offset in zip(tokens, predictions, offset_mapping, strict=True):
            if token in (
                self.tokenizer.cls_token,
                self.tokenizer.sep_token,
                self.tokenizer.pad_token,
            ):
                continue
            label = self.id2label.get(int(pred), "O")
            if label.startswith("B-"):
                if cur_label is not None:
                    s, e = self._trim_span(text, cur_start, cur_end)
                    if s < e:
                        entities.append((s, e, cur_label))
                cur_label = label[2:]
                cur_start = int(offset[0])
                cur_end = int(offset[1])
            elif (
                label.startswith("I-")
                and cur_label is not None
                and cur_label == label[2:]
            ):
                cur_end = int(offset[1])
            else:
                if cur_label is not None:
                    s, e = self._trim_span(text, cur_start, cur_end)
                    if s < e:
                        entities.append((s, e, cur_label))
                    cur_label = None
        if cur_label is not None:
            s, e = self._trim_span(text, cur_start, cur_end)
            if s < e:
                entities.append((s, e, cur_label))
        return entities


# ---------------------------------------------------------------------------
# Metrics (same logic as tests/e2e/metrics.py but self-contained)
# ---------------------------------------------------------------------------

Span = tuple[int, int, str]


class SpanMetrics:
    def __init__(self) -> None:
        self.tp: dict[str, int] = defaultdict(int)
        self.fp: dict[str, int] = defaultdict(int)
        self.fn: dict[str, int] = defaultdict(int)

    def update(self, gold: list[Span], predicted: list[Span]) -> None:
        gold_set = set(gold)
        pred_set = set(predicted)
        for _, _, label in gold_set & pred_set:
            self.tp[label] += 1
        for _, _, label in gold_set - pred_set:
            self.fn[label] += 1
        for _, _, label in pred_set - gold_set:
            self.fp[label] += 1

    def per_label(self) -> dict[str, dict[str, float | int]]:
        labels = set(self.tp) | set(self.fp) | set(self.fn)
        out: dict[str, dict[str, float | int]] = {}
        for label in sorted(labels):
            tp, fp, fn = self.tp[label], self.fp[label], self.fn[label]
            prec = tp / (tp + fp) if (tp + fp) else 0.0
            rec = tp / (tp + fn) if (tp + fn) else 0.0
            f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
            out[label] = {
                "tp": tp, "fp": fp, "fn": fn,
                "precision": round(prec, 4),
                "recall": round(rec, 4),
                "f1": round(f1, 4),
            }
        return out

    def micro_f1(self) -> float:
        tp = sum(self.tp.values())
        fp = sum(self.fp.values())
        fn = sum(self.fn.values())
        prec = tp / (tp + fp) if (tp + fp) else 0.0
        rec = tp / (tp + fn) if (tp + fn) else 0.0
        return 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0

    def macro_f1(self) -> float:
        per = self.per_label()
        if not per:
            return 0.0
        return sum(m["f1"] for m in per.values()) / len(per)


# ---------------------------------------------------------------------------
# Dataset loading and conversion
# ---------------------------------------------------------------------------

def load_ai4privacy_samples(
    num: int,
    seed: int,
    language: str | None = None,
) -> list[dict]:
    """Load samples from ai4privacy/pii-masking-300k and convert to our format.

    Returns list of {"text": ..., "entities": [(start, end, label), ...]}
    where labels are mapped to kiji label names, and unmappable entities
    are dropped.
    """
    ds = load_dataset(
        "ai4privacy/pii-masking-300k",
        split="train",
        streaming=True,
    )
    ds = ds.shuffle(seed=seed, buffer_size=1000)

    samples: list[dict] = []
    for row in ds:
        if language is not None and row.get("language", "").lower() != language.lower():
            continue
        entities: list[Span] = []
        for ent in row["privacy_mask"]:
            kiji_label = AI4PRIVACY_TO_KIJI.get(ent["label"])
            if kiji_label is None:
                continue
            entities.append((ent["start"], ent["end"], kiji_label))
        if not entities:
            continue
        samples.append({
            "text": row["source_text"],
            "entities": entities,
        })
        if len(samples) >= num:
            break
    return samples


# ---------------------------------------------------------------------------
# Verbose per-sample output
# ---------------------------------------------------------------------------

def print_sample_detections(
    index: int,
    text: str,
    gold: list[Span],
    predicted: list[Span],
    elapsed_ms: float,
) -> None:
    """Print gold vs predicted entities for a single sample."""
    print(f"\n{'─'*70}")
    print(f"Sample {index + 1}  ({elapsed_ms:.1f} ms)")
    print(f"{'─'*70}")
    print(f"  Text:")
    for line in text.splitlines():
        print(f"    {line}")

    gold_set = set(gold)
    pred_set = set(predicted)

    tp = gold_set & pred_set
    fn = gold_set - pred_set
    fp = pred_set - gold_set

    print(f"  Expected ({len(gold_set)}):")
    for start, end, label in sorted(gold_set):
        marker = "TP" if (start, end, label) in tp else "FN"
        print(f"    [{label:<20s}] {text[start:end]!r}  ({marker})")
    print(f"  Predicted ({len(pred_set)}):")
    for start, end, label in sorted(pred_set):
        marker = "TP" if (start, end, label) in tp else "FP"
        print(f"    [{label:<20s}] {text[start:end]!r}  ({marker})")
    if not gold_set and not pred_set:
        print("  (no entities)")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description="Benchmark kiji PII model against ai4privacy/pii-masking-300k"
    )
    ap.add_argument(
        "--num", type=int, default=1000,
        help="Number of samples to evaluate (default: 1000).",
    )
    ap.add_argument(
        "--model-path", default="./model/quantized",
        help="Path to the quantized ONNX model directory.",
    )
    ap.add_argument(
        "--report", default=str(Path(__file__).parent / "reports" / "latest.json"),
        help="Path to write the JSON report.",
    )
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument(
        "--language", default=None,
        help="Filter samples by language (e.g. 'English', 'German'). Default: all languages.",
    )
    ap.add_argument(
        "--verbose", action="store_true",
        help="Print per-sample detections (only allowed when --num < 50).",
    )
    args = ap.parse_args()
    if args.verbose and args.num >= 50:
        ap.error("--verbose is only allowed with --num < 50")
    return args


def main() -> int:
    args = parse_args()

    print(f"Loading ONNX model from {args.model_path} ...")
    model = OnnxPIIModel(args.model_path)

    lang_desc = f" (language={args.language})" if args.language else ""
    print(f"Loading {args.num} samples from ai4privacy/pii-masking-300k{lang_desc} ...")
    samples = load_ai4privacy_samples(args.num, seed=args.seed, language=args.language)
    print(f"  Loaded {len(samples)} samples with mappable entities.")

    if not samples:
        print("No samples loaded.", file=sys.stderr)
        return 1

    metrics = SpanMetrics()
    latencies: list[float] = []

    for i, sample in enumerate(samples):
        t0 = time.perf_counter()
        predicted = model.predict(sample["text"])
        elapsed_ms = (time.perf_counter() - t0) * 1000
        latencies.append(elapsed_ms)
        metrics.update(sample["entities"], predicted)
        if args.verbose:
            print_sample_detections(
                i, sample["text"], sample["entities"], predicted, elapsed_ms,
            )
        elif (i + 1) % 200 == 0:
            print(f"  Processed {i + 1}/{len(samples)} ...")

    # Build report
    lat_arr = np.asarray(latencies, dtype=float)
    per_label = metrics.per_label()
    report = {
        "dataset": "ai4privacy/pii-masking-300k",
        "num_samples": len(samples),
        "seed": args.seed,
        "model_path": args.model_path,
        "micro_f1": round(metrics.micro_f1(), 4),
        "macro_f1": round(metrics.macro_f1(), 4),
        "latency_ms": {
            "p50": float(round(np.percentile(lat_arr, 50), 2)),
            "p95": float(round(np.percentile(lat_arr, 95), 2)),
            "p99": float(round(np.percentile(lat_arr, 99), 2)),
            "mean": float(round(np.mean(lat_arr), 2)),
        },
        "per_label": per_label,
    }

    report_path = Path(args.report)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with report_path.open("w") as f:
        json.dump(report, f, indent=2)

    # Print summary
    print(f"\n{'='*60}")
    print(f"Benchmark Results  ({len(samples)} samples)")
    print(f"{'='*60}")
    print(f"  micro-F1 : {report['micro_f1']}")
    print(f"  macro-F1 : {report['macro_f1']}")
    print(f"  latency p50: {report['latency_ms']['p50']} ms")
    print(f"  latency p95: {report['latency_ms']['p95']} ms")
    print(f"  latency p99: {report['latency_ms']['p99']} ms")
    print(f"\nPer-label breakdown:")
    print(f"  {'Label':<22s} {'Prec':>6s} {'Rec':>6s} {'F1':>6s}  {'TP':>5s} {'FP':>5s} {'FN':>5s}")
    print(f"  {'-'*62}")
    for label, m in sorted(per_label.items()):
        print(
            f"  {label:<22s} {m['precision']:6.3f} {m['recall']:6.3f} {m['f1']:6.3f}"
            f"  {m['tp']:5d} {m['fp']:5d} {m['fn']:5d}"
        )
    print(f"\nReport saved to {report_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
