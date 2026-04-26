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

from model.dataset.huggingface.import_ai4privacy import convert_ai4privacy_sample

# ---------------------------------------------------------------------------
# ONNX model wrapper
# ---------------------------------------------------------------------------

DEFAULT_ENTITY_CONFIDENCE_THRESHOLD = 0.25


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


def softmax_confidence(token_logits: np.ndarray, class_idx: int) -> float:
    """Return the softmax confidence for one decoded class."""
    max_logit = float(np.max(token_logits))
    exp_scores = np.exp(token_logits - max_logit)
    return float(exp_scores[class_idx] / np.sum(exp_scores))


def split_bio_label(label: str) -> tuple[str, bool, bool]:
    """Split a BIO label into base label and B/I flags."""
    is_beginning = label.startswith("B-")
    is_inside = label.startswith("I-")
    if is_beginning or is_inside:
        return label[2:], is_beginning, is_inside
    return label, False, False


def is_entity_joiner_token(token_text: str) -> bool:
    """Mirror backend compact-entity joiner detection."""
    trimmed = token_text.strip()
    return bool(trimmed) and all(c in ".,@_-+:/#%&=" for c in trimmed)


def token_text_at(text: str, offset: np.ndarray) -> str:
    """Return token text from an offset mapping, with bounds checks."""
    start = int(offset[0])
    end = int(offset[1])
    if start < 0 or end > len(text) or end <= start:
        return ""
    return text[start:end]


def token_starts_at_previous_end(
    token_index: int,
    current_tokens: list[int],
    offsets: np.ndarray,
) -> bool:
    """Check whether the current token is contiguous with the current entity."""
    if not current_tokens or token_index >= len(offsets):
        return False
    previous_token_index = current_tokens[-1]
    if previous_token_index >= len(offsets):
        return False
    return int(offsets[previous_token_index][1]) == int(offsets[token_index][0])


class OnnxPIIModel:
    """Thin wrapper around the quantized ONNX model for inference."""

    def __init__(
        self,
        model_dir: str,
        entity_confidence_threshold: float = DEFAULT_ENTITY_CONFIDENCE_THRESHOLD,
    ):
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
        self.entity_confidence_threshold = entity_confidence_threshold

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
            self.start_transitions = np.array(
                crf["start_transitions"], dtype=np.float32
            )
            self.end_transitions = np.array(crf["end_transitions"], dtype=np.float32)
        else:
            self.transitions = None
            self.start_transitions = None
            self.end_transitions = None

    def _decoded_label(
        self,
        index: int,
        predictions: list[int] | np.ndarray,
        logits: np.ndarray,
    ) -> tuple[str, float]:
        """Return decoded label and softmax confidence for one token."""
        if index >= len(predictions):
            return "O", 0.0
        class_id = int(predictions[index])
        if class_id < 0 or class_id >= logits.shape[-1]:
            return "O", 0.0
        label = self.id2label.get(class_id, "O")
        return label, softmax_confidence(logits[index], class_id)

    def _bridge_joiner_token(
        self,
        index: int,
        token_text: str,
        label: str,
        confidence: float,
        predictions: list[int] | np.ndarray,
        logits: np.ndarray,
        current_label: str | None,
    ) -> tuple[str, float]:
        """Mirror backend joiner bridging for compact entities like emails."""
        if (
            label != "O"
            or current_label is None
            or not is_entity_joiner_token(token_text)
        ):
            return label, confidence

        next_label, next_confidence = self._decoded_label(
            index + 1, predictions, logits
        )
        next_base_label, next_is_beginning, next_is_inside = split_bio_label(next_label)
        if (
            next_confidence >= self.entity_confidence_threshold
            and (next_is_beginning or next_is_inside)
            and next_base_label == current_label
        ):
            return f"I-{current_label}", confidence

        return label, confidence

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
        seq_logits = logits[0]  # (seq_len, num_labels)

        if self.transitions is not None:
            # Use Viterbi decoding with CRF transition constraints
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
        cur_tokens: list[int] = []

        def finish_current() -> None:
            nonlocal cur_label, cur_start, cur_end, cur_tokens
            if cur_label is None:
                return
            s, e = self._trim_span(text, cur_start, cur_end)
            if s < e:
                entities.append((s, e, cur_label))
            cur_label = None
            cur_tokens = []

        for index, (token, offset) in enumerate(
            zip(tokens, offset_mapping, strict=True)
        ):
            if token in (
                self.tokenizer.cls_token,
                self.tokenizer.sep_token,
                self.tokenizer.pad_token,
            ):
                continue

            label, confidence = self._decoded_label(index, predictions, seq_logits)
            token_text = token_text_at(text, offset)
            if confidence < self.entity_confidence_threshold:
                label = "O"
            label, confidence = self._bridge_joiner_token(
                index,
                token_text,
                label,
                confidence,
                predictions,
                seq_logits,
                cur_label,
            )

            base_label, is_beginning, is_inside = split_bio_label(label)
            is_same_compact_entity = (
                label != "O"
                and cur_label is not None
                and cur_label == base_label
                and token_starts_at_previous_end(index, cur_tokens, offset_mapping)
            )

            if (
                label != "O"
                and cur_label is not None
                and cur_label == base_label
                and (is_inside or is_same_compact_entity)
            ):
                cur_end = int(offset[1])
                cur_tokens.append(index)
            elif label != "O" and (is_beginning or cur_label is None):
                finish_current()
                cur_label = base_label
                cur_start = int(offset[0])
                cur_end = int(offset[1])
                cur_tokens = [index]
            else:
                finish_current()

        finish_current()
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
                "tp": tp,
                "fp": fp,
                "fn": fn,
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


class RelaxedOverlapMetrics(SpanMetrics):
    """Same-label one-to-one span-overlap scoring."""

    @staticmethod
    def _overlap_len(left: Span, right: Span) -> int:
        return max(0, min(left[1], right[1]) - max(left[0], right[0]))

    def update(self, gold: list[Span], predicted: list[Span]) -> None:
        gold_spans = sorted(set(gold))
        pred_spans = sorted(set(predicted))
        candidates: list[tuple[int, int, int]] = []
        for gold_idx, gold_span in enumerate(gold_spans):
            for pred_idx, pred_span in enumerate(pred_spans):
                if gold_span[2] != pred_span[2]:
                    continue
                overlap_len = self._overlap_len(gold_span, pred_span)
                if overlap_len > 0:
                    candidates.append((overlap_len, gold_idx, pred_idx))

        matched_gold: set[int] = set()
        matched_pred: set[int] = set()
        for _overlap_len, gold_idx, pred_idx in sorted(candidates, reverse=True):
            if gold_idx in matched_gold or pred_idx in matched_pred:
                continue
            matched_gold.add(gold_idx)
            matched_pred.add(pred_idx)
            self.tp[gold_spans[gold_idx][2]] += 1

        for idx, span in enumerate(gold_spans):
            if idx not in matched_gold:
                self.fn[span[2]] += 1
        for idx, span in enumerate(pred_spans):
            if idx not in matched_pred:
                self.fp[span[2]] += 1


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
    if language is not None:
        normalized_language = language.lower()
        ds = ds.filter(
            lambda row: row.get("language", "").lower() == normalized_language
        )
    ds = ds.shuffle(seed=seed, buffer_size=1000)

    samples: list[dict] = []
    for row in ds:
        sample = convert_ai4privacy_sample(row)
        if sample is None:
            continue

        entities: list[Span] = []
        for ent in sample["privacy_mask"]:
            entities.append((ent["start"], ent["end"], ent["label"]))
        if not entities:
            continue
        samples.append(
            {
                "text": sample["text"],
                "entities": entities,
            }
        )
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
    print(f"\n{'─' * 70}")
    print(f"Sample {index + 1}  ({elapsed_ms:.1f} ms)")
    print(f"{'─' * 70}")
    print("  Text:")
    for line in text.splitlines():
        print(f"    {line}")

    gold_set = set(gold)
    pred_set = set(predicted)

    tp = gold_set & pred_set

    print(f"  Expected ({len(gold_set)}):")
    if gold_set:
        for start, end, label in sorted(gold_set):
            marker = "TP" if (start, end, label) in tp else "FN"
            print(f"    [{label:<20s}] {text[start:end]!r}  ({marker})")
    else:
        print("    (none)")
    print(f"  Predicted ({len(pred_set)}):")
    if pred_set:
        for start, end, label in sorted(pred_set):
            marker = "TP" if (start, end, label) in tp else "FP"
            print(f"    [{label:<20s}] {text[start:end]!r}  ({marker})")
    else:
        print("    (none)")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description="Benchmark kiji PII model against ai4privacy/pii-masking-300k"
    )
    ap.add_argument(
        "--num",
        type=int,
        default=1000,
        help="Number of samples to evaluate (default: 1000).",
    )
    ap.add_argument(
        "--model-path",
        default="./model/quantized",
        help="Path to the quantized ONNX model directory.",
    )
    ap.add_argument(
        "--report",
        default=str(Path(__file__).parent / "reports" / "latest.json"),
        help="Path to write the JSON report.",
    )
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument(
        "--confidence-threshold",
        type=float,
        default=DEFAULT_ENTITY_CONFIDENCE_THRESHOLD,
        help="Minimum token confidence before treating a label as O. Mirrors backend default.",
    )
    ap.add_argument(
        "--language",
        default=None,
        help="Filter samples by language (e.g. 'English', 'German'). Default: all languages.",
    )
    ap.add_argument(
        "--verbose",
        action="store_true",
        help="Print per-sample detections (only allowed when --num < 50).",
    )
    args = ap.parse_args()
    if args.verbose and args.num >= 50:
        ap.error("--verbose is only allowed with --num < 50")
    return args


def main() -> int:
    args = parse_args()

    print(f"Loading ONNX model from {args.model_path} ...")
    model = OnnxPIIModel(
        args.model_path,
        entity_confidence_threshold=args.confidence_threshold,
    )

    lang_desc = f" (language={args.language})" if args.language else ""
    print(f"Loading {args.num} samples from ai4privacy/pii-masking-300k{lang_desc} ...")
    samples = load_ai4privacy_samples(args.num, seed=args.seed, language=args.language)
    print(f"  Loaded {len(samples)} samples with mappable entities.")

    if not samples:
        print("No samples loaded.", file=sys.stderr)
        return 1

    exact_metrics = SpanMetrics()
    relaxed_metrics = RelaxedOverlapMetrics()
    latencies: list[float] = []

    for i, sample in enumerate(samples):
        t0 = time.perf_counter()
        predicted = model.predict(sample["text"])
        elapsed_ms = (time.perf_counter() - t0) * 1000
        latencies.append(elapsed_ms)
        exact_metrics.update(sample["entities"], predicted)
        relaxed_metrics.update(sample["entities"], predicted)
        if args.verbose:
            print_sample_detections(
                i,
                sample["text"],
                sample["entities"],
                predicted,
                elapsed_ms,
            )
        elif (i + 1) % 200 == 0:
            print(f"  Processed {i + 1}/{len(samples)} ...")

    # Build report
    lat_arr = np.asarray(latencies, dtype=float)
    exact_per_label = exact_metrics.per_label()
    relaxed_per_label = relaxed_metrics.per_label()
    exact_micro_f1 = exact_metrics.micro_f1()
    exact_macro_f1 = exact_metrics.macro_f1()
    relaxed_micro_f1 = relaxed_metrics.micro_f1()
    relaxed_macro_f1 = relaxed_metrics.macro_f1()
    report = {
        "dataset": "ai4privacy/pii-masking-300k",
        "num_samples": len(samples),
        "seed": args.seed,
        "language": args.language,
        "model_path": args.model_path,
        "confidence_threshold": args.confidence_threshold,
        "exact_span_f1": round(exact_micro_f1, 4),
        "exact_span_macro_f1": round(exact_macro_f1, 4),
        "relaxed_overlap_f1": round(relaxed_micro_f1, 4),
        "relaxed_overlap_macro_f1": round(relaxed_macro_f1, 4),
        "micro_f1": round(exact_micro_f1, 4),
        "macro_f1": round(exact_macro_f1, 4),
        "latency_ms": {
            "p50": float(round(np.percentile(lat_arr, 50), 2)),
            "p95": float(round(np.percentile(lat_arr, 95), 2)),
            "p99": float(round(np.percentile(lat_arr, 99), 2)),
            "mean": float(round(np.mean(lat_arr), 2)),
        },
        "per_label": exact_per_label,
        "exact_span": {
            "micro_f1": round(exact_micro_f1, 4),
            "macro_f1": round(exact_macro_f1, 4),
            "per_label": exact_per_label,
        },
        "relaxed_same_label_overlap": {
            "micro_f1": round(relaxed_micro_f1, 4),
            "macro_f1": round(relaxed_macro_f1, 4),
            "per_label": relaxed_per_label,
        },
    }

    report_path = Path(args.report)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with report_path.open("w") as f:
        json.dump(report, f, indent=2)

    # Print summary
    print(f"\n{'=' * 60}")
    print(f"Benchmark Results  ({len(samples)} samples)")
    print(f"{'=' * 60}")
    print(f"  exact span micro-F1          : {report['exact_span_f1']}")
    print(f"  exact span macro-F1          : {report['exact_span_macro_f1']}")
    print(f"  relaxed overlap micro-F1    : {report['relaxed_overlap_f1']}")
    print(f"  relaxed overlap macro-F1    : {report['relaxed_overlap_macro_f1']}")
    print(f"  latency p50: {report['latency_ms']['p50']} ms")
    print(f"  latency p95: {report['latency_ms']['p95']} ms")
    print(f"  latency p99: {report['latency_ms']['p99']} ms")

    print("\nExact span per-label breakdown:")
    print(
        f"  {'Label':<22s} {'Prec':>6s} {'Rec':>6s} {'F1':>6s}  {'TP':>5s} {'FP':>5s} {'FN':>5s}"
    )
    print(f"  {'-' * 62}")
    for label, m in sorted(exact_per_label.items()):
        print(
            f"  {label:<22s} {m['precision']:6.3f} {m['recall']:6.3f} {m['f1']:6.3f}"
            f"  {m['tp']:5d} {m['fp']:5d} {m['fn']:5d}"
        )

    print("\nRelaxed same-label overlap per-label breakdown:")
    print(
        f"  {'Label':<22s} {'Prec':>6s} {'Rec':>6s} {'F1':>6s}  {'TP':>5s} {'FP':>5s} {'FN':>5s}"
    )
    print(f"  {'-' * 62}")
    for label, m in sorted(relaxed_per_label.items()):
        print(
            f"  {label:<22s} {m['precision']:6.3f} {m['recall']:6.3f} {m['f1']:6.3f}"
            f"  {m['tp']:5d} {m['fp']:5d} {m['fn']:5d}"
        )
    print(f"\nReport saved to {report_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
