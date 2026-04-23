"""Span-based detection metrics and latency / error aggregates."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field

import numpy as np

Span = tuple[int, int, str]


@dataclass
class SpanMetrics:
    tp: dict[str, int] = field(default_factory=lambda: defaultdict(int))
    fp: dict[str, int] = field(default_factory=lambda: defaultdict(int))
    fn: dict[str, int] = field(default_factory=lambda: defaultdict(int))

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
            precision = tp / (tp + fp) if (tp + fp) else 0.0
            recall = tp / (tp + fn) if (tp + fn) else 0.0
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
            out[label] = {
                "tp": tp,
                "fp": fp,
                "fn": fn,
                "precision": round(precision, 4),
                "recall": round(recall, 4),
                "f1": round(f1, 4),
            }
        return out

    def micro_f1(self) -> float:
        tp = sum(self.tp.values())
        fp = sum(self.fp.values())
        fn = sum(self.fn.values())
        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        if precision + recall == 0:
            return 0.0
        return 2 * precision * recall / (precision + recall)

    def macro_f1(self) -> float:
        per = self.per_label()
        if not per:
            return 0.0
        return sum(m["f1"] for m in per.values()) / len(per)


@dataclass
class LatencyStats:
    samples: list[float] = field(default_factory=list)

    def add(self, ms: float) -> None:
        self.samples.append(ms)

    def percentiles(self) -> dict[str, float | None]:
        if not self.samples:
            return {"p50": None, "p95": None, "p99": None}
        arr = np.asarray(self.samples, dtype=float)
        return {
            "p50": float(round(np.percentile(arr, 50), 2)),
            "p95": float(round(np.percentile(arr, 95), 2)),
            "p99": float(round(np.percentile(arr, 99), 2)),
        }


@dataclass
class ErrorCounts:
    buckets: dict[str, int] = field(default_factory=lambda: defaultdict(int))
    timeouts: int = 0

    def add(self, status: int | None) -> None:
        if status is None:
            self.timeouts += 1
            return
        if 200 <= status < 300:
            self.buckets["2xx"] += 1
        elif 400 <= status < 500:
            self.buckets["4xx"] += 1
        elif 500 <= status < 600:
            self.buckets["5xx"] += 1
        else:
            self.buckets["other"] += 1

    @property
    def five_xx(self) -> int:
        return self.buckets.get("5xx", 0)

    @property
    def two_xx(self) -> int:
        return self.buckets.get("2xx", 0)
