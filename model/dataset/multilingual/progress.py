"""Thread-safe progress tracking for a pipeline run (CLI and UI)."""

import json
import threading
import time
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any

OUTCOMES = ("created", "skipped", "failed", "cancelled")


class ProgressTracker:
    """Collects per-sample outcomes and exposes a JSON snapshot.

    ``on_update`` is called after every change with the snapshot; the UI uses
    it to persist ``progress.json``.
    """

    def __init__(self, on_update=None):
        self._lock = threading.Lock()
        self._on_update = on_update
        self._cancel = threading.Event()
        self.status = "pending"
        self.message = ""
        self.total = 0
        self.counts: Counter = Counter()
        self.per_language: dict[str, Counter] = {}
        self.per_text_type: dict[str, Counter] = {}
        self.errors: list[dict[str, Any]] = []
        self.started_at: float | None = None
        self.finished_at: float | None = None
        self.output_dir: str | None = None
        self.tasks_file: str | None = None
        self.summary: dict[str, Any] | None = None

    # -------------------------------------------------------------- control
    def request_cancel(self) -> None:
        self._cancel.set()
        with self._lock:
            if self.status == "running":
                self.status = "cancelling"
        self._notify()

    @property
    def cancelled(self) -> bool:
        return self._cancel.is_set()

    # ------------------------------------------------------------- lifecycle
    def start(self, total: int, output_dir: str) -> None:
        with self._lock:
            self.status = "running"
            self.total = total
            self.output_dir = output_dir
            self.started_at = time.time()
        self._notify()

    def record(
        self,
        outcome: str,
        language: str,
        text_type: str,
        error: str | None = None,
        index: int | None = None,
    ) -> None:
        if outcome not in OUTCOMES:
            raise ValueError(f"Unknown outcome {outcome!r}")
        with self._lock:
            self.counts[outcome] += 1
            self.per_language.setdefault(language, Counter())[outcome] += 1
            self.per_text_type.setdefault(text_type, Counter())[outcome] += 1
            if error:
                self.errors.append(
                    {"index": index, "language": language, "error": error}
                )
        self._notify()

    def set_message(self, message: str) -> None:
        with self._lock:
            self.message = message
        self._notify()

    def finish(self, summary: dict[str, Any], tasks_file: str) -> None:
        with self._lock:
            self.summary = summary
            self.tasks_file = tasks_file
            self.finished_at = time.time()
            self.status = "cancelled" if self.cancelled else "completed"
        self._notify()

    def fail(self, message: str) -> None:
        with self._lock:
            self.status = "failed"
            self.message = message
            self.finished_at = time.time()
        self._notify()

    # ---------------------------------------------------------------- views
    @property
    def done(self) -> int:
        return sum(self.counts.values())

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            done = sum(self.counts.values())
            now = self.finished_at or time.time()
            elapsed = (now - self.started_at) if self.started_at else 0.0
            rate = done / elapsed if elapsed > 0 and done else None
            remaining = self.total - done
            eta = remaining / rate if rate and remaining > 0 else None
            return {
                "status": self.status,
                "message": self.message,
                "total": self.total,
                "done": done,
                "percent": round(100 * done / self.total, 1) if self.total else 0.0,
                "created": self.counts["created"],
                "skipped": self.counts["skipped"],
                "failed": self.counts["failed"],
                "cancelled": self.counts["cancelled"],
                "per_language": {k: dict(v) for k, v in self.per_language.items()},
                "per_text_type": {k: dict(v) for k, v in self.per_text_type.items()},
                "errors": list(self.errors[-50:]),
                "started_at": _iso(self.started_at),
                "finished_at": _iso(self.finished_at),
                "elapsed_seconds": round(elapsed, 1),
                "eta_seconds": round(eta, 1) if eta is not None else None,
                "output_dir": self.output_dir,
                "tasks_file": self.tasks_file,
                "summary": self.summary,
            }

    def write(self, path: str | Path) -> None:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(self.snapshot(), indent=2, ensure_ascii=False))

    def _notify(self) -> None:
        if self._on_update is not None:
            self._on_update(self.snapshot())


def _iso(timestamp: float | None) -> str | None:
    return (
        datetime.fromtimestamp(timestamp).isoformat(timespec="seconds")
        if timestamp
        else None
    )
