"""Assemble and write the run report JSON."""

from __future__ import annotations

import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import httpx

from .metrics import ErrorCounts, LatencyStats, SpanMetrics
from .proxy import ProxyResult


def backend_version(base_url: str) -> str | None:
    try:
        r = httpx.get(f"{base_url}/version", timeout=5)
        if r.status_code == 200:
            return r.json().get("version")
    except httpx.HTTPError:
        return None
    return None


def git_commit() -> str | None:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            stderr=subprocess.DEVNULL,
        ).decode().strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return None


def write_report(
    path: str,
    base_url: str,
    dataset_info: dict,
    concurrency: int,
    detection: tuple[SpanMetrics, LatencyStats, ErrorCounts] | None,
    proxy: ProxyResult | None,
) -> dict:
    report: dict = {
        "backend_version": backend_version(base_url),
        "backend_commit": git_commit(),
        "run_timestamp": datetime.now(timezone.utc).isoformat(),
        "dataset": dataset_info,
        "concurrency": concurrency,
    }
    if detection is not None:
        metrics, latency, errors = detection
        report["detection"] = {
            "num_samples": dataset_info["num_samples"],
            "per_label": metrics.per_label(),
            "micro_f1": round(metrics.micro_f1(), 4),
            "macro_f1": round(metrics.macro_f1(), 4),
            "latency_ms": latency.percentiles(),
            "error_5xx_count": errors.five_xx,
            "status_breakdown": dict(errors.buckets),
            "timeouts": errors.timeouts,
        }
    if proxy is not None:
        total = max(proxy.total, 1)
        report["proxy"] = {
            "provider": proxy.provider,
            "model": proxy.model,
            "num_samples": proxy.total,
            "request_success_rate": round(proxy.errors.two_xx / total, 4),
            "mask_leak_rate": round(proxy.mask_leaks / total, 4),
            "latency_ms": proxy.latency.percentiles(),
            "error_5xx_count": proxy.errors.five_xx,
            "status_breakdown": dict(proxy.errors.buckets),
            "timeouts": proxy.errors.timeouts,
        }
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, sort_keys=False)
        f.write("\n")
    return report
