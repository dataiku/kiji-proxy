"""Detection accuracy harness — hits /api/pii/check for every sample."""

from __future__ import annotations

import asyncio

from .client import ThrottledClient
from .metrics import ErrorCounts, LatencyStats, SpanMetrics


async def run_detection(
    client: ThrottledClient,
    samples: list[dict],
) -> tuple[SpanMetrics, LatencyStats, ErrorCounts]:
    metrics = SpanMetrics()
    latency = LatencyStats()
    errors = ErrorCounts()

    async def one(sample: dict) -> None:
        response, elapsed = await client.post_json(
            "/api/pii/check",
            {"message": sample["text"]},
        )
        latency.add(elapsed)
        if response is None:
            errors.add(None)
            return
        errors.add(response.status_code)
        if response.status_code != 200:
            return
        data = response.json()
        gold: list[tuple[int, int, str]] = [
            (e["start"], e["end"], e["label"]) for e in sample["entities"]
        ]
        predicted: list[tuple[int, int, str]] = [
            (e["start"], e["end"], e["label"])
            for e in data.get("detected_entities", [])
        ]
        metrics.update(gold, predicted)

    await asyncio.gather(*(one(s) for s in samples))
    return metrics, latency, errors
