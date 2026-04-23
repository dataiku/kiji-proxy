"""Proxy round-trip harness — sends samples through /v1/chat/completions
against a real upstream LLM provider. Requires OPENAI_API_KEY in the env.

Verifies:
  - request_success_rate: fraction of requests returning 2xx
  - mask_leak_rate: fraction of responses that still contain a mask value
    produced by the proxy (i.e., restoration missed one). Mask values are
    learned per-sample from a prior /api/pii/check call.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass

from .client import ThrottledClient
from .metrics import ErrorCounts, LatencyStats


@dataclass
class ProxyResult:
    latency: LatencyStats
    errors: ErrorCounts
    mask_leaks: int
    total: int
    provider: str
    model: str


async def run_proxy(
    client: ThrottledClient,
    samples: list[dict],
    *,
    model: str,
    provider: str = "openai",
) -> ProxyResult:
    latency = LatencyStats()
    errors = ErrorCounts()
    mask_leaks = 0
    total = 0

    async def one(sample: dict) -> None:
        nonlocal mask_leaks, total

        # Pre-fetch the mask values for this text so we can check for leakage
        # in the final response body.
        check_r, _ = await client.post_json(
            "/api/pii/check",
            {"message": sample["text"]},
        )
        masked_values: set[str] = set()
        if check_r is not None and check_r.status_code == 200:
            entities = check_r.json().get("entities", {}) or {}
            masked_values = {k for k in entities if k}

        body = {
            "model": model,
            "messages": [{"role": "user", "content": sample["text"]}],
        }
        response, elapsed = await client.post_json("/v1/chat/completions", body)
        latency.add(elapsed)
        total += 1
        if response is None:
            errors.add(None)
            return
        errors.add(response.status_code)
        if response.status_code != 200:
            return

        try:
            content = response.json()["choices"][0]["message"]["content"] or ""
        except (KeyError, IndexError, ValueError, TypeError):
            return
        if any(mv in content for mv in masked_values):
            mask_leaks += 1

    await asyncio.gather(*(one(s) for s in samples))
    return ProxyResult(
        latency=latency,
        errors=errors,
        mask_leaks=mask_leaks,
        total=total,
        provider=provider,
        model=model,
    )
