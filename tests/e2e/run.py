"""Entry point for the e2e evaluation harness.

Usage:
    uv run python -m tests.e2e.run [--num N] [--proxy-samples M] [--skip-proxy] ...

Prerequisites:
    - Backend must be running at --backend-url (default http://127.0.0.1:8080).
      Start it with `make go-backend-dev` in a separate shell.
    - For proxy round-trip tests, export OPENAI_API_KEY (or pass --skip-proxy).
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
import time
from pathlib import Path

import httpx

from .client import ThrottledClient
from .detection import run_detection
from .proxy import run_proxy
from .reporter import write_report

HERE = Path(__file__).parent


async def wait_for_backend(base_url: str, timeout_s: float = 15.0) -> bool:
    deadline = time.monotonic() + timeout_s
    async with httpx.AsyncClient(timeout=5) as client:
        while time.monotonic() < deadline:
            try:
                r = await client.get(f"{base_url}/health")
                if r.status_code == 200:
                    return True
            except httpx.HTTPError:
                pass
            await asyncio.sleep(0.5)
    return False


def load_dataset(path: str, limit: int | None = None) -> list[dict]:
    samples: list[dict] = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            samples.append(json.loads(line))
            if limit is not None and len(samples) >= limit:
                break
    return samples


async def warm_up(base_url: str) -> None:
    async with httpx.AsyncClient(timeout=30) as client:
        try:
            await client.post(
                f"{base_url}/api/pii/check",
                json={"message": "warm up"},
            )
        except httpx.HTTPError:
            pass


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="kiji-proxy end-to-end evaluator")
    ap.add_argument("--num", type=int, default=750, help="Number of detection samples (default: 750).")
    ap.add_argument("--proxy-samples", type=int, default=100, help="Number of proxy round-trip samples (default: 100).")
    ap.add_argument("--skip-proxy", action="store_true", help="Skip the proxy round-trip phase.")
    ap.add_argument("--providers", default="openai", help="Comma-separated list; only 'openai' is currently supported.")
    ap.add_argument("--model", default="gpt-4o-mini", help="Upstream model name for proxy tests.")
    ap.add_argument("--report", default=str(HERE / "reports" / "latest.json"))
    ap.add_argument("--dataset", default=str(HERE / "dataset" / "samples.jsonl"))
    ap.add_argument("--concurrency", type=int, default=10)
    ap.add_argument("--backend-url", default="http://127.0.0.1:8080")
    ap.add_argument("--seed", type=int, default=42, help="Recorded in the report; dataset is pre-seeded.")
    return ap.parse_args()


async def amain() -> int:
    args = parse_args()

    if not await wait_for_backend(args.backend_url):
        print(
            f"Backend not responding at {args.backend_url}.\n"
            "Start it with `make go-backend-dev` in a separate shell.",
            file=sys.stderr,
        )
        return 2

    samples = load_dataset(args.dataset, limit=args.num)
    if not samples:
        print(
            f"No samples loaded from {args.dataset}.\n"
            "Generate the dataset with `make test-e2e-dataset`.",
            file=sys.stderr,
        )
        return 2

    await warm_up(args.backend_url)

    client = ThrottledClient(args.backend_url, concurrency=args.concurrency)
    detection_result = None
    proxy_result = None
    try:
        print(f"Running detection on {len(samples)} samples (concurrency={args.concurrency})...")
        detection_result = await run_detection(client, samples)

        if not args.skip_proxy and args.proxy_samples > 0:
            providers = [p.strip() for p in args.providers.split(",") if p.strip()]
            if providers != ["openai"]:
                print(
                    "Proxy round-trip currently only supports 'openai'; skipping.",
                    file=sys.stderr,
                )
            elif not os.environ.get("OPENAI_API_KEY"):
                print(
                    "OPENAI_API_KEY not set; skipping proxy round-trip.",
                    file=sys.stderr,
                )
            else:
                subset = samples[: args.proxy_samples]
                print(f"Running proxy round-trip on {len(subset)} samples via {args.model}...")
                proxy_result = await run_proxy(
                    client,
                    subset,
                    model=args.model,
                    provider="openai",
                )
    finally:
        await client.close()

    dataset_info = {
        "file": os.path.basename(args.dataset),
        "num_samples": len(samples),
        "seed": args.seed,
    }
    report = write_report(
        args.report,
        args.backend_url,
        dataset_info,
        args.concurrency,
        detection_result,
        proxy_result,
    )

    print(f"\nReport: {args.report}")
    det = report.get("detection") or {}
    if det:
        print(f"  detection micro-F1 : {det['micro_f1']}")
        print(f"  detection macro-F1 : {det['macro_f1']}")
        print(f"  detection p95 (ms) : {det['latency_ms']['p95']}")
        print(f"  detection 5xx count: {det['error_5xx_count']}")
    prx = report.get("proxy") or {}
    if prx:
        print(f"  proxy success rate : {prx['request_success_rate']:.1%}")
        print(f"  proxy mask leak    : {prx['mask_leak_rate']:.1%}")
        print(f"  proxy p95 (ms)     : {prx['latency_ms']['p95']}")
        print(f"  proxy 5xx count    : {prx['error_5xx_count']}")
    return 0


def main() -> None:
    sys.exit(asyncio.run(amain()))


if __name__ == "__main__":
    main()
