"""Run the kiji benchmark across multiple quantized variants and aggregate results.

For each subdirectory under ``--variants-dir`` containing ``model.onnx``, this
script invokes ``python -m tests.benchmark.run`` in a subprocess and collects
the resulting F1 + latency metrics alongside the on-disk model size. The
combined report is written to ``--report`` and printed as a table sorted by
exact-span F1 descending.

Usage:
    uv run python -m tests.benchmark.sweep_quants --num 1000 \\
        --variants-dir ./model/quant_variants
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--num", type=int, default=1000)
    p.add_argument("--variants-dir", default="./model/quant_variants")
    p.add_argument(
        "--report",
        default=str(Path("tests/benchmark/reports/quant_sweep.json")),
    )
    p.add_argument("--seed", type=int, default=42)
    p.add_argument(
        "--language",
        default=None,
        help="Optional language filter (passed through to benchmark).",
    )
    p.add_argument(
        "--only",
        default=None,
        help="Comma-separated list of variant names to run (default: all found).",
    )
    return p.parse_args()


def run_variant(variant_dir: Path, num: int, seed: int, language: str | None) -> dict:
    report_path = variant_dir / "benchmark_report.json"
    cmd = [
        sys.executable,
        "-m",
        "tests.benchmark.run",
        "--num",
        str(num),
        "--model-path",
        str(variant_dir),
        "--report",
        str(report_path),
        "--seed",
        str(seed),
    ]
    if language:
        cmd.extend(["--language", language])

    print(f"\n{'=' * 60}")
    print(f"Running benchmark for variant: {variant_dir.name}")
    print(f"  cmd: {' '.join(cmd)}")
    print(f"{'=' * 60}")

    result = subprocess.run(cmd, check=False)
    if result.returncode != 0:
        return {
            "variant": variant_dir.name,
            "status": "failed",
            "returncode": result.returncode,
        }
    if not report_path.exists():
        return {
            "variant": variant_dir.name,
            "status": "failed",
            "error": "no report produced",
        }

    with report_path.open() as f:
        report = json.load(f)

    size_mb = (variant_dir / "model.onnx").stat().st_size / (1024 * 1024)
    exact_f1 = report.get("exact_span_f1", 0.0)
    return {
        "variant": variant_dir.name,
        "status": "ok",
        "size_mb": round(size_mb, 1),
        "exact_f1": exact_f1,
        "exact_macro_f1": report.get("exact_span_macro_f1", 0.0),
        "relaxed_f1": report.get("relaxed_overlap_f1", 0.0),
        "relaxed_macro_f1": report.get("relaxed_overlap_macro_f1", 0.0),
        "latency_ms": report.get("latency_ms", {}),
        "f1_per_mb": round(exact_f1 / size_mb, 6) if size_mb > 0 else 0.0,
        "num_samples": report.get("num_samples", 0),
        "report_path": str(report_path),
    }


def print_table(rows: list[dict]) -> None:
    print("\n" + "=" * 100)
    print("QUANTIZATION SWEEP — sorted by exact F1 descending")
    print("=" * 100)
    header = (
        f"{'variant':<24s} {'size_MB':>9s} {'exact_F1':>9s} {'macro_F1':>9s} "
        f"{'relaxed':>8s} {'p50_ms':>8s} {'p95_ms':>8s} {'F1/MB':>9s}"
    )
    print(header)
    print("-" * 100)
    ok_rows = [r for r in rows if r.get("status") == "ok"]
    ok_rows.sort(key=lambda r: r.get("exact_f1", 0.0), reverse=True)
    for r in ok_rows:
        lat = r.get("latency_ms", {})
        print(
            f"{r['variant']:<24s} {r['size_mb']:>9.1f} {r['exact_f1']:>9.4f} "
            f"{r['exact_macro_f1']:>9.4f} {r['relaxed_f1']:>8.4f} "
            f"{lat.get('p50', 0):>8.1f} {lat.get('p95', 0):>8.1f} "
            f"{r['f1_per_mb']:>9.6f}"
        )
    failed = [r for r in rows if r.get("status") != "ok"]
    if failed:
        print("\nFailed variants:")
        for r in failed:
            print(f"  - {r['variant']}: {r.get('error') or r.get('returncode')}")


def main() -> int:
    args = parse_args()
    variants_dir = Path(args.variants_dir).resolve()

    if not variants_dir.exists():
        print(f"ERROR: variants dir not found: {variants_dir}", file=sys.stderr)
        return 1

    candidates = sorted(
        d for d in variants_dir.iterdir() if d.is_dir() and (d / "model.onnx").exists()
    )
    if args.only:
        wanted = {v.strip() for v in args.only.split(",") if v.strip()}
        candidates = [d for d in candidates if d.name in wanted]

    if not candidates:
        print(
            f"No variant directories with model.onnx found under {variants_dir}",
            file=sys.stderr,
        )
        return 1

    print(f"Found {len(candidates)} variant(s): {[d.name for d in candidates]}")

    rows = []
    for variant_dir in candidates:
        rows.append(run_variant(variant_dir, args.num, args.seed, args.language))

    report_path = Path(args.report).resolve()
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with report_path.open("w") as f:
        json.dump(
            {
                "num_samples_requested": args.num,
                "seed": args.seed,
                "language": args.language,
                "variants": rows,
            },
            f,
            indent=2,
        )

    print_table(rows)
    print(f"\nFull report written to: {report_path}")
    return 0 if all(r.get("status") == "ok" for r in rows) else 2


if __name__ == "__main__":
    sys.exit(main())
