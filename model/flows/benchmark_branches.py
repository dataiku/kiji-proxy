#!/usr/bin/env -S uv run --extra training --extra quantization --extra signing --script
# /// script
# requires-python = ">=3.13"
# dependencies = []
# ///
"""Benchmark training across git branches and compare metrics.

Trains the model on each specified branch using isolated git worktrees,
collects evaluation metrics from trainer_state.json, and writes a
comparison report to a JSON file.

Usage:
    # Compare feature branches against main
    uv run --extra training --extra quantization --extra signing \
        python model/flows/benchmark_branches.py

    # Specific branches only
    uv run --extra training --extra quantization --extra signing \
        python model/flows/benchmark_branches.py \
        --branches main fix/shuffle-before-train-val-split feat/class-weights-from-label-frequency

    # Quick test with subsampled data
    uv run --extra training --extra quantization --extra signing \
        python model/flows/benchmark_branches.py --subsample 500 --epochs 3

    # Custom output
    uv run --extra training --extra quantization --extra signing \
        python model/flows/benchmark_branches.py --output results/benchmark.json
"""

import argparse
import json
import os
import shutil
import subprocess
import tempfile
import time
from datetime import datetime
from pathlib import Path

METRICS_KEYS = [
    "eval_pii_f1_weighted",
    "eval_pii_f1_macro",
    "eval_pii_precision_weighted",
    "eval_pii_precision_macro",
    "eval_pii_recall_weighted",
    "eval_pii_recall_macro",
    "eval_coref_f1_weighted",
    "eval_coref_f1_macro",
    "eval_loss",
]


def get_repo_root() -> Path:
    result = subprocess.run(
        ["git", "rev-parse", "--show-toplevel"],
        capture_output=True,
        text=True,
        check=True,
    )
    return Path(result.stdout.strip())


def get_branch_commit(branch: str) -> str:
    result = subprocess.run(
        ["git", "rev-parse", "--short", branch],
        capture_output=True,
        text=True,
        check=True,
    )
    return result.stdout.strip()


def create_worktree(repo_root: Path, branch: str, base_dir: Path) -> Path:
    """Create an isolated git worktree for a branch."""
    worktree_path = base_dir / branch.replace("/", "_")
    subprocess.run(
        ["git", "worktree", "add", str(worktree_path), branch],
        cwd=repo_root,
        check=True,
        capture_output=True,
        text=True,
    )
    return worktree_path


def remove_worktree(repo_root: Path, worktree_path: Path):
    """Clean up a git worktree."""
    subprocess.run(
        ["git", "worktree", "remove", "--force", str(worktree_path)],
        cwd=repo_root,
        capture_output=True,
        text=True,
    )


def write_benchmark_config(
    worktree: Path,
    epochs: int,
    subsample: int,
    training_samples_dir: str,
) -> Path:
    """Write a training config TOML tailored for benchmarking."""
    output_dir = worktree / "model" / "benchmark_output"
    output_dir.mkdir(parents=True, exist_ok=True)

    config_path = worktree / "benchmark_config.toml"
    config_path.write_text(f"""\
[model]
name = "distilbert-base-cased"

[training]
num_epochs = {epochs}
batch_size = 4
learning_rate = 2e-5
warmup_steps = 500
weight_decay = 0.01
early_stopping_enabled = true
early_stopping_patience = 3
early_stopping_threshold = 0.005

[paths]
training_samples_dir = "{training_samples_dir}"
output_dir = "{output_dir}"

[data]
subsample_count = {subsample}

[pipeline]
skip_export = true
skip_quantization = true
skip_signing = true
""")
    return config_path


def run_training(worktree: Path, config_path: Path) -> dict:
    """Run the training pipeline in a worktree and return metrics."""
    env = os.environ.copy()
    env["WANDB_DISABLED"] = "true"
    env["WANDB_MODE"] = "disabled"

    cmd = [
        "uv",
        "run",
        "--extra",
        "training",
        "--extra",
        "quantization",
        "--extra",
        "signing",
        "python",
        "model/flows/training_pipeline.py",
        "--config",
        "config-file",
        str(config_path),
        "run",
    ]

    print(f"    Running: {' '.join(cmd[:6])}...")
    result = subprocess.run(
        cmd,
        cwd=worktree,
        env=env,
        capture_output=True,
        text=True,
    )

    if result.returncode != 0:
        print("    STDERR (last 40 lines):")
        for line in result.stderr.strip().splitlines()[-40:]:
            print(f"      {line}")
        return {"error": f"Training failed with exit code {result.returncode}"}

    return extract_metrics(worktree / "model" / "benchmark_output")


def extract_metrics(output_dir: Path) -> dict:
    """Extract final evaluation metrics from trainer_state.json."""
    trainer_state = output_dir / "trainer_state.json"
    if not trainer_state.exists():
        return {"error": f"trainer_state.json not found in {output_dir}"}

    state = json.loads(trainer_state.read_text())
    log_history = state.get("log_history", [])

    # Find the last evaluation entry (has eval_loss)
    eval_entries = [e for e in log_history if "eval_loss" in e]
    if not eval_entries:
        return {"error": "No evaluation entries found in trainer_state.json"}

    last_eval = eval_entries[-1]
    metrics = {}
    for key in METRICS_KEYS:
        if key in last_eval:
            metrics[key] = last_eval[key]

    metrics["epochs_completed"] = last_eval.get("epoch", None)
    return metrics


def compute_deltas(baseline: dict, branch_metrics: dict) -> dict:
    """Compute metric deltas relative to baseline."""
    deltas = {}
    for key in METRICS_KEYS:
        base_val = baseline.get(key)
        branch_val = branch_metrics.get(key)
        if base_val is not None and branch_val is not None:
            delta = branch_val - base_val
            # For loss, negative delta is better; for F1/precision/recall, positive is better
            deltas[f"{key}_delta"] = round(delta, 6)
            if base_val != 0:
                deltas[f"{key}_pct_change"] = round((delta / abs(base_val)) * 100, 2)
    return deltas


def print_comparison_table(results: dict):
    """Print a readable comparison table to stdout."""
    branches = list(results["branches"].keys())
    baseline_name = results["baseline_branch"]

    print("\n" + "=" * 90)
    print("BENCHMARK RESULTS")
    print("=" * 90)

    # Header
    header = f"{'Metric':<35}"
    for branch in branches:
        short = branch.split("/")[-1][:16]
        header += f"  {short:>16}"
    print(header)
    print("-" * 90)

    for key in METRICS_KEYS:
        row = f"{key:<35}"
        for branch in branches:
            bdata = results["branches"][branch]
            if "error" in bdata:
                row += f"  {'ERROR':>16}"
            elif key in bdata.get("metrics", {}):
                val = bdata["metrics"][key]
                row += f"  {val:>16.6f}"
            else:
                row += f"  {'N/A':>16}"
        print(row)

        # Delta row (skip for baseline)
        if len(branches) > 1:
            delta_row = (
                f"{'  (delta vs ' + baseline_name.split('/')[-1][:10] + ')':<35}"
            )
            for branch in branches:
                bdata = results["branches"][branch]
                delta_key = f"{key}_delta"
                if branch == baseline_name or "error" in bdata:
                    delta_row += f"  {'---':>16}"
                elif delta_key in bdata.get("deltas", {}):
                    d = bdata["deltas"][delta_key]
                    sign = "+" if d > 0 else ""
                    delta_row += f"  {sign}{d:>15.6f}"
                else:
                    delta_row += f"  {'N/A':>16}"
            print(delta_row)

    print("=" * 90)


def main():
    parser = argparse.ArgumentParser(
        description="Benchmark training across git branches"
    )
    parser.add_argument(
        "--branches",
        nargs="*",
        default=None,
        help="Branches to benchmark (default: main + all local feat/fix branches)",
    )
    parser.add_argument(
        "--baseline",
        default="main",
        help="Baseline branch to compare against (default: main)",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=5,
        help="Number of training epochs (default: 5)",
    )
    parser.add_argument(
        "--subsample",
        type=int,
        default=0,
        help="Limit training samples (0 = all, useful for quick tests)",
    )
    parser.add_argument(
        "--training-samples-dir",
        default=None,
        help="Path to training samples (default: use config default)",
    )
    parser.add_argument(
        "--output",
        default="model/benchmark_results.json",
        help="Path for output JSON report",
    )

    args = parser.parse_args()
    repo_root = get_repo_root()

    # Determine branches to benchmark
    if args.branches:
        branches = args.branches
    else:
        result = subprocess.run(
            ["git", "branch", "--format=%(refname:short)"],
            capture_output=True,
            text=True,
            check=True,
        )
        all_branches = result.stdout.strip().splitlines()
        branches = [args.baseline] + [
            b
            for b in all_branches
            if (b.startswith("feat/") or b.startswith("fix/")) and b != args.baseline
        ]

    # Ensure baseline is first
    if args.baseline in branches:
        branches.remove(args.baseline)
    branches.insert(0, args.baseline)

    # Resolve training samples dir to absolute path so worktrees can find it
    if args.training_samples_dir:
        training_samples_dir = str(Path(args.training_samples_dir).resolve())
    else:
        training_samples_dir = str(
            (
                repo_root / "model" / "dataset" / "data_samples" / "training_samples"
            ).resolve()
        )

    print(f"Benchmarking {len(branches)} branches:")
    for b in branches:
        marker = " (baseline)" if b == args.baseline else ""
        print(f"  - {b}{marker}")
    print(f"Epochs: {args.epochs}, Subsample: {args.subsample or 'all'}")
    print(f"Training data: {training_samples_dir}")
    print()

    results = {
        "generated_at": datetime.now().isoformat(),
        "baseline_branch": args.baseline,
        "config": {
            "epochs": args.epochs,
            "subsample": args.subsample,
            "training_samples_dir": training_samples_dir,
        },
        "branches": {},
    }

    worktree_base = Path(tempfile.mkdtemp(prefix="kiji_bench_"))

    try:
        for branch in branches:
            print(f"\n{'=' * 60}")
            print(f"  Branch: {branch}")
            print(f"{'=' * 60}")

            commit = get_branch_commit(branch)
            print(f"  Commit: {commit}")

            # Create worktree
            print("  Creating worktree...")
            worktree = create_worktree(repo_root, branch, worktree_base)

            # Write benchmark config
            config_path = write_benchmark_config(
                worktree, args.epochs, args.subsample, training_samples_dir
            )

            # Train
            start = time.time()
            metrics = run_training(worktree, config_path)
            elapsed = time.time() - start

            branch_result = {
                "commit": commit,
                "training_wall_time_seconds": round(elapsed, 1),
                "metrics": metrics if "error" not in metrics else {},
            }
            if "error" in metrics:
                branch_result["error"] = metrics["error"]

            results["branches"][branch] = branch_result

            # Clean up worktree
            print("  Cleaning up worktree...")
            remove_worktree(repo_root, worktree)

            if "error" not in metrics:
                f1 = metrics.get("eval_pii_f1_weighted", "N/A")
                print(f"  Result: eval_pii_f1_weighted = {f1}")
            else:
                print(f"  Result: FAILED - {metrics['error']}")

        # Compute deltas against baseline
        baseline_metrics = results["branches"].get(args.baseline, {}).get("metrics", {})
        if baseline_metrics:
            for branch, bdata in results["branches"].items():
                if branch != args.baseline and "error" not in bdata:
                    bdata["deltas"] = compute_deltas(baseline_metrics, bdata["metrics"])

        # Print comparison
        print_comparison_table(results)

        # Write output
        output_path = repo_root / args.output
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(results, indent=2))
        print(f"\nResults written to: {output_path}")

    finally:
        # Clean up remaining worktrees
        if worktree_base.exists():
            for child in worktree_base.iterdir():
                if child.is_dir():
                    remove_worktree(repo_root, child)
            shutil.rmtree(worktree_base, ignore_errors=True)
        # Prune stale worktree references
        subprocess.run(
            ["git", "worktree", "prune"],
            cwd=repo_root,
            capture_output=True,
        )


if __name__ == "__main__":
    main()
