"""Train from a training TOML in one process, without Metaflow.

For TPU VMs. The Metaflow pipeline runs each step in its own process: on
Ctrl-C it gives the training step 5 seconds and then SIGKILLs it, and
``@retry`` restarts a failed training from scratch. Hard kills and quick
restarts are what leave a TPU in an unrecoverable state, so on a TPU run
training here instead and stop it only with Ctrl-C (SIGINT).

The TOML is read by ``TrainingConfig.from_toml``, like the pipeline. The
model, tokenizer, label mappings and checkpoints go to ``paths.output_dir``;
``run_summary.json`` there records the split sizes, timings and final metrics.
ONNX export and signing are separate steps (``model/src/quantitize.py``).

Usage, from the repo root:
    python model/flows/experiments/train_tpu.py \
        --config model/flows/experiments/mdeberta_german_mix_v4.toml

    # smoke run: 3,000 rows, one epoch, separate output directory
    python model/flows/experiments/train_tpu.py --config <toml> \
        --num-samples 3000 --num-epochs 1 --output-dir /tmp/smoke
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import tomllib
from pathlib import Path

from absl import logging

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT / "model"))
sys.path.insert(0, str(REPO_ROOT))

from src.config import EnvironmentSetup, TrainingConfig  # noqa: E402


def build_config(args: argparse.Namespace) -> TrainingConfig:
    cfg = tomllib.loads(Path(args.config).read_text())
    training = cfg.setdefault("training", {})
    paths = cfg.setdefault("paths", {})
    if args.output_dir:
        paths["output_dir"] = args.output_dir
    if args.num_epochs is not None:
        training["num_epochs"] = args.num_epochs
    if args.max_eval_samples is not None:
        training["max_eval_samples"] = args.max_eval_samples
    config = TrainingConfig.from_toml(cfg)
    if args.seed is not None:
        # --seed beats TRAINING_SEED, which from_toml puts ahead of the file.
        config.seed = args.seed
    return config


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--config", required=True, help="Training TOML")
    ap.add_argument("--output-dir", help="Override paths.output_dir")
    ap.add_argument("--num-epochs", type=int, help="Override training.num_epochs")
    ap.add_argument(
        "--max-eval-samples", type=int, help="Override training.max_eval_samples"
    )
    ap.add_argument(
        "--seed", type=int, help="Override training.seed (else TRAINING_SEED)"
    )
    ap.add_argument(
        "--num-samples",
        type=int,
        default=0,
        help="Use only this many rows (smoke runs; 0 = all)",
    )
    args = ap.parse_args()
    logging.set_verbosity(logging.INFO)

    config = build_config(args)
    EnvironmentSetup.disable_wandb()
    EnvironmentSetup.check_gpu()
    config.print_summary()
    logging.info(f"  Seed: {config.seed}")

    from src.preprocessing import DatasetProcessor
    from src.trainer import PIITrainer

    summary: dict = {"config": args.config, "seed": config.seed}
    start = time.time()
    train_dataset, val_dataset, mappings = DatasetProcessor(config).prepare_datasets(
        subsample_count=args.num_samples
    )
    summary["train_rows"] = len(train_dataset)
    summary["validation_rows"] = len(val_dataset)
    summary["preprocess_seconds"] = round(time.time() - start, 1)

    trainer = PIITrainer(config)
    trainer.load_label_mappings(mappings)
    trainer.initialize_model()

    start = time.time()
    try:
        hf_trainer = trainer.train(train_dataset, val_dataset)
    except KeyboardInterrupt:
        logging.warning("Interrupted (SIGINT); checkpoints so far are kept.")
        return 130
    summary["train_seconds"] = round(time.time() - start, 1)
    summary["best_checkpoint"] = hf_trainer.state.best_model_checkpoint
    summary["best_metric"] = hf_trainer.state.best_metric
    summary["epochs_run"] = hf_trainer.state.epoch
    summary["global_steps"] = hf_trainer.state.global_step

    start = time.time()
    results = trainer.evaluate(val_dataset, hf_trainer)
    summary["final_eval_seconds"] = round(time.time() - start, 1)
    summary["final_eval"] = {
        k: v for k, v in results.items() if isinstance(v, int | float)
    }

    if hf_trainer.is_world_process_zero():
        out = Path(config.output_dir) / "run_summary.json"
        out.write_text(json.dumps(summary, indent=2))
        logging.info(f"Run summary written to {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
