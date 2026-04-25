"""Custom callbacks for training."""

import sys
import time

from absl import logging
from transformers import TrainerCallback


class CleanMetricsCallback(TrainerCallback):
    """Custom callback to print clean, readable metrics during training."""

    def __init__(self):
        self.train_start_time = None
        self.total_steps = None

    def on_train_begin(self, args, state, control, **kwargs):
        self.train_start_time = time.time()
        self.total_steps = state.max_steps
        logging.warning(
            f"  Training: {self.total_steps} steps, "
            f"{args.num_train_epochs:.0f} epochs, "
            f"batch_size={args.per_device_train_batch_size}"
        )
        sys.stderr.flush()

    def on_train_end(self, args, state, control, **kwargs):
        elapsed = time.time() - self.train_start_time
        mins = elapsed / 60
        logging.warning(
            f"  Training complete: {state.global_step} steps in {mins:.1f} min "
            f"({state.global_step / elapsed:.1f} steps/s)"
        )
        sys.stderr.flush()

    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        """Called after evaluation - print a formatted metrics table."""
        if metrics is None:
            return

        step = state.global_step
        epoch = metrics.get("epoch", 0)

        lines = []
        lines.append("")
        lines.append("=" * 62)
        lines.append(f"  Evaluation at Step {step} (Epoch {epoch:.1f})")
        lines.append("=" * 62)

        # Overall metrics
        pii_f1 = metrics.get("eval_pii_f1", 0)
        pii_p = metrics.get("eval_pii_precision", 0)
        pii_r = metrics.get("eval_pii_recall", 0)
        loss = metrics.get("eval_loss", 0)

        lines.append(
            f"  Overall   F1={pii_f1:.4f}  P={pii_p:.4f}  R={pii_r:.4f}  Loss={loss:.4f}"
        )
        lines.append("-" * 62)

        # Per-entity metrics table
        lines.append(f"  {'Entity':<20s} {'F1':>7s} {'Prec':>7s} {'Rec':>7s}")
        lines.append(f"  {'─' * 20} {'─' * 7} {'─' * 7} {'─' * 7}")

        entity_metrics = {}
        for key, val in metrics.items():
            if key.startswith("eval_pii_f1_") and key not in (
                "eval_pii_f1_macro",
                "eval_pii_f1_weighted",
                "eval_pii_f1_micro_avg",
                "eval_pii_f1_macro_avg",
                "eval_pii_f1_weighted_avg",
            ):
                entity = key[len("eval_pii_f1_") :]
                entity_metrics[entity] = {
                    "f1": val,
                    "p": metrics.get(f"eval_pii_precision_{entity}", 0),
                    "r": metrics.get(f"eval_pii_recall_{entity}", 0),
                }

        for entity in sorted(entity_metrics, key=lambda e: entity_metrics[e]["f1"]):
            m = entity_metrics[entity]
            lines.append(
                f"  {entity:<20s} {m['f1']:>7.4f} {m['p']:>7.4f} {m['r']:>7.4f}"
            )

        # Aggregates
        lines.append(f"  {'─' * 20} {'─' * 7} {'─' * 7} {'─' * 7}")
        for avg in ("micro_avg", "macro_avg", "weighted_avg"):
            f1 = metrics.get(f"eval_pii_f1_{avg}")
            if f1 is not None:
                p = metrics.get(f"eval_pii_precision_{avg}", 0)
                r = metrics.get(f"eval_pii_recall_{avg}", 0)
                label = avg.replace("_", " ")
                lines.append(f"  {label:<20s} {f1:>7.4f} {p:>7.4f} {r:>7.4f}")

        # Runtime
        runtime = metrics.get("eval_runtime", 0)
        sps = metrics.get("eval_samples_per_second", 0)
        lines.append("-" * 62)
        lines.append(f"  Eval time: {runtime:.1f}s ({sps:.0f} samples/s)")
        lines.append("=" * 62)

        logging.warning("\n".join(lines))
        sys.stderr.flush()

    def on_log(self, args, state, control, logs=None, **kwargs):
        """Called when logging - only print training loss, suppress eval dicts."""
        if logs is None:
            return

        # Skip eval logs (handled by on_evaluate)
        if any(k.startswith("eval_") for k in logs):
            return

        # Print training metrics concisely
        loss = logs.get("loss")
        lr = logs.get("learning_rate")
        epoch = logs.get("epoch", 0)
        if loss is not None:
            step = state.global_step
            pct = step / self.total_steps * 100 if self.total_steps else 0
            elapsed = time.time() - self.train_start_time
            eta_s = (elapsed / step * (self.total_steps - step)) if step > 0 else 0
            eta_m = eta_s / 60
            lr_str = f"  lr={lr:.2e}" if lr is not None else ""
            logging.warning(
                f"  Step {step:>6d}/{self.total_steps} ({pct:4.1f}%)"
                f"  loss={loss:.4f}{lr_str}  epoch={epoch:.1f}"
                f"  ETA={eta_m:.0f}m"
            )
            sys.stderr.flush()
