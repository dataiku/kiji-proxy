"""Custom callbacks for training."""

from absl import logging
from transformers import TrainerCallback


class CleanMetricsCallback(TrainerCallback):
    """Custom callback to print clean, readable metrics during training."""

    def on_log(self, args, state, control, logs=None, **kwargs):
        """Called when logging metrics - print clean format and clear original logs."""
        if logs is None:
            return

        # Check if this is eval metrics
        is_eval = any(k.startswith("eval_") for k in logs.keys())

        if is_eval:
            # Format and print metrics cleanly
            step = state.global_step
            epoch = logs.get("epoch", 0)

            logging.warning(f"\n{'=' * 60}")
            logging.warning(f"📊 Evaluation at Step {step} (Epoch {epoch:.2f})")
            logging.warning(f"={'=' * 60}")

            # PII metrics
            f1 = logs.get("eval_f1_weighted") or logs.get("eval_f1")
            acc = logs.get("eval_accuracy")
            if f1 is not None:
                logging.warning("🔍 PII Detection:")
                logging.warning(f"   F1:  {f1:.4f}")
                if acc is not None:
                    logging.warning(f"   Acc: {acc:.4f}")

            # Loss
            loss = logs.get("eval_loss")
            if loss is not None:
                logging.warning(f"📉 Loss: {loss:.4f}")

            logging.warning(f"={'=' * 60}\n")
            # Don't clear logs - trainer needs them for best model tracking
