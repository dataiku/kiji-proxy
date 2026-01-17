"""Training logic and trainer classes."""

import shutil
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
from absl import logging
from datasets import Dataset
from sklearn.metrics import (
    f1_score,
    precision_score,
    recall_score,
)
from torch.nn import functional
from transformers import (
    AutoTokenizer,
    EarlyStoppingCallback,
    Trainer,
    TrainingArguments,
)

# Import from local modules
try:
    from .callbacks import CleanMetricsCallback
    from .config import TrainingConfig
    from .model import (
        MaskedSparseCategoricalCrossEntropy,
        MultiTaskLoss,
        MultiTaskPIIDetectionModel,
    )
except ImportError:
    # Fallback for direct execution
    from callbacks import CleanMetricsCallback
    from config import TrainingConfig

    from model import (
        MaskedSparseCategoricalCrossEntropy,
        MultiTaskLoss,
        MultiTaskPIIDetectionModel,
    )


class MultiTaskTrainer(Trainer):
    """Custom Trainer for multi-task learning (PII + co-reference detection)."""

    def __init__(self, multi_task_loss_fn=None, **kwargs):
        """
        Initialize multi-task trainer.

        Args:
            multi_task_loss_fn: Multi-task loss function to use
            **kwargs: Additional arguments for Trainer
        """
        super().__init__(**kwargs)
        self.multi_task_loss_fn = multi_task_loss_fn

    def compute_loss(
        self,
        model,
        inputs,
        return_outputs: bool = False,
        num_items_in_batch: int | None = None,
    ):
        """
        Override the default loss computation to use multi-task loss.

        Args:
            model: The model being trained
            inputs: Input batch containing input_ids, attention_mask, pii_labels, coref_labels
            return_outputs: Whether to return outputs
            num_items_in_batch: Number of items in batch (optional, for compatibility with newer transformers)

        Returns:
            Loss value (and outputs if return_outputs=True)
        """
        # Extract inputs
        input_ids = inputs.get("input_ids")
        attention_mask = inputs.get("attention_mask")
        pii_labels = inputs.get("pii_labels")
        coref_labels = inputs.get("coref_labels")

        # Forward pass
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            pii_labels=pii_labels,
            coref_labels=coref_labels,
        )

        # Compute multi-task loss
        if self.multi_task_loss_fn is not None:
            loss = self.multi_task_loss_fn(
                outputs["pii_logits"],
                pii_labels,
                outputs["coref_logits"],
                coref_labels,
            )
        else:
            # Fallback: simple sum of individual losses
            pii_loss = functional.cross_entropy(
                outputs["pii_logits"].view(-1, outputs["pii_logits"].size(-1)),
                pii_labels.view(-1),
                ignore_index=-100,
            )
            coref_loss = functional.cross_entropy(
                outputs["coref_logits"].view(-1, outputs["coref_logits"].size(-1)),
                coref_labels.view(-1),
                ignore_index=-100,
            )
            loss = pii_loss + coref_loss

        return (loss, outputs) if return_outputs else loss

    def prediction_step(self, model, inputs, prediction_loss_only, ignore_keys=None):
        """
        Override prediction step to handle multi-task outputs.

        Args:
            model: The model
            inputs: Input batch
            prediction_loss_only: Whether to only compute loss
            ignore_keys: Keys to ignore

        Returns:
            Tuple of (loss, logits, labels) or (loss, None, None)
        """
        has_labels = "pii_labels" in inputs or "coref_labels" in inputs
        inputs = self._prepare_inputs(inputs)

        with torch.no_grad():
            if has_labels:
                loss, outputs = self.compute_loss(model, inputs, return_outputs=True)
            else:
                loss = None
                outputs = model(**inputs)

        if prediction_loss_only:
            return (loss, None, None)

        # Extract logits and labels for both tasks
        pii_logits = outputs.get("pii_logits")
        coref_logits = outputs.get("coref_logits")
        pii_labels = inputs.get("pii_labels")
        coref_labels = inputs.get("coref_labels")

        # Return predictions as torch tensors (not numpy) so padding can be applied
        # Use dict format for better compatibility with padding functions
        # The transformers library will convert to numpy after padding
        if pii_logits is not None and coref_logits is not None:
            # Keep as tensors on CPU for padding - use dict for compatibility
            predictions = {
                "pii_logits": pii_logits.detach().cpu(),
                "coref_logits": coref_logits.detach().cpu(),
            }
        else:
            predictions = None

        if pii_labels is not None and coref_labels is not None:
            # Keep as tensors on CPU for padding - use dict for compatibility
            labels = {
                "pii_labels": pii_labels.detach().cpu(),
                "coref_labels": coref_labels.detach().cpu(),
            }
        else:
            labels = None

        return (loss, predictions, labels)


class PIITrainer:
    """Main trainer class for multi-task PII and co-reference detection model."""

    def __init__(self, config: TrainingConfig):
        """
        Initialize multi-task trainer.

        Args:
            config: Training configuration
        """
        self.config = config
        self.tokenizer = AutoTokenizer.from_pretrained(config.model_name)
        self.model = None
        self.pii_label2id = None
        self.pii_id2label = None
        self.coref_id2label = None
        self.num_coref_labels = None
        self.multi_task_loss_fn = None

        if config.use_wandb:
            try:
                import wandb

                wandb.init(
                    project="pii-detection-multitask",
                    name=f"bert-{config.model_name.split('/')[-1]}",
                )
            except Exception as e:
                logging.warning(f"Warning: wandb not available ({e})")
                self.config.use_wandb = False

    def load_label_mappings(self, mappings: dict, coref_info: dict):
        """Load label mappings from dataset processor."""
        self.pii_label2id = mappings["pii"]["label2id"]
        self.pii_id2label = {int(k): v for k, v in mappings["pii"]["id2label"].items()}
        self.coref_id2label = {
            int(k): v for k, v in mappings["coref"]["id2label"].items()
        }
        self.num_coref_labels = coref_info["num_coref_labels"]
        logging.info(f"✅ Loaded {len(self.pii_label2id)} PII label mappings")
        logging.info(f"✅ Loaded {self.num_coref_labels} co-reference label mappings")

    def initialize_model(self):
        """Initialize the multi-task model."""
        if self.pii_label2id is None or self.num_coref_labels is None:
            raise ValueError("Label mappings must be loaded first")

        num_pii_labels = len(self.pii_label2id)

        self.model = MultiTaskPIIDetectionModel(
            model_name=self.config.model_name,
            num_pii_labels=num_pii_labels,
            num_coref_labels=self.num_coref_labels,
            id2label_pii=self.pii_id2label,
            id2label_coref=self.coref_id2label,
        )

        # Initialize multi-task loss function
        if self.config.use_custom_loss:
            pii_loss_fn = MaskedSparseCategoricalCrossEntropy(
                pad_label=-100,
                class_weights=self.config.class_weights,
                num_classes=num_pii_labels,
                reduction="mean",
            )
            coref_loss_fn = MaskedSparseCategoricalCrossEntropy(
                pad_label=-100,
                num_classes=self.num_coref_labels,
                reduction="mean",
            )

            self.multi_task_loss_fn = MultiTaskLoss(
                pii_loss_fn=pii_loss_fn,
                coref_loss_fn=coref_loss_fn,
                pii_weight=self.config.pii_loss_weight,
                coref_weight=self.config.coref_loss_weight,
            )
            logging.info(
                f"✅ Initialized multi-task loss (PII: {num_pii_labels} classes, "
                f"Co-ref: {self.num_coref_labels} classes)"
            )

        logging.info(
            f"✅ Model initialized with {num_pii_labels} PII labels and {self.num_coref_labels} co-reference labels"
        )

    def compute_metrics(self, eval_pred) -> dict[str, float]:
        """
        Compute evaluation metrics for both tasks.

        Args:
            eval_pred: EvalPrediction object with predictions and label_ids

        Returns:
            Dictionary of metrics for both tasks
        """
        predictions = eval_pred.predictions
        label_ids = eval_pred.label_ids

        # Handle different prediction formats
        if isinstance(predictions, dict):
            pii_predictions = predictions.get("pii_logits", predictions)
            coref_predictions = predictions.get("coref_logits")
            pii_labels = (
                label_ids.get("pii_labels")
                if isinstance(label_ids, dict)
                else label_ids
            )
            coref_labels = (
                label_ids.get("coref_labels") if isinstance(label_ids, dict) else None
            )
        elif isinstance(predictions, (tuple, list)) and len(predictions) == 2:
            # Tuple/list format: (pii_predictions, coref_predictions) or [pii_predictions, coref_predictions]
            pii_predictions, coref_predictions = predictions
            if isinstance(label_ids, (tuple, list)) and len(label_ids) == 2:
                pii_labels, coref_labels = label_ids
            else:
                pii_labels = label_ids
                coref_labels = None
        else:
            # Single task fallback
            pii_predictions = predictions
            pii_labels = label_ids
            coref_predictions = None
            coref_labels = None

        # PII detection metrics
        pii_preds = np.argmax(pii_predictions, axis=2)
        pii_true_preds = [
            [
                self.pii_id2label.get(int(p), "O")
                for (p, label_id) in zip(prediction, label, strict=True)
                if label_id != -100
            ]
            for prediction, label in zip(pii_preds, pii_labels, strict=True)
        ]
        pii_true_labels = [
            [
                self.pii_id2label.get(int(label_id), "O")
                for (p, label_id) in zip(prediction, label, strict=True)
                if label_id != -100
            ]
            for prediction, label in zip(pii_preds, pii_labels, strict=True)
        ]

        pii_flat_preds = [item for sublist in pii_true_preds for item in sublist]
        pii_flat_labels = [item for sublist in pii_true_labels for item in sublist]

        # Compute PII detection metrics
        pii_f1_weighted = f1_score(pii_flat_labels, pii_flat_preds, average="weighted")
        pii_f1_macro = f1_score(pii_flat_labels, pii_flat_preds, average="macro")
        pii_precision_weighted = precision_score(
            pii_flat_labels, pii_flat_preds, average="weighted", zero_division=0
        )
        pii_precision_macro = precision_score(
            pii_flat_labels, pii_flat_preds, average="macro", zero_division=0
        )
        pii_recall_weighted = recall_score(
            pii_flat_labels, pii_flat_preds, average="weighted", zero_division=0
        )
        pii_recall_macro = recall_score(
            pii_flat_labels, pii_flat_preds, average="macro", zero_division=0
        )

        # Per-class metrics
        unique_labels = sorted(set(pii_flat_labels + pii_flat_preds))
        pii_f1_per_class = f1_score(
            pii_flat_labels,
            pii_flat_preds,
            average=None,
            labels=unique_labels,
            zero_division=0,
        )
        pii_precision_per_class = precision_score(
            pii_flat_labels,
            pii_flat_preds,
            average=None,
            labels=unique_labels,
            zero_division=0,
        )
        pii_recall_per_class = recall_score(
            pii_flat_labels,
            pii_flat_preds,
            average=None,
            labels=unique_labels,
            zero_division=0,
        )

        # Build metrics dictionary
        metrics = {
            "eval_pii_f1_weighted": pii_f1_weighted,
            "eval_pii_f1_macro": pii_f1_macro,
            "eval_pii_precision_weighted": pii_precision_weighted,
            "eval_pii_precision_macro": pii_precision_macro,
            "eval_pii_recall_weighted": pii_recall_weighted,
            "eval_pii_recall_macro": pii_recall_macro,
            # Keep backward compatibility
            "eval_pii_f1": pii_f1_weighted,
        }

        # Add per-class metrics (limit to reasonable number of classes)
        non_o_labels = [label for label in unique_labels if label != "O"]
        if len(non_o_labels) <= 20:
            for label, f1, prec, rec in zip(
                unique_labels,
                pii_f1_per_class,
                pii_precision_per_class,
                pii_recall_per_class,
                strict=True,
            ):
                # Sanitize label name for metric key
                safe_label = label.replace("-", "_").replace(" ", "_")
                metrics[f"eval_pii_f1_{safe_label}"] = float(f1)
                metrics[f"eval_pii_precision_{safe_label}"] = float(prec)
                metrics[f"eval_pii_recall_{safe_label}"] = float(rec)

        # Co-reference detection metrics
        if coref_predictions is not None and coref_labels is not None:
            coref_preds = np.argmax(coref_predictions, axis=2)
            coref_true_preds = [
                [
                    int(p)
                    for (p, label_id) in zip(prediction, label, strict=True)
                    if label_id != -100
                ]
                for prediction, label in zip(coref_preds, coref_labels, strict=True)
            ]
            coref_true_labels = [
                [
                    int(label_id)
                    for (p, label_id) in zip(prediction, label, strict=True)
                    if label_id != -100
                ]
                for prediction, label in zip(coref_preds, coref_labels, strict=True)
            ]

            coref_flat_preds = [
                item for sublist in coref_true_preds for item in sublist
            ]
            coref_flat_labels = [
                item for sublist in coref_true_labels for item in sublist
            ]

            # Compute co-reference detection metrics
            coref_f1_weighted = f1_score(
                coref_flat_labels, coref_flat_preds, average="weighted", zero_division=0
            )
            coref_f1_macro = f1_score(
                coref_flat_labels, coref_flat_preds, average="macro", zero_division=0
            )
            coref_precision_weighted = precision_score(
                coref_flat_labels, coref_flat_preds, average="weighted", zero_division=0
            )
            coref_precision_macro = precision_score(
                coref_flat_labels, coref_flat_preds, average="macro", zero_division=0
            )
            coref_recall_weighted = recall_score(
                coref_flat_labels, coref_flat_preds, average="weighted", zero_division=0
            )
            coref_recall_macro = recall_score(
                coref_flat_labels, coref_flat_preds, average="macro", zero_division=0
            )

            # Per-class metrics for co-reference
            unique_coref_labels = sorted(set(coref_flat_labels + coref_flat_preds))
            coref_f1_per_class = f1_score(
                coref_flat_labels,
                coref_flat_preds,
                average=None,
                labels=unique_coref_labels,
                zero_division=0,
            )
            coref_precision_per_class = precision_score(
                coref_flat_labels,
                coref_flat_preds,
                average=None,
                labels=unique_coref_labels,
                zero_division=0,
            )
            coref_recall_per_class = recall_score(
                coref_flat_labels,
                coref_flat_preds,
                average=None,
                labels=unique_coref_labels,
                zero_division=0,
            )

            # Add co-reference metrics
            metrics.update(
                {
                    "eval_coref_f1_weighted": coref_f1_weighted,
                    "eval_coref_f1_macro": coref_f1_macro,
                    "eval_coref_precision_weighted": coref_precision_weighted,
                    "eval_coref_precision_macro": coref_precision_macro,
                    "eval_coref_recall_weighted": coref_recall_weighted,
                    "eval_coref_recall_macro": coref_recall_macro,
                    # Keep backward compatibility
                    "eval_coref_f1": coref_f1_weighted,
                }
            )

            # Add per-class metrics for co-reference
            if len(unique_coref_labels) <= 20:
                for label, f1, prec, rec in zip(
                    unique_coref_labels,
                    coref_f1_per_class,
                    coref_precision_per_class,
                    coref_recall_per_class,
                    strict=True,
                ):
                    safe_label = f"cluster_{label}"
                    metrics[f"eval_coref_f1_{safe_label}"] = float(f1)
                    metrics[f"eval_coref_precision_{safe_label}"] = float(prec)
                    metrics[f"eval_coref_recall_{safe_label}"] = float(rec)

        return metrics

    def train(self, train_dataset: Dataset, val_dataset: Dataset) -> Trainer:
        """
        Train the multi-task model.

        Args:
            train_dataset: Training dataset
            val_dataset: Validation dataset

        Returns:
            Trained Trainer instance
        """
        if self.model is None:
            raise ValueError("Model must be initialized first")

        # Custom data collator for multi-task learning
        def multi_task_collator(features):
            """Collate function for multi-task learning with padding."""
            # Get pad token ID from tokenizer
            pad_token_id = (
                self.tokenizer.pad_token_id
                if self.tokenizer.pad_token_id is not None
                else 0
            )

            # Find maximum sequence length in batch
            max_length = max(len(f["input_ids"]) for f in features)

            batch = {}

            # Pad and convert input_ids
            padded_input_ids = []
            padded_attention_mask = []
            padded_pii_labels = []
            padded_coref_labels = []

            for f in features:
                seq_len = len(f["input_ids"])
                padding_length = max_length - seq_len

                # Pad input_ids with pad_token_id
                padded_input_ids.append(
                    f["input_ids"] + [pad_token_id] * padding_length
                )

                # Pad attention_mask with 0s
                padded_attention_mask.append(f["attention_mask"] + [0] * padding_length)

                # Pad labels with -100 (ignore index)
                padded_pii_labels.append(f["pii_labels"] + [-100] * padding_length)
                padded_coref_labels.append(f["coref_labels"] + [-100] * padding_length)

            # Convert to tensors
            batch["input_ids"] = torch.tensor(padded_input_ids, dtype=torch.long)
            batch["attention_mask"] = torch.tensor(
                padded_attention_mask, dtype=torch.long
            )
            batch["pii_labels"] = torch.tensor(padded_pii_labels, dtype=torch.long)
            batch["coref_labels"] = torch.tensor(padded_coref_labels, dtype=torch.long)

            return batch

        # Suppress transformers logging output (we use custom callback)
        import transformers

        transformers.logging.set_verbosity_error()

        # Training arguments
        training_args = TrainingArguments(
            output_dir=self.config.output_dir,
            num_train_epochs=self.config.num_epochs,
            per_device_train_batch_size=self.config.batch_size,
            per_device_eval_batch_size=self.config.batch_size,
            warmup_steps=self.config.warmup_steps,
            weight_decay=self.config.weight_decay,
            learning_rate=self.config.learning_rate,
            logging_dir=f"{self.config.output_dir}/logs",
            logging_steps=self.config.logging_steps,
            eval_strategy="steps",
            eval_steps=self.config.eval_steps,
            save_strategy="steps",
            save_steps=self.config.save_steps,
            load_best_model_at_end=True,
            metric_for_best_model="eval_pii_f1",  # Use PII F1 as primary metric
            greater_is_better=True,
            report_to=None,
            save_total_limit=3,
            seed=self.config.seed,
            dataloader_pin_memory=False,
            remove_unused_columns=False,
            logging_first_step=False,
            disable_tqdm=False,  # Keep progress bar
            log_level="error",  # Suppress info logs
        )

        # Set up callbacks
        callbacks = []

        # Add clean metrics logging callback
        callbacks.append(CleanMetricsCallback())

        if self.config.early_stopping_enabled:
            early_stopping_callback = EarlyStoppingCallback(
                early_stopping_patience=self.config.early_stopping_patience,
                early_stopping_threshold=self.config.early_stopping_threshold,
            )
            callbacks.append(early_stopping_callback)
            logging.info(
                f"✅ Early stopping enabled (patience={self.config.early_stopping_patience}, "
                f"threshold={self.config.early_stopping_threshold})"
            )

        # Initialize multi-task trainer
        trainer = MultiTaskTrainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            data_collator=multi_task_collator,
            compute_metrics=self.compute_metrics,
            multi_task_loss_fn=self.multi_task_loss_fn,
            callbacks=callbacks if callbacks else None,
        )

        logging.info("✅ Using MultiTaskTrainer with multi-task loss")

        # Train
        logging.info("\n🏋️  Starting multi-task training...")
        logging.info("=" * 60)
        trainer.train()

        # Save
        trainer.save_model()
        self.tokenizer.save_pretrained(self.config.output_dir)
        logging.info(
            f"\n✅ Training completed. Model saved to {self.config.output_dir}"
        )

        return trainer

    def evaluate(self, test_dataset: Dataset, trainer: Trainer | None = None) -> dict:
        """
        Evaluate the model on test dataset.

        Args:
            test_dataset: Test dataset
            trainer: Optional trainer instance

        Returns:
            Evaluation results
        """
        if trainer is None:
            raise ValueError("Trainer must be provided for evaluation")

        results = trainer.evaluate()

        logging.info("\n📊 Evaluation Results:")
        logging.info("\n🔍 PII Detection Metrics:")
        pii_metrics = {k: v for k, v in results.items() if k.startswith("eval_pii_")}
        # Group metrics by type
        for metric_type in ["f1", "precision", "recall"]:
            metric_keys = [k for k in pii_metrics.keys() if metric_type in k]
            if metric_keys:
                logging.info(f"  {metric_type.upper()}:")
                for key in sorted(metric_keys):
                    if "per_class" not in key and not any(
                        label in key for label in ["B-", "I-", "CLUSTER_"]
                    ):  # Skip per-class in summary
                        logging.info(f"    {key}: {pii_metrics[key]:.4f}")

        if any(k.startswith("eval_coref_") for k in results.keys()):
            logging.info("\n🔍 Co-reference Detection Metrics:")
            coref_metrics = {
                k: v for k, v in results.items() if k.startswith("eval_coref_")
            }
            for metric_type in ["f1", "precision", "recall"]:
                metric_keys = [k for k in coref_metrics.keys() if metric_type in k]
                if metric_keys:
                    logging.info(f"  {metric_type.upper()}:")
                    for key in sorted(metric_keys):
                        if "per_class" not in key and "cluster_" not in key:
                            logging.info(f"    {key}: {coref_metrics[key]:.4f}")

        return results

    def save_to_google_drive(self, drive_folder: str = "MyDrive/pii_models"):
        """
        Copy the trained model to Google Drive with timestamp.

        Args:
            drive_folder: Target folder path in Google Drive (relative to mount point)

        Returns:
            Path to saved model in Google Drive
        """
        # Construct full Google Drive path
        drive_path = f"/content/drive/{drive_folder}"

        # Create target directory if it doesn't exist
        Path(drive_path).mkdir(parents=True, exist_ok=True)

        # Create model name with timestamp
        model_name = Path(self.config.output_dir).name
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_name_with_timestamp = f"{model_name}_{timestamp}"
        target_path = Path(drive_path) / model_name_with_timestamp

        logging.info("\n💾 Copying model to Google Drive...")
        logging.info(f"   Source: {self.config.output_dir}")
        logging.info(f"   Target: {target_path}")

        shutil.copytree(self.config.output_dir, target_path)
        logging.info(f"✅ Model successfully saved to Google Drive at {target_path}")

        return target_path
