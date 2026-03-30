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
        PIIDetectionModel,
    )
except ImportError:
    # Fallback for direct execution
    from callbacks import CleanMetricsCallback
    from config import TrainingConfig

    from model import (
        MaskedSparseCategoricalCrossEntropy,
        PIIDetectionModel,
    )


class PIIModelTrainer(Trainer):
    """Custom Trainer for PII detection."""

    def __init__(self, pii_loss_fn=None, **kwargs):
        """
        Initialize PII trainer.

        Args:
            pii_loss_fn: Loss function for PII detection
            **kwargs: Additional arguments for Trainer
        """
        super().__init__(**kwargs)
        self.pii_loss_fn = pii_loss_fn

    def compute_loss(
        self,
        model,
        inputs,
        return_outputs: bool = False,
        num_items_in_batch: int | None = None,
    ):
        """
        Compute PII detection loss.

        Args:
            model: The model being trained
            inputs: Input batch containing input_ids, attention_mask, pii_labels
            return_outputs: Whether to return outputs
            num_items_in_batch: Number of items in batch (optional)

        Returns:
            Loss value (and outputs if return_outputs=True)
        """
        input_ids = inputs.get("input_ids")
        attention_mask = inputs.get("attention_mask")
        pii_labels = inputs.get("pii_labels")

        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            pii_labels=pii_labels,
        )

        if self.pii_loss_fn is not None:
            loss = self.pii_loss_fn(outputs["pii_logits"], pii_labels)
        else:
            loss = functional.cross_entropy(
                outputs["pii_logits"].view(-1, outputs["pii_logits"].size(-1)),
                pii_labels.view(-1),
                ignore_index=-100,
            )

        return (loss, outputs) if return_outputs else loss

    def prediction_step(self, model, inputs, prediction_loss_only, ignore_keys=None):
        """
        Override prediction step to handle PII outputs.

        Args:
            model: The model
            inputs: Input batch
            prediction_loss_only: Whether to only compute loss
            ignore_keys: Keys to ignore

        Returns:
            Tuple of (loss, logits, labels) or (loss, None, None)
        """
        has_labels = "pii_labels" in inputs
        inputs = self._prepare_inputs(inputs)

        with torch.no_grad():
            if has_labels:
                loss, outputs = self.compute_loss(model, inputs, return_outputs=True)
            else:
                loss = None
                outputs = model(**inputs)

        if prediction_loss_only:
            return (loss, None, None)

        pii_logits = outputs.get("pii_logits")
        pii_labels = inputs.get("pii_labels")

        predictions = pii_logits.detach().cpu() if pii_logits is not None else None
        labels = pii_labels.detach().cpu() if pii_labels is not None else None

        return (loss, predictions, labels)


class PIITrainer:
    """Main trainer class for PII detection model."""

    def __init__(self, config: TrainingConfig):
        """
        Initialize PII trainer.

        Args:
            config: Training configuration
        """
        self.config = config
        self.tokenizer = AutoTokenizer.from_pretrained(config.model_name)
        self.model = None
        self.pii_label2id = None
        self.pii_id2label = None
        self.pii_loss_fn = None

        if config.use_wandb:
            try:
                import wandb

                wandb.init(
                    project="pii-detection",
                    name=f"bert-{config.model_name.split('/')[-1]}",
                )
            except Exception as e:
                logging.warning(f"Warning: wandb not available ({e})")
                self.config.use_wandb = False

    def load_label_mappings(self, mappings: dict):
        """Load label mappings from dataset processor."""
        self.pii_label2id = mappings["pii"]["label2id"]
        self.pii_id2label = {int(k): v for k, v in mappings["pii"]["id2label"].items()}
        logging.info(f"✅ Loaded {len(self.pii_label2id)} PII label mappings")

    def initialize_model(self):
        """Initialize the PII detection model."""
        if self.pii_label2id is None:
            raise ValueError("Label mappings must be loaded first")

        num_pii_labels = len(self.pii_label2id)

        self.model = PIIDetectionModel(
            model_name=self.config.model_name,
            num_pii_labels=num_pii_labels,
            id2label_pii=self.pii_id2label,
        )

        # Initialize loss function
        if self.config.use_custom_loss:
            self.pii_loss_fn = MaskedSparseCategoricalCrossEntropy(
                pad_label=-100,
                class_weights=self.config.class_weights,
                num_classes=num_pii_labels,
                reduction="mean",
            )
            logging.info(f"✅ Initialized PII loss ({num_pii_labels} classes)")

        logging.info(f"✅ Model initialized with {num_pii_labels} PII labels")

    def compute_metrics(self, eval_pred) -> dict[str, float]:
        """
        Compute evaluation metrics for PII detection.

        Args:
            eval_pred: EvalPrediction object with predictions and label_ids

        Returns:
            Dictionary of metrics
        """
        predictions = eval_pred.predictions
        label_ids = eval_pred.label_ids

        # Handle numpy array predictions
        pii_predictions = predictions
        pii_labels = label_ids

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
                safe_label = label.replace("-", "_").replace(" ", "_")
                metrics[f"eval_pii_f1_{safe_label}"] = float(f1)
                metrics[f"eval_pii_precision_{safe_label}"] = float(prec)
                metrics[f"eval_pii_recall_{safe_label}"] = float(rec)

        return metrics

    def train(self, train_dataset: Dataset, val_dataset: Dataset) -> Trainer:
        """
        Train the PII detection model.

        Args:
            train_dataset: Training dataset
            val_dataset: Validation dataset

        Returns:
            Trained Trainer instance
        """
        if self.model is None:
            raise ValueError("Model must be initialized first")

        # Data collator for PII detection
        def data_collator(features):
            """Collate function with padding."""
            pad_token_id = (
                self.tokenizer.pad_token_id
                if self.tokenizer.pad_token_id is not None
                else 0
            )

            max_length = max(len(f["input_ids"]) for f in features)

            batch = {}
            padded_input_ids = []
            padded_attention_mask = []
            padded_pii_labels = []

            for f in features:
                seq_len = len(f["input_ids"])
                padding_length = max_length - seq_len

                padded_input_ids.append(
                    f["input_ids"] + [pad_token_id] * padding_length
                )
                padded_attention_mask.append(f["attention_mask"] + [0] * padding_length)
                padded_pii_labels.append(f["pii_labels"] + [-100] * padding_length)

            batch["input_ids"] = torch.tensor(padded_input_ids, dtype=torch.long)
            batch["attention_mask"] = torch.tensor(
                padded_attention_mask, dtype=torch.long
            )
            batch["pii_labels"] = torch.tensor(padded_pii_labels, dtype=torch.long)

            return batch

        # Suppress transformers logging output (we use custom callback)
        import logging as python_logging

        import transformers

        transformers.logging.set_verbosity_error()

        trainer_logger = python_logging.getLogger("transformers.trainer")
        trainer_logger.setLevel(python_logging.ERROR)

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
            eval_strategy="epoch",
            save_strategy="epoch",
            load_best_model_at_end=True,
            metric_for_best_model="eval_pii_f1",
            greater_is_better=True,
            report_to=None,
            save_total_limit=3,
            seed=self.config.seed,
            dataloader_pin_memory=False,
            remove_unused_columns=False,
            logging_first_step=False,
            disable_tqdm=False,
            log_level="error",
        )

        # Set up callbacks
        callbacks = []
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

        # Initialize trainer
        trainer = PIIModelTrainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            data_collator=data_collator,
            compute_metrics=self.compute_metrics,
            pii_loss_fn=self.pii_loss_fn,
            callbacks=callbacks if callbacks else None,
        )

        logging.info("✅ Using PIIModelTrainer")

        # Train
        logging.info("\n🏋️  Starting training...")
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
        for metric_type in ["f1", "precision", "recall"]:
            metric_keys = [k for k in pii_metrics.keys() if metric_type in k]
            if metric_keys:
                logging.info(f"  {metric_type.upper()}:")
                for key in sorted(metric_keys):
                    if "per_class" not in key and not any(
                        label in key for label in ["B-", "I-"]
                    ):
                        logging.info(f"    {key}: {pii_metrics[key]:.4f}")

        return results

    def save_to_google_drive(self, drive_folder: str = "MyDrive/pii_models"):
        """
        Copy the trained model to Google Drive with timestamp.

        Args:
            drive_folder: Target folder path in Google Drive (relative to mount point)

        Returns:
            Path to saved model in Google Drive
        """
        drive_path = f"/content/drive/{drive_folder}"
        Path(drive_path).mkdir(parents=True, exist_ok=True)

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
