"""
PII Detection Model Training with BERT - Word-based Preprocessing

This script trains a BERT-like model for detecting Personally Identifiable Information (PII)
using the ai4privacy/pii-masking-400k dataset from HuggingFace.

Key Features:
- Word-based preprocessing with text replacement approach
- Custom masked loss function for handling padding
- Support for multiple BERT-based models
- Configurable training parameters
- Google Drive integration for Colab (automatic model saving)

Usage:
    # Basic usage (saves to Google Drive by default)
    python script2_refactored.py

    # In Colab, or to customize Google Drive settings:
    main(use_google_drive=True, drive_folder="MyDrive/pii_models")

    # To disable Google Drive saving:
    main(use_google_drive=False)
"""

import json
import logging
import os
import re
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import ClassVar

import numpy as np
import torch
from datasets import Dataset, load_dataset
from sklearn.metrics import f1_score
from torch import nn
from torch.nn import functional
from transformers import (
    AutoModelForTokenClassification,
    AutoTokenizer,
    DataCollatorForTokenClassification,
    Trainer,
    TrainingArguments,
)

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


# =============================================================================
# CONFIGURATION
# =============================================================================


@dataclass
class TrainingConfig:
    """Configuration for PII detection model training."""

    # Model settings
    model_name: str = "distilbert-base-cased"  # 66M params, fast

    # Training parameters
    num_epochs: int = 3
    batch_size: int = 32
    learning_rate: float = 3e-5
    max_samples: int = 400000

    # Training optimization
    warmup_steps: int = 500
    weight_decay: float = 0.01
    save_steps: int = 1000
    eval_steps: int = 500
    logging_steps: int = 100
    seed: int = 42

    # Output and logging
    output_dir: str = "./pii_model"
    use_wandb: bool = False
    use_custom_loss: bool = True
    class_weights: dict[int, float] = field(default_factory=dict)

    # Dataset settings
    eval_size_ratio: float = 0.05  # Validation set size as ratio of training
    max_sequence_length: int = 512

    def __post_init__(self):
        """Create output directory after initialization."""
        Path(self.output_dir).mkdir(parents=True, exist_ok=True)

    def print_summary(self):
        """Print configuration summary."""
        logger.info("\n📋 Training Configuration:")
        logger.info(f"  Model: {self.model_name}")
        logger.info(f"  Epochs: {self.num_epochs}")
        logger.info(f"  Batch Size: {self.batch_size}")
        logger.info(f"  Learning Rate: {self.learning_rate}")
        logger.info(f"  Max Samples: {self.max_samples}")
        logger.info(f"  Output Dir: {self.output_dir}")
        logger.info(f"  Custom Loss: {self.use_custom_loss}")


# =============================================================================
# ENVIRONMENT SETUP
# =============================================================================


class EnvironmentSetup:
    """Handles environment setup and package installation."""

    @staticmethod
    def mount_google_drive(mount_point: str = "/content/drive"):
        """
        Mount Google Drive in Colab environment.

        Args:
            mount_point: Path where Google Drive should be mounted
        """
        try:
            from google.colab import drive

            drive.mount(mount_point)
            logger.info(f"✅ Google Drive mounted at {mount_point}")
            return True
        except ImportError:
            logger.warning("⚠️  Not running in Google Colab - skipping Drive mount")
            return False
        except Exception:
            logger.exception("❌ Failed to mount Google Drive")
            return False

    @staticmethod
    def disable_wandb():
        """Disable Weights & Biases to avoid API key prompts."""
        os.environ["WANDB_DISABLED"] = "true"
        os.environ["WANDB_MODE"] = "disabled"
        os.environ["WANDB_PROJECT"] = ""
        os.environ["WANDB_ENTITY"] = ""
        logger.info("✅ Weights & Biases (wandb) disabled")

    @staticmethod
    def install_package(package_list: list[str], index_url: str | None = None):
        """Install packages with optional index URL."""
        cmd = [sys.executable, "-m", "pip", "install", "-q"]
        if index_url:
            cmd.extend(["--index-url", index_url])
        cmd.extend(package_list)

        try:
            subprocess.check_call(cmd)
            logger.info(f"✅ Successfully installed: {', '.join(package_list)}")
        except subprocess.CalledProcessError:
            logger.exception(f"❌ Failed to install: {', '.join(package_list)}")
            if index_url:
                logger.info("Trying fallback installation...")
                cmd_fallback = [sys.executable, "-m", "pip", "install", "-q", *package_list]
                subprocess.check_call(cmd_fallback)
                logger.info("✅ Fallback installation successful")
            else:
                raise

    @staticmethod
    def setup_pytorch():
        """Install PyTorch with CUDA support if available."""
        logger.info("Installing PyTorch...")
        try:
            EnvironmentSetup.install_package(
                ["torch", "torchvision", "torchaudio"],
                index_url="https://download.pytorch.org/whl/cu118",
            )
        except Exception:
            logger.warning("CUDA installation failed, installing CPU version...")
            EnvironmentSetup.install_package(["torch", "torchvision", "torchaudio"])

    @staticmethod
    def setup_dependencies():
        """Install all required dependencies."""
        logger.info("Installing required packages...")
        packages = [
            "transformers",
            "datasets",
            "scikit-learn",
            "tqdm",
            "psutil",
            "accelerate",
        ]
        EnvironmentSetup.install_package(packages)

    @staticmethod
    def check_gpu():
        """Check and print GPU availability."""
        logger.info(f"\nCUDA available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            logger.info(f"GPU: {torch.cuda.get_device_name(0)}")
            logger.info(
                f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB"
            )


# =============================================================================
# CUSTOM LOSS FUNCTION
# =============================================================================


class MaskedSparseCategoricalCrossEntropy(nn.Module):
    """
    PyTorch implementation of masked sparse categorical cross-entropy loss.

    This loss function ignores padding tokens (typically labeled as -100) and
    supports class weights for handling imbalanced datasets.
    """

    def __init__(
        self,
        pad_label: int = -100,
        class_weights: dict[int, float] | None = None,
        num_classes: int | None = None,
        reduction: str = "mean",
    ):
        """
        Initialize the masked loss function.

        Args:
            pad_label: Label value for padding tokens (HuggingFace standard: -100)
            class_weights: Dictionary mapping class IDs to weights
            num_classes: Total number of classes
            reduction: How to reduce the loss ('mean', 'sum', 'none')
        """
        super().__init__()
        self.pad_label = pad_label
        self.class_weights = class_weights or {}
        self.num_classes = num_classes
        self.reduction = reduction

        if self.num_classes is not None:
            self._build_weight_tensor()

    def _build_weight_tensor(self):
        """Build a weight tensor from class weights dictionary."""
        weight_tensor = torch.ones(self.num_classes, dtype=torch.float32)
        for class_id, weight in self.class_weights.items():
            if 0 <= class_id < self.num_classes:
                weight_tensor[class_id] = float(weight)
        self.register_buffer("weight_tensor", weight_tensor)

    def forward(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
        """
        Compute the masked loss.

        Args:
            y_pred: Model predictions (logits) of shape (batch_size, seq_len, num_classes)
            y_true: True labels of shape (batch_size, seq_len)

        Returns:
            Computed loss value
        """
        # Create mask for non-padded elements
        mask = y_true != self.pad_label

        # Create safe version of y_true to avoid errors with negative labels
        y_true_safe = torch.where(mask, y_true, torch.zeros_like(y_true))

        # Compute cross-entropy loss
        loss = functional.cross_entropy(
            y_pred.view(-1, y_pred.size(-1)), y_true_safe.view(-1), reduction="none"
        )

        # Reshape loss to match input shape
        loss = loss.view(y_true.shape)

        # Apply class weights if available
        if hasattr(self, "weight_tensor"):
            weight_tensor = self.weight_tensor.to(y_true_safe.device)
            sample_weights = weight_tensor[y_true_safe]
            loss = loss * sample_weights

        # Apply padding mask
        loss = torch.where(mask, loss, torch.zeros_like(loss))

        # Apply reduction
        if self.reduction == "mean":
            total_loss = torch.sum(loss)
            total_valid = torch.sum(mask.float())
            return total_loss / torch.clamp(total_valid, min=1e-7)
        elif self.reduction == "sum":
            return torch.sum(loss)
        else:  # 'none'
            return loss


# =============================================================================
# LABEL DEFINITIONS
# =============================================================================


class PIILabels:
    """Manages PII label definitions and mappings."""

    # Standard PII labels
    LABELS: ClassVar[list[str]] = [
        "USERNAME",
        "DATEOFBIRTH",
        "STREET",
        "ZIPCODE",
        "TELEPHONENUM",
        "CREDITCARDNUMBER",
        "EMAIL",
        "CITY",
        "BUILDINGNUM",
        "GIVENNAME",
        "SURNAME",
        "IDCARDNUM",
        "DRIVERLICENSENUM",
        "SOCIALNUM",
        "ACCOUNTNUM",
        "TAXNUM",
    ]

    @classmethod
    def create_label_mappings(cls) -> tuple[dict[str, int], dict[int, str], set]:
        """
        Create label to ID and ID to label mappings.

        Returns:
            Tuple of (label2id, id2label, label_set)
        """
        label2id = {"O": 0}  # "O" represents non-PII tokens
        id2label = {0: "O"}

        for label in cls.LABELS:
            b_label = f"B-{label}"  # Beginning of entity
            i_label = f"I-{label}"  # Inside entity
            label2id[b_label] = len(label2id)
            label2id[i_label] = len(label2id)
            id2label[len(id2label)] = b_label
            id2label[len(id2label)] = i_label

        label_set = set(cls.LABELS)

        return label2id, id2label, label_set

    @classmethod
    def save_mappings(cls, label2id: dict[str, int], id2label: dict[int, str], filepath: str):
        """Save label mappings to JSON file."""
        mappings = {"label2id": label2id, "id2label": id2label}
        with Path(filepath).open("w") as f:
            json.dump(mappings, f, indent=2)
        logger.info(f"✅ Label mappings saved to {filepath}")

    @classmethod
    def load_mappings(cls, filepath: str) -> tuple[dict[str, int], dict[int, str]]:
        """Load label mappings from JSON file."""
        with Path(filepath).open() as f:
            mappings = json.load(f)
        label2id = mappings["label2id"]
        id2label = {int(k): v for k, v in mappings["id2label"].items()}
        return label2id, id2label


# =============================================================================
# DATA PREPROCESSING
# =============================================================================


class WordBasedPreprocessor:
    """
    Handles word-based preprocessing for PII detection.

    This approach replaces sensitive text with label placeholders before tokenization,
    which differs from character offset-based preprocessing.
    """

    def __init__(self, tokenizer: AutoTokenizer, label_set: set, label2id: dict[str, int]):
        """
        Initialize the preprocessor.

        Args:
            tokenizer: HuggingFace tokenizer
            label_set: Set of valid PII labels
            label2id: Label to ID mapping
        """
        self.tokenizer = tokenizer
        self.label_set = label_set
        self.label2id = label2id
        self.error_count = 0

    def generate_sequence_labels(self, text: str, privacy_mask: list[dict]) -> list[str]:
        """
        Generate sequence labels by replacing sensitive text with label placeholders.

        Args:
            text: Source text
            privacy_mask: List of privacy mask dictionaries with 'start', 'end', 'label', 'value'

        Returns:
            List of labels for each word
        """
        # Sort privacy mask by start position (reverse to maintain positions during replacement)
        privacy_mask = sorted(privacy_mask, key=lambda x: x["start"], reverse=True)

        # Replace sensitive text with labels
        for item in privacy_mask:
            label = item["label"]
            start = item["start"]
            end = item["end"]
            value = item["value"]

            # Count words in the sensitive value
            word_count = len(value.split())

            # Replace with appropriate number of label placeholders
            replacement = " ".join([label] * word_count)
            text = text[:start] + replacement + text[end:]

        # Split into words and assign labels
        words = text.split()
        labels = []

        for word in words:
            match = re.search(r"(\w+)", word)
            if match:
                label = match.group(1)
                labels.append(label if label in self.label_set else "O")
            else:
                labels.append("O")

        return labels

    def tokenize_and_align_labels(self, examples: dict) -> dict:
        """
        Tokenize text and align labels using word-based approach.

        Args:
            examples: Batch of examples from dataset

        Returns:
            Tokenized inputs with aligned labels
        """
        # Split text into words
        words = [t.split() for t in examples["source_text"]]

        # Tokenize
        tokenized_inputs = self.tokenizer(
            words, truncation=True, is_split_into_words=True, max_length=512
        )

        # Generate sequence labels
        source_labels = [
            self.generate_sequence_labels(text, mask)
            for text, mask in zip(examples["source_text"], examples["privacy_mask"], strict=True)
        ]

        # Align labels with tokens
        labels = []
        for i, label in enumerate(source_labels):
            word_ids = tokenized_inputs.word_ids(batch_index=i)
            previous_label = None
            label_ids = [-100]  # Start with padding

            try:
                for word_idx in word_ids:
                    if word_idx is None:
                        continue
                    elif label[word_idx] == "O":
                        label_ids.append(self.label2id["O"])
                    elif previous_label == label[word_idx]:
                        # Continuation of entity (Inside)
                        label_ids.append(self.label2id[f"I-{label[word_idx]}"])
                    else:
                        # Beginning of entity
                        label_ids.append(self.label2id[f"B-{label[word_idx]}"])

                    previous_label = label[word_idx]

                # Truncate and add final padding
                label_ids = [*label_ids[:511], -100]
                labels.append(label_ids)

            except Exception:
                self.error_count += 1
                # If error, mark entire sequence as ignored
                labels.append([-100] * len(tokenized_inputs["input_ids"][i]))

        tokenized_inputs["labels"] = labels
        return tokenized_inputs


# =============================================================================
# DATASET PROCESSING
# =============================================================================


class DatasetProcessor:
    """Handles dataset loading and processing."""

    def __init__(self, config: TrainingConfig):
        """
        Initialize dataset processor.

        Args:
            config: Training configuration
        """
        self.config = config
        self.tokenizer = AutoTokenizer.from_pretrained(config.model_name)
        self.label2id, self.id2label, self.label_set = PIILabels.create_label_mappings()
        self.preprocessor = WordBasedPreprocessor(self.tokenizer, self.label_set, self.label2id)

    def process_dataset(self, split: str = "train", max_samples: int | None = None) -> Dataset:
        """
        Process dataset using word-based preprocessing.

        Args:
            split: Dataset split to process ('train', 'validation', 'test')
            max_samples: Maximum number of samples to process

        Returns:
            Processed dataset
        """
        logger.info(f"\n📥 Loading {split} dataset...")
        dataset = load_dataset("ai4privacy/pii-masking-400k", split=split)

        if max_samples:
            dataset = dataset.select(range(min(max_samples, len(dataset))))

        logger.info(f"Processing {len(dataset)} samples...")

        # Reset error count
        self.preprocessor.error_count = 0

        # Process dataset
        processed_dataset = dataset.map(
            self.preprocessor.tokenize_and_align_labels,
            batched=True,
            batch_size=1000,
            remove_columns=dataset.column_names,
        )

        logger.info(f"✅ Successfully processed {len(processed_dataset)} samples")
        if self.preprocessor.error_count > 0:
            logger.warning(f"⚠️  Warning: {self.preprocessor.error_count} samples had errors")

        return processed_dataset

    def prepare_datasets(self) -> tuple[Dataset, Dataset]:
        """
        Prepare training and validation datasets.

        Returns:
            Tuple of (train_dataset, val_dataset)
        """
        # Process training dataset
        train_dataset = self.process_dataset("train", self.config.max_samples)

        # Calculate validation size
        eval_size = int(self.config.max_samples * self.config.eval_size_ratio)
        val_dataset = self.process_dataset("validation", eval_size)

        # Save label mappings
        mappings_path = Path(self.config.output_dir) / "label_mappings.json"
        PIILabels.save_mappings(self.label2id, self.id2label, mappings_path)

        logger.info("\n📊 Dataset Summary:")
        logger.info(f"  Training samples: {len(train_dataset)}")
        logger.info(f"  Validation samples: {len(val_dataset)}")
        logger.info(f"  Number of labels: {len(self.label2id)}")

        return train_dataset, val_dataset


# =============================================================================
# CUSTOM TRAINER
# =============================================================================


class CustomPIITrainer(Trainer):
    """Custom Trainer that uses the masked loss function."""

    def __init__(self, custom_loss_fn: nn.Module | None = None, **kwargs):
        """
        Initialize custom trainer.

        Args:
            custom_loss_fn: Custom loss function to use
            **kwargs: Additional arguments for Trainer
        """
        super().__init__(**kwargs)
        self.custom_loss_fn = custom_loss_fn

    def compute_loss(
        self,
        model,
        inputs,
        return_outputs: bool = False,
    ):
        """
        Override the default loss computation to use custom masked loss.

        Args:
            model: The model being trained
            inputs: Input batch
            return_outputs: Whether to return outputs
            num_items_in_batch: Number of items in batch

        Returns:
            Loss value (and outputs if return_outputs=True)
        """
        labels = inputs.get("labels")
        outputs = model(**inputs)
        logits = outputs.get("logits")

        if self.custom_loss_fn is not None:
            loss = self.custom_loss_fn(logits, labels)
        else:
            loss = outputs.loss

        return (loss, outputs) if return_outputs else loss


# =============================================================================
# PII TRAINER
# =============================================================================


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
        self.label2id = None
        self.id2label = None
        self.custom_loss_fn = None

        if config.use_wandb:
            try:
                import wandb

                wandb.init(
                    project="pii-detection",
                    name=f"bert-{config.model_name.split('/')[-1]}",
                )
            except Exception as e:
                logger.warning(f"Warning: wandb not available ({e})")
                self.config.use_wandb = False

    def load_label_mappings(self, mappings_path: str):
        """Load label mappings from JSON file."""
        self.label2id, self.id2label = PIILabels.load_mappings(mappings_path)
        logger.info(f"✅ Loaded {len(self.label2id)} label mappings")

    def initialize_model(self):
        """Initialize the model for token classification."""
        if self.label2id is None:
            raise ValueError("Label mappings must be loaded first")

        self.model = AutoModelForTokenClassification.from_pretrained(
            self.config.model_name,
            num_labels=len(self.label2id),
            id2label=self.id2label,
            label2id=self.label2id,
        )

        # Initialize custom loss function if requested
        if self.config.use_custom_loss:
            self.custom_loss_fn = MaskedSparseCategoricalCrossEntropy(
                pad_label=-100,
                class_weights=self.config.class_weights,
                num_classes=len(self.label2id),
                reduction="mean",
            )
            logger.info(f"✅ Initialized custom masked loss with {len(self.label2id)} classes")

        logger.info(f"✅ Model initialized with {len(self.label2id)} labels")

    def compute_metrics(self, eval_pred) -> dict[str, float]:
        """
        Compute evaluation metrics.

        Args:
            eval_pred: Evaluation predictions

        Returns:
            Dictionary of metrics
        """
        predictions, labels = eval_pred
        predictions = np.argmax(predictions, axis=2)

        # Remove ignored index (special tokens)
        true_predictions = [
            [self.id2label[p] for (p, label_id) in zip(prediction, label, strict=True) if label_id != -100]
            for prediction, label in zip(predictions, labels, strict=True)
        ]
        true_labels = [
            [self.id2label[label_id] for (p, label_id) in zip(prediction, label, strict=True) if label_id != -100]
            for prediction, label in zip(predictions, labels, strict=True)
        ]

        # Flatten and calculate F1 score
        flat_predictions = [item for sublist in true_predictions for item in sublist]
        flat_labels = [item for sublist in true_labels for item in sublist]

        f1 = f1_score(flat_labels, flat_predictions, average="weighted")

        return {"f1": f1}

    def train(self, train_dataset: Dataset, val_dataset: Dataset) -> Trainer:
        """
        Train the model.

        Args:
            train_dataset: Training dataset
            val_dataset: Validation dataset

        Returns:
            Trained Trainer instance
        """
        if self.model is None:
            raise ValueError("Model must be initialized first")

        # Data collator
        data_collator = DataCollatorForTokenClassification(tokenizer=self.tokenizer, padding=True)

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
            metric_for_best_model="eval_f1",
            greater_is_better=True,
            report_to=None,
            save_total_limit=3,
            seed=self.config.seed,
            dataloader_pin_memory=False,
            remove_unused_columns=False,
        )

        # Initialize trainer
        if self.config.use_custom_loss and self.custom_loss_fn is not None:
            trainer = CustomPIITrainer(
                model=self.model,
                args=training_args,
                train_dataset=train_dataset,
                eval_dataset=val_dataset,
                data_collator=data_collator,
                compute_metrics=self.compute_metrics,
                custom_loss_fn=self.custom_loss_fn,
            )
            logger.info("✅ Using CustomPIITrainer with masked loss")
        else:
            trainer = Trainer(
                model=self.model,
                args=training_args,
                train_dataset=train_dataset,
                eval_dataset=val_dataset,
                data_collator=data_collator,
                compute_metrics=self.compute_metrics,
            )
            logger.info("✅ Using standard Trainer")

        # Train
        logger.info("\n🏋️  Starting training...")
        logger.info("=" * 60)
        trainer.train()

        # Save
        trainer.save_model()
        self.tokenizer.save_pretrained(self.config.output_dir)
        logger.info(f"\n✅ Training completed. Model saved to {self.config.output_dir}")

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
            # Load the trained model
            self.model = AutoModelForTokenClassification.from_pretrained(self.config.output_dir)

            data_collator = DataCollatorForTokenClassification(
                tokenizer=self.tokenizer, padding=True
            )

            training_args = TrainingArguments(
                output_dir=self.config.output_dir,
                per_device_eval_batch_size=self.config.batch_size,
                report_to=None,
            )

            trainer = Trainer(
                model=self.model,
                args=training_args,
                eval_dataset=test_dataset,
                data_collator=data_collator,
                compute_metrics=self.compute_metrics,
            )

        results = trainer.evaluate()

        logger.info("\n📊 Evaluation Results:")
        for key, value in results.items():
            logger.info(f"  {key}: {value:.4f}")

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

        logger.info("\n💾 Copying model to Google Drive...")
        logger.info(f"   Source: {self.config.output_dir}")
        logger.info(f"   Target: {target_path}")

        shutil.copytree(self.config.output_dir, target_path)
        logger.info(f"✅ Model successfully saved to Google Drive at {target_path}")

        return target_path


# =============================================================================
# MAIN EXECUTION
# =============================================================================


def main(use_google_drive: bool = True, drive_folder: str = "MyDrive/pii_models"):
    """
    Main execution function.

    Args:
        use_google_drive: Whether to save model to Google Drive (Colab only)
        drive_folder: Target folder in Google Drive for saving the model
    """
    logger.info("=" * 60)
    logger.info("PII Detection Model Training - Word-based Preprocessing")
    logger.info("=" * 60)

    # Setup environment
    logger.info("\n1️⃣  Setting up environment...")
    EnvironmentSetup.disable_wandb()

    # Mount Google Drive if requested
    drive_mounted = False
    if use_google_drive:
        drive_mounted = EnvironmentSetup.mount_google_drive()

    EnvironmentSetup.check_gpu()

    # Load configuration
    logger.info("\n2️⃣  Loading configuration...")
    config = TrainingConfig()
    config.print_summary()

    # Prepare datasets
    logger.info("\n3️⃣  Preparing datasets...")
    dataset_processor = DatasetProcessor(config)
    train_dataset, val_dataset = dataset_processor.prepare_datasets()

    # Initialize trainer
    logger.info("\n4️⃣  Initializing trainer...")
    trainer = PIITrainer(config)
    mappings_path = Path(config.output_dir) / "label_mappings.json"
    trainer.load_label_mappings(mappings_path)
    trainer.initialize_model()

    # Train model
    logger.info("\n5️⃣  Training model...")
    start_time = time.time()
    trained_trainer = trainer.train(train_dataset, val_dataset)
    training_time = time.time() - start_time
    logger.info(f"\n⏱️  Training completed in {training_time / 60:.1f} minutes")

    # Evaluate model
    logger.info("\n6️⃣  Evaluating model...")
    results = trainer.evaluate(val_dataset, trained_trainer)

    # Save to Google Drive if mounted
    drive_path = None
    if use_google_drive and drive_mounted:
        logger.info("\n7️⃣  Saving to Google Drive...")
        try:
            drive_path = trainer.save_to_google_drive(drive_folder)
        except Exception as e:
            logger.warning(f"⚠️  Failed to save to Google Drive: {e}")
            logger.info(f"   Model is still available locally at: {config.output_dir}")

    # Final summary
    logger.info("\n" + "=" * 60)
    logger.info("🎉 TRAINING COMPLETE!")
    logger.info("=" * 60)
    logger.info(f"F1 Score: {results.get('eval_f1', 'N/A'):.4f}")
    logger.info(f"Model saved locally to: {config.output_dir}")
    if drive_path:
        logger.info(f"Model saved to Google Drive: {drive_path}")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
