"""
Data preprocessing and dataset loading for PII detection training.

This module handles:
1. Loading training samples from Label Studio JSON format
2. Converting Label Studio annotations to training format (privacy_mask)
3. Tokenization and label alignment for PII detection
4. Creating HuggingFace datasets for training
"""

import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, ClassVar

from absl import logging
from datasets import Dataset
from transformers import AutoTokenizer

# Import label utilities
try:
    project_root = Path(__file__).parent.parent
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))
    from dataset.label_utils import LabelUtils
    from dataset.tokenization import TokenizationProcessor
except ImportError:
    from dataset.label_utils import LabelUtils
    from dataset.tokenization import TokenizationProcessor


# =============================================================================
# Label Management
# =============================================================================


class PIILabels:
    """Manages PII label definitions and mappings."""

    LABELS: ClassVar[list[str]] = LabelUtils.STANDARD_PII_LABELS

    @classmethod
    def create_label_mappings(cls) -> tuple[dict[str, int], dict[int, str], set]:
        """Create label to ID and ID to label mappings with BIO tagging."""
        label2id = {"O": 0}
        id2label = {0: "O"}

        for label in cls.LABELS:
            b_label = f"B-{label}"
            i_label = f"I-{label}"
            label2id[b_label] = len(label2id)
            label2id[i_label] = len(label2id)
            id2label[len(id2label)] = b_label
            id2label[len(id2label)] = i_label

        return label2id, id2label, set(cls.LABELS)

    @classmethod
    def save_mappings(
        cls, label2id: dict[str, int], id2label: dict[int, str], filepath: str
    ) -> None:
        """Save label mappings to JSON file."""
        mappings = {"label2id": label2id, "id2label": id2label}
        with Path(filepath).open("w") as f:
            json.dump(mappings, f, indent=2)
        logging.info(f"Label mappings saved to {filepath}")

    @classmethod
    def load_mappings(cls, filepath: str) -> tuple[dict[str, int], dict[int, str]]:
        """Load label mappings from JSON file."""
        with Path(filepath).open() as f:
            mappings = json.load(f)
        label2id = mappings["label2id"]
        id2label = {int(k): v for k, v in mappings["id2label"].items()}
        return label2id, id2label


# =============================================================================
# Label Studio Format Conversion
# =============================================================================


@dataclass
class Entity:
    """Represents an annotated entity from Label Studio."""

    id: str
    text: str
    label: str | None
    start: int
    end: int

    @property
    def is_pii(self) -> bool:
        """Check if this is a PII entity (not PRONOUN, REFERENCE, or unlabeled)."""
        return self.label is not None and self.label not in ("PRONOUN", "REFERENCE")


@dataclass
class TrainingSample:
    """Training sample with text and PII annotations."""

    text: str
    privacy_mask: list[dict[str, str]]
    language: str | None = None
    country: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary format."""
        return {
            "text": self.text,
            "privacy_mask": self.privacy_mask,
            "language": self.language,
            "country": self.country,
        }


class LabelStudioConverter:
    """Converts Label Studio format to training format."""

    @staticmethod
    def parse_result(result: list[dict]) -> dict[str, Entity]:
        """
        Parse Label Studio result into entities.

        Args:
            result: List of annotation items from Label Studio

        Returns:
            Dictionary of entities
        """
        entities: dict[str, Entity] = {}

        for idx, item in enumerate(result):
            if "value" in item:
                # Entity annotation - use item id if present, otherwise generate one
                entity_id = item.get("id", f"ent-auto-{idx}")
                value = item.get("value", {})
                labels = value.get("labels", [])

                entities[entity_id] = Entity(
                    id=entity_id,
                    text=value.get("text", ""),
                    label=labels[0] if labels else None,
                    start=value.get("start", 0),
                    end=value.get("end", 0),
                )

        return entities

    @staticmethod
    def build_privacy_mask(entities: dict[str, Entity]) -> list[dict[str, str]]:
        """
        Build privacy mask from PII entities.

        Args:
            entities: Dictionary of entity_id -> Entity

        Returns:
            List of privacy mask items with 'value' and 'label'
        """
        # Build privacy mask from PII entities
        privacy_mask = []
        seen_values = set()

        for entity in entities.values():
            if entity.is_pii and entity.text not in seen_values:
                privacy_mask.append(
                    {
                        "value": entity.text,
                        "label": entity.label,
                    }
                )
                seen_values.add(entity.text)

        return privacy_mask

    @classmethod
    def convert(cls, ls_sample: dict, file_name: str = "") -> TrainingSample | None:
        """
        Convert a Label Studio sample to training format.

        Args:
            ls_sample: Label Studio format sample
            file_name: Source file name for error messages

        Returns:
            TrainingSample or None if conversion fails
        """
        try:
            # Extract text
            text = ls_sample.get("data", {}).get("text", "")
            if not text:
                logging.debug(f"Sample missing text in file: {file_name}")
                return None

            # Get annotations or predictions
            result = None
            if ls_sample.get("annotations") and len(ls_sample["annotations"]) > 0:
                result = ls_sample["annotations"][0].get("result", [])
            elif ls_sample.get("predictions") and len(ls_sample["predictions"]) > 0:
                result = ls_sample["predictions"][0].get("result", [])

            if not result:
                logging.debug(f"Sample has no annotations/predictions: {file_name}")
                return None

            # Parse entities
            entities = cls.parse_result(result)

            # Build privacy mask
            privacy_mask = cls.build_privacy_mask(entities)

            return TrainingSample(
                text=text,
                privacy_mask=privacy_mask,
                language=ls_sample.get("data", {}).get("language"),
                country=ls_sample.get("data", {}).get("country"),
            )

        except Exception as e:
            logging.debug(f"Failed to convert sample {file_name}: {e}")
            return None


# =============================================================================
# Dataset Processing
# =============================================================================


class DatasetProcessor:
    """Handles dataset loading and processing from local JSON files."""

    def __init__(self, config):
        """
        Initialize dataset processor.

        Args:
            config: Training configuration with model_name, training_samples_dir, etc.
        """
        self.config = config
        self.tokenizer = AutoTokenizer.from_pretrained(config.model_name)
        self.label2id, self.id2label = LabelUtils.create_standard_label2id()
        self.tokenization_processor = TokenizationProcessor(
            self.tokenizer, self.label2id, self.id2label
        )

    def convert_labelstudio_to_training_format(
        self, ls_sample: dict, file_name: str
    ) -> dict | None:
        """
        Convert Label Studio format to training format.

        This is a wrapper around LabelStudioConverter for backward compatibility.
        """
        sample = LabelStudioConverter.convert(ls_sample, file_name)
        return sample.to_dict() if sample else None

    def load_training_samples(self) -> list[dict]:
        """
        Load training samples from local JSON files.

        Returns:
            List of training samples in dictionary format
        """
        samples_dir = Path(self.config.training_samples_dir)
        if not samples_dir.exists():
            raise ValueError(f"Training samples directory not found: {samples_dir}")

        json_files = list(samples_dir.glob("*.json"))
        logging.info(f"Loading training samples from {samples_dir}...")
        logging.info(f"Found {len(json_files)} JSON files")

        samples = []
        converted_count = 0
        skipped_count = 0

        for json_file in json_files:
            try:
                with json_file.open() as f:
                    ls_sample = json.load(f)

                converted = self.convert_labelstudio_to_training_format(
                    ls_sample, json_file.name
                )
                if converted:
                    samples.append(converted)
                    converted_count += 1
                else:
                    skipped_count += 1

            except json.JSONDecodeError as e:
                logging.warning(f"JSON decode error in {json_file.name}: {e}")
                skipped_count += 1
            except Exception as e:
                logging.warning(f"Error loading {json_file.name}: {e}")
                skipped_count += 1

        logging.info(f"Loaded {converted_count} training samples")
        if skipped_count > 0:
            logging.info(f"Skipped {skipped_count} files")

        if not samples:
            raise ValueError(
                f"No samples could be loaded from {len(json_files)} files. "
                "Check that files are in Label Studio format."
            )

        return samples

    def _tokenize_sample(self, sample: dict) -> dict:
        """
        Tokenize a single sample for training.

        Args:
            sample: Training sample with text, privacy_mask

        Returns:
            Tokenized sample with input_ids, attention_mask, labels
        """
        text = sample["text"]
        privacy_mask = sample["privacy_mask"]

        # Tokenize for PII detection
        pii_sample = self.tokenization_processor.create_pii_sample(text, privacy_mask)

        return {
            "input_ids": pii_sample["input_ids"],
            "attention_mask": pii_sample["attention_mask"],
            "labels": pii_sample["labels"],
        }

    def prepare_datasets(
        self, subsample_count: int = 0
    ) -> tuple[Dataset, Dataset, dict]:
        """
        Prepare training and validation datasets.

        Args:
            subsample_count: Limit to N samples (0 = use all)

        Returns:
            Tuple of (train_dataset, val_dataset, label_mappings)
        """
        # Load samples
        all_samples = self.load_training_samples()
        all_samples = [s for s in all_samples if s is not None]

        # Subsample if requested
        if subsample_count > 0 and len(all_samples) > subsample_count:
            logging.info(f"Subsampling from {len(all_samples)} to {subsample_count}")
            all_samples = all_samples[:subsample_count]

        if not all_samples:
            raise ValueError("No training samples found!")

        # Tokenize all samples
        logging.info("Tokenizing samples...")
        formatted_samples = []
        for i, sample in enumerate(all_samples):
            try:
                formatted_samples.append(self._tokenize_sample(sample))
                if (i + 1) % 500 == 0:
                    logging.info(f"  Tokenized {i + 1}/{len(all_samples)} samples")
            except Exception as e:
                logging.error(f"Failed to tokenize sample {i}: {e}")
                raise

        if not formatted_samples:
            raise ValueError("No samples could be tokenized!")

        # Split into train/val
        split_idx = int(len(formatted_samples) * (1 - self.config.eval_size_ratio))
        train_samples = formatted_samples[:split_idx]
        val_samples = formatted_samples[split_idx:]

        # Create datasets
        train_dataset = Dataset.from_list(train_samples)
        val_dataset = Dataset.from_list(val_samples)

        # Create label mappings
        mappings = {
            "label2id": self.label2id,
            "id2label": {str(k): v for k, v in self.id2label.items()},
        }

        # Save mappings
        mappings_path = Path(self.config.output_dir) / "label_mappings.json"
        mappings_path.parent.mkdir(parents=True, exist_ok=True)
        with mappings_path.open("w") as f:
            json.dump(mappings, f, indent=2)
        logging.info(f"Label mappings saved to {mappings_path}")

        # Log summary
        logging.info("Dataset Summary:")
        logging.info(f"  Training samples: {len(train_dataset)}")
        logging.info(f"  Validation samples: {len(val_dataset)}")
        logging.info(f"  PII labels: {len(self.label2id)}")

        return train_dataset, val_dataset, mappings
