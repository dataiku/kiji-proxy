"""Data preprocessing and dataset loading."""
import json
import logging
from pathlib import Path
from typing import ClassVar

from datasets import Dataset
from transformers import AutoTokenizer

# Import label utilities
try:
    import sys
    project_root = Path(__file__).parent.parent
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))
    from dataset.label_utils import LabelUtils
except ImportError:
    from dataset.label_utils import LabelUtils

logger = logging.getLogger(__name__)


class PIILabels:
    """Manages PII label definitions and mappings."""

    # Standard PII labels - use labels from LabelUtils
    LABELS: ClassVar[list[str]] = LabelUtils.STANDARD_PII_LABELS

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
    def save_mappings(
        cls, label2id: dict[str, int], id2label: dict[int, str], filepath: str
    ):
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


class DatasetProcessor:
    """Handles dataset loading and processing from local JSON files."""

    def __init__(self, config):
        """
        Initialize dataset processor.

        Args:
            config: Training configuration
        """
        self.config = config
        self.tokenizer = AutoTokenizer.from_pretrained(config.model_name)

    def load_training_samples(self) -> list[dict]:
        """
        Load training samples from local JSON files.

        Returns:
            List of training samples
        """
        samples_dir = Path(self.config.training_samples_dir)
        if not samples_dir.exists():
            raise ValueError(f"Training samples directory not found: {samples_dir}")

        samples = []
        json_files = list(samples_dir.glob("*.json"))

        logger.info(f"\n📥 Loading training samples from {samples_dir}...")
        logger.info(f"Found {len(json_files)} JSON files")

        for json_file in json_files:
            try:
                with json_file.open() as f:
                    sample = json.load(f)
                    samples.append(sample)
            except Exception as e:
                logger.warning(f"⚠️  Failed to load {json_file}: {e}")

        logger.info(f"✅ Loaded {len(samples)} training samples")
        return samples

    def prepare_datasets(self) -> tuple[Dataset, Dataset, dict, dict]:
        """
        Prepare training and validation datasets from local JSON files.

        Returns:
            Tuple of (train_dataset, val_dataset, label_mappings, coref_mappings)
        """
        # Load all samples
        all_samples = self.load_training_samples()

        if len(all_samples) == 0:
            raise ValueError("No training samples found!")

        # Extract label mappings from first sample (they should be consistent)
        first_sample = all_samples[0]
        pii_label2id = first_sample["pii_sample"]["label2id"]
        pii_id2label = {int(k): v for k, v in first_sample["pii_sample"]["id2label"].items()}
        coref_id2label = {int(k): v for k, v in first_sample["coreference_sample"]["cluster_id2label"].items()}

        # Determine number of coreference classes (max cluster ID + 1 for NO_COREF)
        # Find max cluster ID from all samples to get accurate count
        max_coref_id = 0
        for sample in all_samples:
            coref_labels = sample["coreference_sample"]["coreference_labels"]
            max_in_sample = max((label for label in coref_labels if label >= 0), default=0)
            max_coref_id = max(max_coref_id, max_in_sample)
        num_coref_labels = max_coref_id + 1

        # Prepare dataset format
        def format_sample(sample: dict) -> dict:
            """Format a single sample for training."""
            return {
                "input_ids": sample["input_ids"],
                "attention_mask": sample["attention_mask"],
                "pii_labels": sample["pii_sample"]["labels"],
                "coref_labels": sample["coreference_sample"]["coreference_labels"],
            }

        formatted_samples = [format_sample(s) for s in all_samples]

        # Split into train and validation
        split_idx = int(len(formatted_samples) * (1 - self.config.eval_size_ratio))
        train_samples = formatted_samples[:split_idx]
        val_samples = formatted_samples[split_idx:]

        # Create HuggingFace datasets
        train_dataset = Dataset.from_list(train_samples)
        val_dataset = Dataset.from_list(val_samples)

        # Save label mappings
        mappings_path = Path(self.config.output_dir) / "label_mappings.json"
        mappings = {
            "pii": {
                "label2id": pii_label2id,
                "id2label": {str(k): v for k, v in pii_id2label.items()},
            },
            "coref": {
                "id2label": {str(k): v for k, v in coref_id2label.items()},
            },
        }
        with mappings_path.open("w") as f:
            json.dump(mappings, f, indent=2)
        logger.info(f"✅ Label mappings saved to {mappings_path}")

        logger.info("\n📊 Dataset Summary:")
        logger.info(f"  Training samples: {len(train_dataset)}")
        logger.info(f"  Validation samples: {len(val_dataset)}")
        logger.info(f"  PII labels: {len(pii_label2id)}")
        logger.info(f"  Co-reference labels: {num_coref_labels}")

        return train_dataset, val_dataset, mappings, {"num_coref_labels": num_coref_labels}

