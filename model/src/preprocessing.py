"""
Data preprocessing and dataset loading for PII detection training.

This module handles:
1. Loading training samples from Label Studio JSON format
2. Converting Label Studio annotations to training format (privacy_mask, coreferences)
3. Tokenization and label alignment for PII detection
4. Tokenization and label alignment for coreference detection
5. Creating HuggingFace datasets for training
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
class Relation:
    """Represents a coreference relation from Label Studio."""

    from_id: str
    to_id: str


@dataclass
class TrainingSample:
    """Training sample with text, PII annotations, and coreferences."""

    text: str
    privacy_mask: list[dict[str, str]]
    coreferences: list[dict[str, Any]]
    language: str | None = None
    country: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary format."""
        return {
            "text": self.text,
            "privacy_mask": self.privacy_mask,
            "coreferences": self.coreferences,
            "language": self.language,
            "country": self.country,
        }


class LabelStudioConverter:
    """Converts Label Studio format to training format."""

    @staticmethod
    def parse_result(result: list[dict]) -> tuple[dict[str, Entity], list[Relation]]:
        """
        Parse Label Studio result into entities and relations.

        Args:
            result: List of annotation items from Label Studio

        Returns:
            Tuple of (entities dict, relations list)
        """
        entities: dict[str, Entity] = {}
        relations: list[Relation] = []

        for item in result:
            if "value" in item:
                # Entity annotation
                entity_id = item["id"]
                value = item.get("value", {})
                labels = value.get("labels", [])

                entities[entity_id] = Entity(
                    id=entity_id,
                    text=value.get("text", ""),
                    label=labels[0] if labels else None,
                    start=value.get("start", 0),
                    end=value.get("end", 0),
                )
            elif "from_id" in item:
                # Relation annotation
                relations.append(
                    Relation(
                        from_id=item["from_id"],
                        to_id=item["to_id"],
                    )
                )

        return entities, relations

    @staticmethod
    def build_coreference_clusters(
        entities: dict[str, Entity], relations: list[Relation]
    ) -> list[dict[str, Any]]:
        """
        Build coreference clusters from entities and relations.

        Args:
            entities: Dictionary of entity_id -> Entity
            relations: List of coreference relations

        Returns:
            List of coreference cluster dictionaries
        """
        # Group relations by target entity (to_id)
        # Each to_id becomes the "main" entity of a cluster
        clusters_by_target: dict[str, list[str]] = {}
        for rel in relations:
            # Skip self-references
            if rel.from_id == rel.to_id:
                continue
            if rel.to_id not in clusters_by_target:
                clusters_by_target[rel.to_id] = []
            clusters_by_target[rel.to_id].append(rel.from_id)

        # Build cluster objects
        coreferences = []
        cluster_id = 0

        # Get PII entities for label inference
        pii_entities = {eid: e for eid, e in entities.items() if e.is_pii}

        for main_id, referring_ids in clusters_by_target.items():
            main_entity = entities.get(main_id)
            if not main_entity:
                continue

            # Determine entity type for the cluster
            entity_type = LabelStudioConverter._infer_entity_type(
                main_entity, referring_ids, entities, pii_entities
            )

            # Skip clusters without a valid PII type
            if entity_type is None:
                continue

            # Collect all mentions
            mentions = [main_entity.text]
            for ref_id in referring_ids:
                ref_entity = entities.get(ref_id)
                if ref_entity:
                    mentions.append(ref_entity.text)

            # Only create cluster if there are multiple mentions
            if len(mentions) > 1:
                coreferences.append(
                    {
                        "mentions": mentions,
                        "entity_type": entity_type,
                        "cluster_id": cluster_id,
                    }
                )
                cluster_id += 1

        return coreferences

    @staticmethod
    def _infer_entity_type(
        main_entity: Entity,
        referring_ids: list[str],
        all_entities: dict[str, Entity],
        pii_entities: dict[str, Entity],
    ) -> str | None:
        """
        Infer the entity type for a coreference cluster.

        Tries multiple strategies:
        1. Use main entity's label if it's a PII label
        2. Check if any referring entity has a PII label
        3. Check if main entity text contains any PII entity's text
        """
        # Strategy 1: Main entity has PII label
        if main_entity.is_pii:
            return main_entity.label

        # Strategy 2: Check referring entities
        for ref_id in referring_ids:
            ref_entity = all_entities.get(ref_id)
            if ref_entity and ref_entity.is_pii:
                return ref_entity.label

        # Strategy 3: Text containment (e.g., "Amina Al-Farouq" contains "Amina")
        main_text_lower = main_entity.text.lower()
        for pii_entity in pii_entities.values():
            if pii_entity.text.lower() in main_text_lower:
                return pii_entity.label

        return None

    @staticmethod
    def build_privacy_mask(
        entities: dict[str, Entity], coreferences: list[dict[str, Any]]
    ) -> list[dict[str, str]]:
        """
        Build privacy mask from PII entities.

        Args:
            entities: Dictionary of entity_id -> Entity
            coreferences: List of coreference clusters (used to avoid duplicates)

        Returns:
            List of privacy mask items with 'value' and 'label'
        """
        # Collect texts already included in coreference clusters
        coref_texts = set()
        for coref in coreferences:
            coref_texts.update(coref["mentions"])

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

            # Parse entities and relations
            entities, relations = cls.parse_result(result)

            # Build coreference clusters
            coreferences = cls.build_coreference_clusters(entities, relations)

            # Build privacy mask
            privacy_mask = cls.build_privacy_mask(entities, coreferences)

            return TrainingSample(
                text=text,
                privacy_mask=privacy_mask,
                coreferences=coreferences,
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
            sample: Training sample with text, privacy_mask, coreferences

        Returns:
            Tokenized sample with input_ids, attention_mask, pii_labels, coref_labels
        """
        text = sample["text"]
        privacy_mask = sample["privacy_mask"]
        coreferences = sample.get("coreferences", [])

        # Tokenize for PII detection
        pii_sample = self.tokenization_processor.create_pii_sample(text, privacy_mask)

        # Tokenize for coreference detection
        coref_sample = self.tokenization_processor.create_coreference_sample(
            text, coreferences
        )

        # Validate consistency
        if pii_sample["input_ids"] != coref_sample["input_ids"]:
            raise ValueError("Input IDs mismatch between PII and coref tokenization")

        return {
            "input_ids": pii_sample["input_ids"],
            "attention_mask": pii_sample["attention_mask"],
            "pii_labels": pii_sample["labels"],
            "coref_labels": coref_sample["coreference_labels"],
        }

    def prepare_datasets(
        self, subsample_count: int = 0
    ) -> tuple[Dataset, Dataset, dict, dict]:
        """
        Prepare training and validation datasets.

        Args:
            subsample_count: Limit to N samples (0 = use all)

        Returns:
            Tuple of (train_dataset, val_dataset, label_mappings, coref_info)
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

        # Determine max coreference cluster ID
        max_coref_id = 0
        for sample in all_samples:
            for coref in sample.get("coreferences", []):
                max_coref_id = max(max_coref_id, coref.get("cluster_id", 0))
        num_coref_labels = max_coref_id + 2  # +1 for NO_COREF, +1 for 0-indexing

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
        coref_id2label = {0: "NO_COREF"}
        for i in range(1, num_coref_labels):
            coref_id2label[i] = f"CLUSTER_{i - 1}"

        mappings = {
            "pii": {
                "label2id": self.label2id,
                "id2label": {str(k): v for k, v in self.id2label.items()},
            },
            "coref": {
                "id2label": {str(k): v for k, v in coref_id2label.items()},
            },
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
        logging.info(f"  Coreference labels: {num_coref_labels}")

        return (
            train_dataset,
            val_dataset,
            mappings,
            {"num_coref_labels": num_coref_labels},
        )
