"""Data preprocessing and dataset loading."""

import json
import sys
from pathlib import Path
from typing import ClassVar

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
        logging.info(f"✅ Label mappings saved to {filepath}")

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

        # Create label mappings for tokenization
        self.label2id, self.id2label = LabelUtils.create_standard_label2id()

        # Create tokenization processor
        self.tokenization_processor = TokenizationProcessor(
            self.tokenizer, self.label2id, self.id2label
        )

    def convert_labelstudio_to_training_format(
        self, ls_sample: dict, file_name: str
    ) -> dict | None:
        """
        Convert Label Studio format to training format.

        Args:
            ls_sample: Label Studio format sample with 'data', 'annotations', and/or 'predictions'
            file_name: Name of the file being processed (for error messages)

        Returns:
            Training format sample with 'text', 'privacy_mask', and 'coreferences'
            Returns None if the sample cannot be converted
        """
        try:
            # Extract text from data
            text = ls_sample.get("data", {}).get("text", "")
            if not text:
                logging.debug(f"Sample missing text in file: {file_name}")
                return None

            # Get annotations or predictions (prefer annotations if available)
            result = None
            if ls_sample.get("annotations") and len(ls_sample["annotations"]) > 0:
                result = ls_sample["annotations"][0].get("result", [])
            elif ls_sample.get("predictions") and len(ls_sample["predictions"]) > 0:
                result = ls_sample["predictions"][0].get("result", [])
            else:
                # No annotations or predictions
                return None

            # Skip if result is empty
            if not result:
                return None

            # Parse entities and relations from result
            entities = {}  # entity_id -> entity info
            relations = []  # list of relations

            for item in result:
                # Entity annotation (has "value" field)
                if "value" in item:
                    entity_id = item["id"]
                    value = item.get("value", {})
                    labels = value.get("labels", [])
                    entities[entity_id] = {
                        "text": value.get("text", ""),
                        "label": labels[0] if labels else None,
                        "start": value.get("start"),
                        "end": value.get("end"),
                    }
                # Relation annotation (has "from_id" field)
                elif "from_id" in item:
                    relations.append(
                        {
                            "from_id": item["from_id"],
                            "to_id": item["to_id"],
                            "type": item.get("type", "relation"),
                        }
                    )

            # Build privacy_mask from entities
            privacy_mask = []

            # Build coreferences from relations
            # Group entities by their target (to_id)
            entity_references = {}  # to_id -> list of from_ids
            for relation in relations:
                to_id = relation["to_id"]
                from_id = relation["from_id"]
                if to_id not in entity_references:
                    entity_references[to_id] = []
                entity_references[to_id].append(from_id)

            # Track which entities are part of coreference clusters
            processed_entities = set()
            coreferences = []
            cluster_id = 0  # Start cluster IDs at 0

            # Build coreference clusters
            for main_entity_id, referencing_ids in entity_references.items():
                if main_entity_id not in entities:
                    continue

                main_entity = entities[main_entity_id]

                # Add main entity to privacy_mask (skip if no label, e.g. coreference span markers)
                if main_entity_id not in processed_entities:
                    if main_entity["label"]:
                        privacy_mask.append(
                            {
                                "value": main_entity["text"],
                                "label": main_entity["label"],
                            }
                        )
                    processed_entities.add(main_entity_id)

                # Build coreference cluster with mentions
                mentions = [main_entity["text"]]
                for ref_id in referencing_ids:
                    if ref_id in entities:
                        mentions.append(entities[ref_id]["text"])
                        processed_entities.add(ref_id)

                # Add coreference cluster if there are multiple mentions
                if len(mentions) > 1:
                    coreferences.append(
                        {
                            "mentions": mentions,
                            "entity_type": main_entity["label"],
                            "cluster_id": cluster_id,
                        }
                    )
                    cluster_id += 1

            # Add remaining entities (not part of coreferences) to privacy_mask
            for entity_id, entity in entities.items():
                if entity_id not in processed_entities and entity["label"]:
                    privacy_mask.append(
                        {
                            "value": entity["text"],
                            "label": entity["label"],
                        }
                    )

            # Return converted sample
            return {
                "text": text,
                "privacy_mask": privacy_mask,
                "coreferences": coreferences,
                "language": ls_sample.get("data", {}).get("language"),
                "country": ls_sample.get("data", {}).get("country"),
            }

        except Exception as e:
            logging.debug(f"Failed to convert Label Studio sample in {file_name}: {e}")
            return None

    # Coreference marker labels used in Label Studio annotations.
    # These are not PII entities and should not trigger non-standard label filtering.
    _COREFERENCE_LABELS = {"PRONOUN", "REFERENCE", "MENTION", "pronoun", "reference"}

    def _has_non_standard_labels(self, sample: dict) -> bool:
        """
        Check if a converted sample contains non-standard NER labels.

        Samples with labels outside STANDARD_PII_LABELS would confuse the model
        during training and should be skipped entirely. Coreference marker labels
        (PRONOUN, REFERENCE, MENTION) are ignored since they are not NER entities.

        Args:
            sample: Converted training format sample with 'privacy_mask'

        Returns:
            True if any entity in privacy_mask has a non-standard label
        """
        standard_labels = set(LabelUtils.STANDARD_PII_LABELS)
        for entity in sample.get("privacy_mask", []):
            label = entity.get("label")
            if (
                label
                and label not in standard_labels
                and label not in self._COREFERENCE_LABELS
            ):
                return True
        return False

    def load_training_samples(self) -> list[dict]:
        """
        Load training samples from local JSON files.
        Supports the Label Studio format.

        Samples containing non-standard NER labels are skipped entirely
        to avoid confusing the model during training.

        Returns:
            List of training samples
        """
        samples_dir = Path(self.config.training_samples_dir)
        if not samples_dir.exists():
            raise ValueError(f"Training samples directory not found: {samples_dir}")

        samples = []
        json_files = list(samples_dir.glob("*.json"))

        logging.info(f"\n📥 Loading training samples from {samples_dir}...")
        logging.info(f"Found {len(json_files)} JSON files")

        # Track conversion statistics
        converted_count = 0
        skipped_count = 0
        non_standard_count = 0

        for json_file in json_files:
            try:
                with json_file.open() as f:
                    sample = json.load(f)

                # Convert to training format
                converted = self.convert_labelstudio_to_training_format(
                    sample, file_name=json_file.name
                )
                if converted is None:
                    skipped_count += 1
                elif self._has_non_standard_labels(converted):
                    non_standard_count += 1
                else:
                    samples.append(converted)
                    converted_count += 1

            except json.JSONDecodeError as e:
                logging.warning(f"⚠️  JSON decode error in {json_file.name}: {e}")
                skipped_count += 1
            except Exception as e:
                logging.warning(f"⚠️  Error loading {json_file.name}: {e}")
                skipped_count += 1

        # Print statistics
        logging.info(f"\n📊 Preprocessing Summary:")
        logging.info(f"  Files processed:              {len(json_files):,}")
        logging.info(f"  Available for training:        {converted_count:,}")
        if non_standard_count > 0:
            logging.info(f"  Skipped (non-standard labels): {non_standard_count:,}")
        if skipped_count > 0:
            logging.info(f"  Skipped (conversion errors):   {skipped_count:,}")

        if len(samples) == 0:
            raise ValueError(
                f"No samples could be loaded from {len(json_files)} files. "
                "Please check that files are in Label Studio format with 'data', 'text', "
                "and 'annotations' or 'predictions' fields."
            )

        return samples

    def prepare_datasets(
        self, subsample_count: int = 0
    ) -> tuple[Dataset, Dataset, dict, dict]:
        """
        Prepare training and validation datasets from local JSON files.
        Tokenization is performed on-the-fly during dataset preparation.

        Args:
            subsample_count: Limit to N samples (0 = use all)

        Returns:
            Tuple of (train_dataset, val_dataset, label_mappings, coref_mappings)
        """
        # Load all samples (raw text, privacy_mask, coreferences)
        all_samples = self.load_training_samples()

        # Filter out None samples
        all_samples = [s for s in all_samples if s is not None]

        # Subsample if requested
        if subsample_count > 0 and len(all_samples) > subsample_count:
            logging.info(
                f"📉 Subsampling from {len(all_samples)} to {subsample_count} samples"
            )
            all_samples = all_samples[:subsample_count]

        if len(all_samples) == 0:
            raise ValueError("No training samples found!")

        # New code path: tokenize on-the-fly
        logging.info("🔄 Tokenizing samples on-the-fly during dataset preparation...")

        # Determine max coreference cluster ID from all samples
        max_coref_id = 0
        for sample in all_samples:
            coreferences = sample.get("coreferences", [])
            for coref in coreferences:
                cluster_id = coref.get("cluster_id", 0)
                max_coref_id = max(max_coref_id, cluster_id)
        num_coref_labels = max_coref_id + 2  # +1 for NO_COREF (0), +1 for 0-indexed

        # Prepare dataset format with on-the-fly tokenization
        def format_sample(sample: dict) -> dict:
            """Format a single sample for training by tokenizing on-the-fly."""
            text = sample["text"]
            privacy_mask = sample["privacy_mask"]
            coreferences = sample.get("coreferences", [])

            # Tokenize PII sample
            pii_sample = self.tokenization_processor.create_pii_sample(
                text, privacy_mask
            )

            # Tokenize coreference sample
            coreference_sample = self.tokenization_processor.create_coreference_sample(
                text, coreferences
            )

            # Validate that tokenization is consistent
            if coreference_sample["input_ids"] != pii_sample["input_ids"]:
                raise ValueError(
                    "Input IDs do not match between PII and coreference samples"
                )
            if coreference_sample["attention_mask"] != pii_sample["attention_mask"]:
                raise ValueError(
                    "Attention masks do not match between PII and coreference samples"
                )

            return {
                "input_ids": pii_sample["input_ids"],
                "attention_mask": pii_sample["attention_mask"],
                "pii_labels": pii_sample["labels"],
                "coref_labels": coreference_sample["coreference_labels"],
            }

        # Tokenize all samples
        logging.info("📝 Tokenizing samples...")
        formatted_samples = []
        for i, sample in enumerate(all_samples):
            try:
                formatted_samples.append(format_sample(sample))
                if (i + 1) % 100 == 0:
                    logging.info(f"  Tokenized {i + 1}/{len(all_samples)} samples...")
            except Exception as e:
                logging.error(f"❌ Failed to tokenize sample {i}: {e}")
                raise  # Re-raise the original exception with full traceback

        if len(formatted_samples) == 0:
            raise ValueError("No samples could be tokenized!")

        # Split into train and validation
        split_idx = int(len(formatted_samples) * (1 - self.config.eval_size_ratio))
        train_samples = formatted_samples[:split_idx]
        val_samples = formatted_samples[split_idx:]

        # Create HuggingFace datasets
        train_dataset = Dataset.from_list(train_samples)
        val_dataset = Dataset.from_list(val_samples)

        # Prepare label mappings
        pii_label2id = self.label2id
        pii_id2label = self.id2label

        # Create coreference label mappings
        coref_id2label = {0: "NO_COREF"}
        for i in range(1, num_coref_labels):
            coref_id2label[i] = f"CLUSTER_{i - 1}"

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
        logging.info(f"✅ Label mappings saved to {mappings_path}")

        logging.info("\n📊 Dataset Summary:")
        logging.info(f"  Training samples: {len(train_dataset)}")
        logging.info(f"  Validation samples: {len(val_dataset)}")
        logging.info(f"  PII labels: {len(pii_label2id)}")
        logging.info(f"  Co-reference labels: {num_coref_labels}")

        return (
            train_dataset,
            val_dataset,
            mappings,
            {"num_coref_labels": num_coref_labels},
        )
