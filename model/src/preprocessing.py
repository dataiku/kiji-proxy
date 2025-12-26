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
    from dataset.tokenization import TokenizationProcessor
except ImportError:
    from dataset.label_utils import LabelUtils
    from dataset.tokenization import TokenizationProcessor

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

        # Create label mappings for tokenization
        self.label2id, self.id2label = LabelUtils.create_standard_label2id()

        # Create tokenization processor
        self.tokenization_processor = TokenizationProcessor(
            self.tokenizer, self.label2id, self.id2label
        )

    def convert_labelstudio_to_training_format(self, ls_sample: dict) -> dict | None:
        """
        Convert Label Studio format to training format.

        Args:
            ls_sample: Label Studio format sample with 'data', 'annotations', and/or 'predictions'

        Returns:
            Training format sample with 'text', 'privacy_mask', and 'coreferences'
            Returns None if the sample cannot be converted
        """
        try:
            # Extract text from data
            text = ls_sample.get("data", {}).get("text", "")
            if not text:
                logger.warning("⚠️  Sample missing text")
                import pprint
                print(f"file name: {ls_sample.get('file_name', 'unknown')}")
                print("ls_sample: type", type(ls_sample))
                print("ls_sample: keys", ls_sample.keys())
                print("ls_sample: data", ls_sample.get("data", {}))
                print("ls_sample: data: text", ls_sample.get("data", {}).get("text", ""))
                print("ls_sample: data: language", ls_sample.get("data", {}).get("language", ""))
                print("ls_sample: data: country", ls_sample.get("data", {}).get("country", ""))
                print("ls_sample: annotations", ls_sample.get("annotations", []))
                print("ls_sample: predictions", ls_sample.get("predictions", []))
                print("-" * 100)
                pprint.pprint(ls_sample)
                print("-" * 100)
                exit()
                return None

            # Get annotations or predictions (prefer annotations if available)
            result = None
            if ls_sample.get("annotations") and len(ls_sample["annotations"]) > 0:
                result = ls_sample["annotations"][0].get("result", [])
                logger.debug("Using annotations")
            elif ls_sample.get("predictions") and len(ls_sample["predictions"]) > 0:
                result = ls_sample["predictions"][0].get("result", [])
                logger.debug("Using predictions")
            else:
                # No annotations or predictions - return None without warning
                # (the calling code will track this statistic)
                return None

            # Parse entities and relations from result
            entities = {}  # entity_id -> entity info
            relations = []  # list of relations

            for item in result:
                # Entity annotation (has "value" field)
                if "value" in item:
                    entity_id = item["id"]
                    value = item.get("value", {})
                    entities[entity_id] = {
                        "text": value.get("text", ""),
                        "label": value.get("labels", [None])[0],
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

                # Add main entity to privacy_mask
                if main_entity_id not in processed_entities:
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
                if entity_id not in processed_entities:
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
            logger.warning(f"⚠️  Failed to convert Label Studio sample: {e}")
            return None

    def load_training_samples(self) -> list[dict]:
        """
        Load training samples from local JSON files.
        Supports both Label Studio format and legacy training format.

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

        # Track conversion statistics
        labelstudio_converted = 0
        labelstudio_failed = 0
        legacy_format = 0
        unknown_format = 0
        no_annotations = 0

        for json_file in json_files:
            try:
                with json_file.open() as f:
                    sample = json.load(f)

                    # Detect format: Label Studio has 'data', 'annotations', or 'predictions'
                    if "data" in sample and (
                        "annotations" in sample or "predictions" in sample
                    ):
                        # Label Studio format - convert it
                        converted = self.convert_labelstudio_to_training_format(sample)
                        if converted:
                            samples.append(converted)
                            labelstudio_converted += 1
                        else:
                            labelstudio_failed += 1
                            # Check if it's because of missing annotations/predictions
                            has_annotations = sample.get("annotations") and len(sample["annotations"]) > 0
                            has_predictions = sample.get("predictions") and len(sample["predictions"]) > 0
                            if not has_annotations and not has_predictions:
                                no_annotations += 1
                    elif "text" in sample and "privacy_mask" in sample:
                        # Legacy training format - use as-is
                        samples.append(sample)
                        legacy_format += 1
                    else:
                        logger.warning(f"⚠️  Unknown format in {json_file.name}")
                        unknown_format += 1

            except Exception as e:
                logger.warning(f"⚠️  Failed to load {json_file}: {e}")

        # Print detailed statistics
        logger.info("\n📊 Loading Statistics:")
        logger.info(f"  Total files: {len(json_files)}")
        logger.info(f"  Label Studio converted: {labelstudio_converted}")
        logger.info(f"  Label Studio failed: {labelstudio_failed}")
        if no_annotations > 0:
            logger.info(f"    - No annotations/predictions: {no_annotations}")
        logger.info(f"  Legacy format: {legacy_format}")
        logger.info(f"  Unknown format: {unknown_format}")
        logger.info(f"✅ Loaded {len(samples)} training samples")

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
            logger.info(
                f"📉 Subsampling from {len(all_samples)} to {subsample_count} samples"
            )
            all_samples = all_samples[:subsample_count]

        if len(all_samples) == 0:
            raise ValueError("No training samples found!")

        # convert labelstudio format to training format
        all_samples = [self.convert_labelstudio_to_training_format(sample) for sample in all_samples]
        # filter out None samples
        all_samples = [sample for sample in all_samples if sample is not None]

        # New code path: tokenize on-the-fly
        logger.info("🔄 Tokenizing samples on-the-fly during dataset preparation...")

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
        logger.info("📝 Tokenizing samples...")
        formatted_samples = []
        for i, sample in enumerate(all_samples):
            try:
                formatted_samples.append(format_sample(sample))
                if (i + 1) % 100 == 0:
                    logger.info(f"  Tokenized {i + 1}/{len(all_samples)} samples...")
            except Exception as e:
                raise ValueError(f"Failed to tokenize sample {i}: {e}")
                print (f"file: {json_file.name}")
                logger.warning(f"⚠️  Failed to tokenize sample {i}: {e}")
                continue

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
        logger.info(f"✅ Label mappings saved to {mappings_path}")

        logger.info("\n📊 Dataset Summary:")
        logger.info(f"  Training samples: {len(train_dataset)}")
        logger.info(f"  Validation samples: {len(val_dataset)}")
        logger.info(f"  PII labels: {len(pii_label2id)}")
        logger.info(f"  Co-reference labels: {num_coref_labels}")

        return (
            train_dataset,
            val_dataset,
            mappings,
            {"num_coref_labels": num_coref_labels},
        )
