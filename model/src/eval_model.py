"""
PII Detection Model Evaluation Script

This script:
1. Loads a trained multi-task PII detection model (local)
2. Runs inference on test cases
3. Displays detected PII entities and co-reference clusters

Usage:
    # Using local model:
    python eval_model.py --local-model "./model/trained"

    # With custom number of test cases:
    python eval_model.py --local-model "./model/trained" --num-tests 5

The script evaluates both PII detection and co-reference resolution capabilities
of the multi-task model, showing detected entities and how they are clustered.
"""

import argparse
import json
import sys
import time
from pathlib import Path

import torch
from absl import logging
from safetensors import safe_open
from transformers import AutoTokenizer

# Add project root to path for imports
project_root = Path(__file__).parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

try:
    from model.model import PIIDetectionModel
except ImportError:
    from .model import PIIDetectionModel

try:
    from model.src.checkpoint_utils import (
        load_compatible_state_dict,
        normalize_state_dict_keys,
    )
except ImportError:
    try:
        from src.checkpoint_utils import (
            load_compatible_state_dict,
            normalize_state_dict_keys,
        )
    except ImportError:
        from .checkpoint_utils import (
            load_compatible_state_dict,
            normalize_state_dict_keys,
        )


# =============================================================================
# DEVICE UTILITIES
# =============================================================================


def get_device():
    """Get the best available device (MPS > CUDA > CPU)."""
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    elif torch.cuda.is_available():
        return torch.device("cuda")
    else:
        return torch.device("cpu")


# =============================================================================
# MODEL LOADER
# =============================================================================


class PIIModelLoader:
    """Loads and manages multi-task PII detection model."""

    def __init__(self, model_path: str):
        """
        Initialize model loader.

        Args:
            model_path: Path to the saved model directory
        """
        self.model_path = model_path
        self.model = None
        self.tokenizer = None
        self.pii_label2id = None
        self.pii_id2label = None
        self.coref_id2label = None
        self.device = get_device()

    def load_model(self):
        """Load multi-task model, tokenizer, and label mappings."""
        logging.info(f"\n📥 Loading model from: {self.model_path}")

        # Load label mappings
        mappings_path = Path(self.model_path) / "label_mappings.json"
        if not mappings_path.exists():
            raise FileNotFoundError(
                f"Label mappings not found at {mappings_path}. "
                "Make sure the model was trained and saved correctly."
            )

        with mappings_path.open() as f:
            mappings = json.load(f)

        # Load PII label mappings
        self.pii_label2id = mappings["pii"]["label2id"]
        self.pii_id2label = {int(k): v for k, v in mappings["pii"]["id2label"].items()}
        logging.info(f"✅ Loaded {len(self.pii_label2id)} PII label mappings")

        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
        logging.info("✅ Loaded tokenizer")

        # Load model config to get base model name
        config_path = Path(self.model_path) / "config.json"
        model_type_defaults = {
            "bert": "bert-base-cased",
            "distilbert": "distilbert-base-cased",
            "roberta": "roberta-base",
            "deberta-v2": "microsoft/deberta-v3-base",
        }
        if config_path.exists():
            with config_path.open() as f:
                model_config = json.load(f)
            base_model_name = model_config.get("_name_or_path", "")
            if not base_model_name or base_model_name in model_type_defaults:
                model_type = model_config.get("model_type", "distilbert")
                base_model_name = model_type_defaults.get(
                    model_type, "microsoft/deberta-v3-base"
                )
        else:
            base_model_name = "microsoft/deberta-v3-base"
            logging.warning(
                "⚠️  config.json not found, using default: microsoft/deberta-v3-base"
            )

        # Determine number of labels
        num_pii_labels = len(self.pii_label2id)

        # Find model weights file
        model_weights_path = Path(self.model_path) / "pytorch_model.bin"
        if not model_weights_path.exists():
            # Try alternative naming
            model_weights_path = Path(self.model_path) / "model.safetensors"
            if not model_weights_path.exists():
                # Try to find any .bin file
                bin_files = list(Path(self.model_path).glob("*.bin"))
                if bin_files:
                    model_weights_path = bin_files[0]
                    logging.info(f"   Found weights: {model_weights_path.name}")

        state_dict = None
        if model_weights_path.exists():
            logging.info(f"📦 Loading weights from: {model_weights_path.name}")

            # Handle safetensors files
            if model_weights_path.suffix == ".safetensors":
                state_dict = {}
                with safe_open(model_weights_path, framework="pt", device="cpu") as f:
                    for key in f.keys():
                        state_dict[key] = f.get_tensor(key)
            else:
                # Handle .bin files - use weights_only=False for PyTorch 2.6+
                state_dict = torch.load(
                    model_weights_path, map_location="cpu", weights_only=False
                )

            state_dict = normalize_state_dict_keys(state_dict)

        logging.info("📋 Model configuration:")
        logging.info(f"   Base model: {base_model_name}")
        logging.info(f"   PII labels: {num_pii_labels}")

        # Load PII detection model
        self.model = PIIDetectionModel(
            model_name=base_model_name,
            num_pii_labels=num_pii_labels,
            id2label_pii=self.pii_id2label,
        )

        # Load model weights into the model
        if state_dict is not None:
            # Move tensors to the correct device
            if model_weights_path.suffix != ".safetensors":
                # For .bin files, we already loaded to CPU, now move to device
                state_dict = {k: v.to(self.device) for k, v in state_dict.items()}
            else:
                # For safetensors, reload to device
                state_dict = {}
                with safe_open(
                    model_weights_path, framework="pt", device=str(self.device)
                ) as f:
                    for key in f.keys():
                        state_dict[key] = f.get_tensor(key)
                state_dict = normalize_state_dict_keys(state_dict)

            load_info = load_compatible_state_dict(
                self.model,
                state_dict,
                source=str(model_weights_path),
            )
            if load_info.unexpected_keys:
                logging.warning(
                    "⚠️  Ignoring unexpected checkpoint keys: %s",
                    load_info.unexpected_keys[:20],
                )
            logging.info("✅ Model weights loaded")
        else:
            logging.warning(
                "⚠️  Model weights not found, using randomly initialized model"
            )

        self.model.to(self.device)
        self.model.eval()

        device_name = (
            "MPS (Apple Silicon)" if self.device.type == "mps" else str(self.device)
        )
        logging.info(f"✅ Loaded model on device: {device_name}")

    def predict(
        self, text: str
    ) -> tuple[
        list[tuple[str, str, int, int]], dict[int, list[tuple[str, int, int]]], float
    ]:
        """
        Run inference on input text and measure inference time.

        Args:
            text: Input text to analyze

        Returns:
            Tuple of (entities, coref_clusters, inference_time_ms)
            - entities: List of tuples (entity_text, label, start_pos, end_pos)
            - coref_clusters: Dict mapping cluster_id to list of (text, start_pos, end_pos)
            - inference_time_ms: Time taken for inference in milliseconds
        """
        if self.model is None or self.tokenizer is None:
            raise ValueError("Model not loaded. Call load_model() first.")

        start_time = time.perf_counter()

        # Tokenize input
        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=512,
            return_offsets_mapping=True,
        )

        offset_mapping = inputs.pop("offset_mapping")[0]
        inputs.pop("token_type_ids", None)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        # Run inference
        with torch.no_grad():
            outputs = self.model(**inputs)
            if hasattr(self.model, "decode"):
                pii_prediction_ids = self.model.decode(
                    outputs["pii_logits"], inputs["attention_mask"]
                )[0]
            else:
                pii_prediction_ids = (
                    torch.argmax(outputs["pii_logits"], dim=-1)[0].cpu().tolist()
                )

        # Convert predictions to labels
        tokens = self.tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])
        if len(pii_prediction_ids) < len(tokens):
            pii_prediction_ids = pii_prediction_ids + [0] * (
                len(tokens) - len(pii_prediction_ids)
            )
        pii_prediction_ids = pii_prediction_ids[: len(tokens)]
        predicted_labels = [
            self.pii_id2label.get(int(label_id), "O") for label_id in pii_prediction_ids
        ]

        # Extract entities
        entities = []
        current_entity = None
        current_label = None
        current_start = None
        current_end = None

        # Helper function to strip trailing punctuation from entity text
        def strip_trailing_punctuation(text: str) -> tuple[str, int]:
            """Strip trailing punctuation and return cleaned text and number of chars stripped."""
            punctuation = ",.;:!?)]}"
            stripped = text.rstrip(punctuation)
            chars_stripped = len(text) - len(stripped)
            return stripped, chars_stripped

        for _idx, (token, label, offset) in enumerate(
            zip(tokens, predicted_labels, offset_mapping, strict=True)
        ):
            # Skip special tokens
            if (
                token
                in [
                    self.tokenizer.cls_token,
                    self.tokenizer.sep_token,
                    self.tokenizer.pad_token,
                ]
                or label == "IGNORE"
            ):
                continue

            # Check if this is a PII token
            if label.startswith("B-"):
                # Save previous entity if exists
                if current_entity is not None:
                    entity_text = text[current_start:current_end]
                    # Strip trailing punctuation
                    entity_text, chars_stripped = strip_trailing_punctuation(
                        entity_text
                    )
                    if entity_text:  # Only add if there's text left after stripping
                        entities.append(
                            (
                                entity_text,
                                current_label,
                                current_start,
                                current_end - chars_stripped,
                            )
                        )

                current_label = label[2:]  # Remove "B-" prefix
                current_start = offset[0].item()
                current_end = offset[1].item()
                current_entity = token

            elif label.startswith("I-") and current_entity is not None:
                # Continue current entity if the label matches. Punctuation inside
                # the entity span, such as dots in emails/URLs, is preserved here
                # and only trailing sentence punctuation is stripped at finalization.
                if current_label == label[2:]:
                    current_end = offset[1].item()
                else:
                    # Different label - save previous and start new
                    entity_text = text[current_start:current_end]
                    # Strip trailing punctuation
                    entity_text, chars_stripped = strip_trailing_punctuation(
                        entity_text
                    )
                    if entity_text:  # Only add if there's text left after stripping
                        entities.append(
                            (
                                entity_text,
                                current_label,
                                current_start,
                                current_end - chars_stripped,
                            )
                        )

                    current_label = label[2:]
                    current_start = offset[0].item()
                    current_end = offset[1].item()
                    current_entity = token

            elif current_entity is not None:  # "O" label or entity ended
                # Save previous entity if exists
                entity_text = text[current_start:current_end]
                # Strip trailing punctuation
                entity_text, chars_stripped = strip_trailing_punctuation(entity_text)
                if entity_text:  # Only add if there's text left after stripping
                    entities.append(
                        (
                            entity_text,
                            current_label,
                            current_start,
                            current_end - chars_stripped,
                        )
                    )
                current_entity = None
                current_label = None

        # Don't forget the last entity
        if current_entity is not None:
            entity_text = text[current_start:current_end]
            # Strip trailing punctuation
            entity_text, chars_stripped = strip_trailing_punctuation(entity_text)
            if entity_text:  # Only add if there's text left after stripping
                entities.append(
                    (
                        entity_text,
                        current_label,
                        current_start,
                        current_end - chars_stripped,
                    )
                )

        end_time = time.perf_counter()
        inference_time_ms = (end_time - start_time) * 1000

        return entities, {}, inference_time_ms


# =============================================================================
# TEST CASES
# =============================================================================

TEST_CASES = [
    "My name is John Smith and my email is john.smith@email.com. I was born on March 15, 1985.",
    "Please contact Sarah Johnson at 555-123-4567 or sarah.j@company.org. She lives in New York.",
    "The patient's date of birth is 03/15/1985 and their social security number is 123-45-6789.",
    "I live at 123 Main Street, Springfield, IL 62701. My phone number is 217-555-1234.",
    "Dr. Emily Chen can be reached at emily.chen@hospital.com or 555-987-6543. Her office is at 789 Medical Center Drive.",
    "My colleague Alex Martinez lives at 456 Oak Avenue, Apt 7B, Boston, MA 02108. You can email him at alex.m@company.com.",
    "Contact info - Name: Jennifer Lee, Tel: +1-555-246-8101, Email: j.lee@tech.com, Employee ID: EMP001234.",
    "Fatima Khaled resides at 2114 Cedar Crescent in Marseille, France. Her ID card number is XA1890274.",
    "The customer's driver license ID is F23098719 and their zip code is 13008. They moved there last year.",
    "Robert Williams was born on 1980-05-20. His email address is robert.williams@example.org and he can be reached at 415-555-0199.",
]


# =============================================================================
# MAIN EXECUTION
# =============================================================================


def print_results(
    text: str,
    entities: list[tuple[str, str, int, int]],
    coref_clusters: dict[int, list[tuple[str, int, int]]],
    case_num: int,
    inference_time_ms: float,
    coref_id2label: dict[int, str] | None = None,
):
    """
    Print inference results in a formatted way.

    Args:
        text: Original input text
        entities: List of detected entities
        coref_clusters: Dict mapping cluster_id to list of (text, start_pos, end_pos)
        case_num: Test case number
        inference_time_ms: Inference time in milliseconds
        coref_id2label: Optional mapping from cluster ID to label name
    """
    logging.info(f"\n{'=' * 80}")
    logging.info(f"Test Case {case_num}")
    logging.info(f"{'=' * 80}")
    logging.info(f"Text: {text}")
    logging.info(f"Inference Time: {inference_time_ms:.2f} ms")

    logging.info("\n🔍 Detected PII Entities:")
    if entities:
        for entity_text, label, start, end in entities:
            logging.info(f"  • [{label}] '{entity_text}' (position {start}-{end})")
    else:
        logging.info("  (No PII entities detected)")

    logging.info("\n🔗 Co-reference Clusters:")
    if coref_clusters:
        # Sort clusters by ID for consistent output
        for cluster_id in sorted(coref_clusters.keys()):
            if coref_id2label and cluster_id in coref_id2label:
                cluster_label = coref_id2label[cluster_id]
            else:
                cluster_label = f"CLUSTER_{cluster_id}"
            spans = coref_clusters[cluster_id]
            logging.info(f"  • {cluster_label} ({len(spans)} mention(s)):")
            for token_text, start, end in spans:
                logging.info(f"      - '{token_text}' (position {start}-{end})")
    else:
        logging.info("  (No co-reference clusters detected)")


def main():
    """Main execution function."""
    parser = argparse.ArgumentParser(description="Evaluate PII detection model")
    parser.add_argument(
        "--local-model",
        type=str,
        default="./model/trained",
        help="Path to local model directory (default: ./model/trained)",
    )
    parser.add_argument(
        "--num-tests",
        type=int,
        default=10,
        help="Number of test cases to run (default: 10)",
    )

    args = parser.parse_args()

    logging.info("=" * 80)
    logging.info("PII Detection Model Evaluation")
    logging.info("=" * 80)

    # Determine model path
    model_path = args.local_model

    # Try to find model if path doesn't exist
    if not Path(model_path).exists():
        logging.warning(f"⚠️  Model path not found: {model_path}")
        # Try common locations
        local_paths = ["./model/trained", "../model/trained", "model/trained"]
        for path in local_paths:
            if Path(path).exists():
                model_path = path
                logging.info(f"✅ Found model at: {model_path}")
                break

    if not Path(model_path).exists():
        logging.error("\n❌ No model found! Please specify a valid model path.")
        logging.error(f"   Searched: {args.local_model}")
        logging.error("   Use --local-model <path> to specify a local model")
        return

    logging.info(f"\n📁 Using model: {model_path}")

    # Check device availability
    device = get_device()
    device_name = "MPS (Apple Silicon)" if device.type == "mps" else str(device)
    logging.info(f"🖥️  Device: {device_name}")

    # Load model
    logging.info("\n📥 Loading model...")
    loader = PIIModelLoader(model_path)
    loader.load_model()

    # Run inference on test cases
    logging.info(
        f"\n🚀 Running inference on {min(args.num_tests, len(TEST_CASES))} test cases..."
    )

    inference_times = []
    total_entities = 0
    total_clusters = 0

    for i, test_text in enumerate(TEST_CASES[: args.num_tests], 1):
        entities, coref_clusters, inference_time_ms = loader.predict(test_text)
        inference_times.append(inference_time_ms)
        total_entities += len(entities)
        total_clusters += len(coref_clusters)
        print_results(
            test_text,
            entities,
            coref_clusters,
            i,
            inference_time_ms,
            loader.coref_id2label,
        )

    # Calculate statistics
    avg_time = sum(inference_times) / len(inference_times) if inference_times else 0
    min_time = min(inference_times) if inference_times else 0
    max_time = max(inference_times) if inference_times else 0
    total_time = sum(inference_times)

    # Summary
    logging.info(f"\n{'=' * 80}")
    logging.info("✅ Evaluation Complete!")
    logging.info(f"{'=' * 80}")
    logging.info(f"Model: {model_path}")
    logging.info(f"Device: {loader.device}")
    logging.info(f"Test cases processed: {min(args.num_tests, len(TEST_CASES))}")
    logging.info("\n📊 Inference Time Statistics:")
    logging.info(f"  Total time: {total_time:.2f} ms ({total_time / 1000:.3f} seconds)")
    logging.info(f"  Average time per test: {avg_time:.2f} ms")
    logging.info(f"  Min time: {min_time:.2f} ms")
    logging.info(f"  Max time: {max_time:.2f} ms")
    logging.info(f"  Throughput: {1000 / avg_time:.2f} texts/second")
    logging.info("\n📈 Detection Statistics:")
    logging.info(f"  Total PII entities detected: {total_entities}")
    logging.info(
        f"  Average entities per test: {total_entities / len(inference_times):.1f}"
    )
    logging.info(f"  Total co-reference clusters detected: {total_clusters}")
    logging.info(
        f"  Average clusters per test: {total_clusters / len(inference_times):.1f}"
    )
    logging.info(f"{'=' * 80}\n")


if __name__ == "__main__":
    main()
