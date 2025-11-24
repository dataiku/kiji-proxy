"""
PII Detection Model Evaluation Script

This script:
1. Loads a trained multi-task PII detection model (local)
2. Runs inference on test cases
3. Displays detected PII entities and co-reference clusters

Usage:
    # Using local model:
    python eval_model.py --local-model "./pii_model"

    # With custom number of test cases:
    python eval_model.py --local-model "./pii_model" --num-tests 5
"""

import argparse
import json
import logging
import sys
import time
from pathlib import Path

import torch
from transformers import AutoModel, AutoTokenizer

# Add project root to path for imports
project_root = Path(__file__).parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

try:
    from model.model import MultiTaskPIIDetectionModel
except ImportError:
    from model import MultiTaskPIIDetectionModel

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# =============================================================================
# DEVICE UTILITIES
# =============================================================================


def get_device():
    """Get the best available device (MPS > CUDA > CPU)."""
    if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
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
        logger.info(f"\n📥 Loading model from: {self.model_path}")

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
        self.pii_id2label = {
            int(k): v for k, v in mappings["pii"]["id2label"].items()
        }
        logger.info(f"✅ Loaded {len(self.pii_label2id)} PII label mappings")

        # Load co-reference label mappings
        if "coref" in mappings:
            self.coref_id2label = {
                int(k): v for k, v in mappings["coref"]["id2label"].items()
            }
            logger.info(f"✅ Loaded {len(self.coref_id2label)} co-reference label mappings")

        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
        logger.info("✅ Loaded tokenizer")

        # Load model config to get base model name
        config_path = Path(self.model_path) / "config.json"
        if config_path.exists():
            with config_path.open() as f:
                model_config = json.load(f)
            # Try to get base model name from config
            base_model_name = (
                model_config.get("_name_or_path")
                or model_config.get("model_type", "distilbert")
            )
            # Convert model_type to full model name if needed
            if base_model_name == "distilbert":
                base_model_name = "distilbert-base-cased"
        else:
            base_model_name = "distilbert-base-cased"
            logger.warning("⚠️  config.json not found, using default: distilbert-base-cased")

        # Determine number of labels
        num_pii_labels = len(self.pii_label2id)
        num_coref_labels = len(self.coref_id2label) if self.coref_id2label else 2

        logger.info(f"📋 Model configuration:")
        logger.info(f"   Base model: {base_model_name}")
        logger.info(f"   PII labels: {num_pii_labels}")
        logger.info(f"   Co-reference labels: {num_coref_labels}")

        # Load multi-task model
        self.model = MultiTaskPIIDetectionModel(
            model_name=base_model_name,
            num_pii_labels=num_pii_labels,
            num_coref_labels=num_coref_labels,
            id2label_pii=self.pii_id2label,
            id2label_coref=self.coref_id2label or {0: "NO_COREF", 1: "CLUSTER_0"},
        )

        # Load model weights
        model_weights_path = Path(self.model_path) / "pytorch_model.bin"
        if not model_weights_path.exists():
            # Try alternative naming
            model_weights_path = Path(self.model_path) / "model.safetensors"
            if not model_weights_path.exists():
                # Try to find any .bin file
                bin_files = list(Path(self.model_path).glob("*.bin"))
                if bin_files:
                    model_weights_path = bin_files[0]
                    logger.info(f"   Found weights: {model_weights_path.name}")

        if model_weights_path.exists():
            logger.info(f"📦 Loading weights from: {model_weights_path.name}")
            state_dict = torch.load(model_weights_path, map_location=self.device)
            # Handle state dict that might have 'model.' prefix
            if any(k.startswith("model.") for k in state_dict.keys()):
                state_dict = {
                    k.replace("model.", ""): v
                    for k, v in state_dict.items()
                    if k.startswith("model.")
                }
            self.model.load_state_dict(state_dict, strict=False)
            logger.info("✅ Model weights loaded")
        else:
            logger.warning("⚠️  Model weights not found, using randomly initialized model")

        self.model.to(self.device)
        self.model.eval()
        
        device_name = "MPS (Apple Silicon)" if self.device.type == "mps" else str(self.device)
        logger.info(f"✅ Loaded model on device: {device_name}")

    def predict(self, text: str) -> tuple[list[tuple[str, str, int, int]], float]:
        """
        Run inference on input text and measure inference time.

        Args:
            text: Input text to analyze

        Returns:
            Tuple of (entities, inference_time_ms)
            - entities: List of tuples (entity_text, label, start_pos, end_pos)
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
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        # Run inference with multi-task model
        with torch.no_grad():
            outputs = self.model(**inputs)
            # Get PII predictions (we focus on PII detection for this evaluation)
            pii_predictions = torch.argmax(outputs["pii_logits"], dim=-1)[0]

        # Convert predictions to labels
        tokens = self.tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])
        predicted_labels = [
            self.pii_id2label.get(p.item(), "O") for p in pii_predictions
        ]

        # Extract entities
        entities = []
        current_entity = None
        current_label = None
        current_start = None
        current_end = None

        for _idx, (token, label, offset) in enumerate(
            zip(tokens, predicted_labels, offset_mapping, strict=True)
        ):
            # Skip special tokens
            if token in [
                self.tokenizer.cls_token,
                self.tokenizer.sep_token,
                self.tokenizer.pad_token,
            ] or label == "IGNORE":
                continue

            # Check if this is a PII token
            if label.startswith("B-"):
                # Save previous entity if exists
                if current_entity is not None:
                    entity_text = text[current_start:current_end]
                    entities.append(
                        (entity_text, current_label, current_start, current_end)
                    )

                # Start new entity
                current_label = label[2:]  # Remove "B-" prefix
                current_start = offset[0].item()
                current_end = offset[1].item()
                current_entity = token

            elif label.startswith("I-") and current_entity is not None:
                # Continue current entity (only if same label)
                if current_label == label[2:]:  # Check label matches
                    current_end = offset[1].item()
                else:
                    # Different label - save previous and start new
                    entity_text = text[current_start:current_end]
                    entities.append(
                        (entity_text, current_label, current_start, current_end)
                    )
                    current_label = label[2:]
                    current_start = offset[0].item()
                    current_end = offset[1].item()

            elif current_entity is not None:  # "O" label or entity ended
                # Save previous entity if exists
                entity_text = text[current_start:current_end]
                entities.append(
                    (entity_text, current_label, current_start, current_end)
                )
                current_entity = None
                current_label = None

        # Don't forget the last entity
        if current_entity is not None:
            entity_text = text[current_start:current_end]
            entities.append((entity_text, current_label, current_start, current_end))

        end_time = time.perf_counter()
        inference_time_ms = (end_time - start_time) * 1000

        return entities, inference_time_ms


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
    case_num: int,
    inference_time_ms: float,
):
    """
    Print inference results in a formatted way.

    Args:
        text: Original input text
        entities: List of detected entities
        case_num: Test case number
        inference_time_ms: Inference time in milliseconds
    """
    logger.info(f"\n{'=' * 80}")
    logger.info(f"Test Case {case_num}")
    logger.info(f"{'=' * 80}")
    logger.info(f"Text: {text}")
    logger.info(f"Inference Time: {inference_time_ms:.2f} ms")
    logger.info("\nDetected PII Entities:")

    if entities:
        for entity_text, label, start, end in entities:
            logger.info(f"  • [{label}] '{entity_text}' (position {start}-{end})")
    else:
        logger.info("  (No PII entities detected)")


def main():
    """Main execution function."""
    parser = argparse.ArgumentParser(description="Evaluate PII detection model")
    parser.add_argument(
        "--local-model",
        type=str,
        default="./pii_model",
        help="Path to local model directory (default: ./pii_model)",
    )
    parser.add_argument(
        "--num-tests",
        type=int,
        default=10,
        help="Number of test cases to run (default: 10)",
    )

    args = parser.parse_args()

    logger.info("=" * 80)
    logger.info("PII Detection Model Evaluation")
    logger.info("=" * 80)

    # Determine model path
    model_path = args.local_model

    # Try to find model if path doesn't exist
    if not Path(model_path).exists():
        logger.warning(f"⚠️  Model path not found: {model_path}")
        # Try common locations
        local_paths = ["./pii_model", "../pii_model", "model/pii_model"]
        for path in local_paths:
            if Path(path).exists():
                model_path = path
                logger.info(f"✅ Found model at: {model_path}")
                break

    if not Path(model_path).exists():
        logger.error("\n❌ No model found! Please specify a valid model path.")
        logger.error(f"   Searched: {args.local_model}")
        logger.error("   Use --local-model <path> to specify a local model")
        return

    logger.info(f"\n📁 Using model: {model_path}")

    # Check device availability
    device = get_device()
    device_name = "MPS (Apple Silicon)" if device.type == "mps" else str(device)
    logger.info(f"🖥️  Device: {device_name}")

    # Load model
    logger.info("\n📥 Loading model...")
    loader = PIIModelLoader(model_path)
    loader.load_model()

    # Run inference on test cases
    logger.info(
        f"\n🚀 Running inference on {min(args.num_tests, len(TEST_CASES))} test cases..."
    )

    inference_times = []
    total_entities = 0

    for i, test_text in enumerate(TEST_CASES[: args.num_tests], 1):
        entities, inference_time_ms = loader.predict(test_text)
        inference_times.append(inference_time_ms)
        total_entities += len(entities)
        print_results(test_text, entities, i, inference_time_ms)

    # Calculate statistics
    avg_time = sum(inference_times) / len(inference_times) if inference_times else 0
    min_time = min(inference_times) if inference_times else 0
    max_time = max(inference_times) if inference_times else 0
    total_time = sum(inference_times)

    # Summary
    logger.info(f"\n{'=' * 80}")
    logger.info("✅ Evaluation Complete!")
    logger.info(f"{'=' * 80}")
    logger.info(f"Model: {model_path}")
    logger.info(f"Device: {loader.device}")
    logger.info(f"Test cases processed: {min(args.num_tests, len(TEST_CASES))}")
    logger.info("\n📊 Inference Time Statistics:")
    logger.info(f"  Total time: {total_time:.2f} ms ({total_time / 1000:.3f} seconds)")
    logger.info(f"  Average time per test: {avg_time:.2f} ms")
    logger.info(f"  Min time: {min_time:.2f} ms")
    logger.info(f"  Max time: {max_time:.2f} ms")
    logger.info(f"  Throughput: {1000 / avg_time:.2f} texts/second")
    logger.info("\n📈 Detection Statistics:")
    logger.info(f"  Total PII entities detected: {total_entities}")
    logger.info(
        f"  Average entities per test: {total_entities / len(inference_times):.1f}"
    )
    logger.info(f"{'=' * 80}\n")


if __name__ == "__main__":
    main()
