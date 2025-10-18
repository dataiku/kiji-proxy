"""
PII Detection Model Evaluation Script

This script:
1. Downloads the most recent model from Google Drive
2. Loads the model and tokenizer
3. Runs inference on 10 test cases
4. Displays detected PII entities

Usage:
    # In Google Colab:
    python eval_model.py

    # With custom Google Drive folder:
    python eval_model.py --drive-folder "MyDrive/my_models"

    # Using local model (skip Google Drive):
    python eval_model.py --local-model "./pii_model"
"""

import argparse
import json
import logging
import re
import time
from datetime import datetime
from pathlib import Path

import torch
from transformers import AutoModelForTokenClassification, AutoTokenizer

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# =============================================================================
# GOOGLE DRIVE UTILITIES
# =============================================================================


class GoogleDriveManager:
    """Handles Google Drive mounting and model retrieval."""

    @staticmethod
    def mount_drive(mount_point: str = "/content/drive") -> bool:
        """
        Mount Google Drive in Colab environment.

        Args:
            mount_point: Path where Google Drive should be mounted

        Returns:
            True if mounted successfully, False otherwise
        """
        try:
            from google.colab import drive

            drive.mount(mount_point)
            logger.info(f"✅ Google Drive mounted at {mount_point}")
        except ImportError:
            logger.warning("⚠️  Not running in Google Colab - skipping Drive mount")
            return False
        except Exception:
            logger.exception("❌ Failed to mount Google Drive")
            return False
        else:
            return True

    @staticmethod
    def find_latest_model(drive_folder: str = "MyDrive/pii_models") -> str | None:
        """
        Find the most recent model in Google Drive folder.

        Args:
            drive_folder: Folder path in Google Drive (relative to mount point)

        Returns:
            Path to the most recent model, or None if not found
        """
        drive_path = Path(f"/content/drive/{drive_folder}")

        if not drive_path.exists():
            logger.error(f"❌ Google Drive folder not found: {drive_path}")
            return None

        # List all directories in the folder
        model_dirs = [
            d for d in drive_path.iterdir() if d.is_dir()
        ]

        if not model_dirs:
            logger.error(f"❌ No models found in {drive_path}")
            return None

        # Parse timestamps and find the most recent
        model_timestamps = []
        for model_dir in model_dirs:
            # Extract timestamp (format: modelname_YYYYMMDD_HHMMSS)
            match = re.search(r"_(\d{8}_\d{6})$", model_dir.name)
            if match:
                timestamp_str = match.group(1)
                try:
                    timestamp = datetime.strptime(timestamp_str, "%Y%m%d_%H%M%S")
                    model_timestamps.append((timestamp, model_dir.name))
                except ValueError:
                    continue

        if not model_timestamps:
            # If no timestamps found, use the first model
            logger.warning(f"⚠️  No timestamped models found, using: {model_dirs[0].name}")
            return str(drive_path / model_dirs[0].name)

        # Sort by timestamp and get the most recent
        model_timestamps.sort(reverse=True)
        latest_model = model_timestamps[0][1]
        latest_path = drive_path / latest_model

        logger.info(f"✅ Found latest model: {latest_model}")
        logger.info(f"   Path: {latest_path}")

        return str(latest_path)


# =============================================================================
# MODEL LOADER
# =============================================================================


class PIIModelLoader:
    """Loads and manages PII detection model."""

    def __init__(self, model_path: str):
        """
        Initialize model loader.

        Args:
            model_path: Path to the saved model directory
        """
        self.model_path = model_path
        self.model = None
        self.tokenizer = None
        self.label2id = None
        self.id2label = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def load_model(self):
        """Load model, tokenizer, and label mappings."""
        logger.info(f"\n📥 Loading model from: {self.model_path}")

        # Load label mappings
        mappings_path = Path(self.model_path) / "label_mappings.json"
        if mappings_path.exists():
            with mappings_path.open() as f:
                mappings = json.load(f)
            self.label2id = mappings["label2id"]
            self.id2label = {int(k): v for k, v in mappings["id2label"].items()}
            logger.info(f"✅ Loaded {len(self.label2id)} label mappings")
        else:
            logger.warning("⚠️  Label mappings not found, will use model's default labels")

        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
        logger.info("✅ Loaded tokenizer")

        # Load model
        self.model = AutoModelForTokenClassification.from_pretrained(self.model_path)
        self.model.to(self.device)
        self.model.eval()
        logger.info(f"✅ Loaded model on device: {self.device}")

        # If label mappings weren't loaded, get them from the model
        if self.id2label is None:
            self.id2label = self.model.config.id2label
            self.label2id = self.model.config.label2id

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

        # Run inference
        with torch.no_grad():
            outputs = self.model(**inputs)
            predictions = torch.argmax(outputs.logits, dim=-1)[0]

        # Convert predictions to labels
        tokens = self.tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])
        predicted_labels = [self.id2label[p.item()] for p in predictions]

        # Extract entities
        entities = []
        current_entity = None
        current_label = None
        current_start = None
        current_end = None

        for _idx, (token, label, offset) in enumerate(zip(tokens, predicted_labels, offset_mapping, strict=True)):
            # Skip special tokens
            if token in [
                self.tokenizer.cls_token,
                self.tokenizer.sep_token,
                self.tokenizer.pad_token,
            ]:
                continue

            # Check if this is a PII token
            if label.startswith("B-"):
                # Save previous entity if exists
                if current_entity is not None:
                    entity_text = text[current_start:current_end]
                    entities.append((entity_text, current_label, current_start, current_end))

                # Start new entity
                current_label = label[2:]  # Remove "B-" prefix
                current_start = offset[0].item()
                current_end = offset[1].item()
                current_entity = token

            elif label.startswith("I-") and current_entity is not None:
                # Continue current entity
                current_end = offset[1].item()

            elif current_entity is not None:  # "O" label or entity ended
                # Save previous entity if exists
                entity_text = text[current_start:current_end]
                entities.append((entity_text, current_label, current_start, current_end))
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
    "My name is John Smith and my email is john.smith@email.com",
    "Please contact Sarah Johnson at 555-123-4567 or sarah.j@company.org",
    "The patient's DOB is 03/15/1985 and SSN is 123-45-6789",
    "Credit card number: 4532-1234-5678-9010, exp: 12/25",
    "I live at 123 Main Street, Springfield, IL 62701",
    "Username: mike_wilson, Password: SecurePass123, Account: 9876543210",
    "Dr. Emily Chen can be reached at emily.chen@hospital.com or 555-987-6543",
    "Tax ID: 98-7654321, Driver's License: D1234567",
    "My colleague Alex Martinez lives at 456 Oak Avenue, Apt 7B, Boston, MA 02108",
    "Contact info - Name: Jennifer Lee, Tel: +1-555-246-8101, Email: j.lee@tech.com, ID: EMP001234",
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
        "--drive-folder",
        type=str,
        default="MyDrive/pii_models",
        help="Google Drive folder containing models (default: MyDrive/pii_models)",
    )
    parser.add_argument(
        "--local-model",
        type=str,
        default=None,
        help="Path to local model (skips Google Drive download)",
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
    model_path = None

    if args.local_model:
        # Use local model
        model_path = args.local_model
        logger.info(f"\n📁 Using local model: {model_path}")
    else:
        # Mount Google Drive and find latest model
        logger.info("\n1️⃣  Mounting Google Drive...")
        if GoogleDriveManager.mount_drive():
            logger.info("\n2️⃣  Finding latest model...")
            model_path = GoogleDriveManager.find_latest_model(args.drive_folder)
        else:
            logger.warning("\n⚠️  Google Drive not available. Trying local fallback...")
            # Try local fallback
            local_paths = ["./pii_model", "../pii_model", "../../pii_model"]
            for path in local_paths:
                if Path(path).exists():
                    model_path = path
                    logger.info(f"✅ Found local model: {model_path}")
                    break

    if model_path is None or not Path(model_path).exists():
        logger.error("\n❌ No model found! Please specify a valid model path.")
        logger.error("   Use --local-model <path> to specify a local model")
        return

    # Load model
    logger.info("\n3️⃣  Loading model...")
    loader = PIIModelLoader(model_path)
    loader.load_model()

    # Run inference on test cases
    logger.info(f"\n4️⃣  Running inference on {min(args.num_tests, len(TEST_CASES))} test cases...")

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
    logger.info(f"  Average entities per test: {total_entities / len(inference_times):.1f}")
    logger.info(f"{'=' * 80}\n")


if __name__ == "__main__":
    main()
