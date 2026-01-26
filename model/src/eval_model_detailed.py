"""
Detailed PII Detection Model Evaluation Script

This script provides detailed, token-level outputs from the model including:
1. Token-by-token predictions with confidence scores
2. Top-k predictions per token
3. Raw logits and probabilities
4. Detailed entity extraction breakdown
5. Co-reference prediction details

Usage:
    # Using local model:
    python eval_model_detailed.py --local-model "./model/trained"

    # With custom number of test cases:
    python eval_model_detailed.py --local-model "./model/trained" --num-tests 5

    # Show top-k predictions per token:
    python eval_model_detailed.py --local-model "./model/trained" --top-k 3

    # Show raw logits:
    python eval_model_detailed.py --local-model "./model/trained" --show-logits
"""

import argparse
import json
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from absl import logging
from safetensors import safe_open
from transformers import AutoTokenizer

# Add project root to path for imports
project_root = Path(__file__).parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

try:
    from model.src.model import MultiTaskPIIConfig, MultiTaskPIIDetectionModel
except ImportError:
    try:
        from src.model import MultiTaskPIIConfig, MultiTaskPIIDetectionModel
    except ImportError:
        from model import MultiTaskPIIConfig, MultiTaskPIIDetectionModel


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


class DetailedPIIModelLoader:
    """Loads and manages multi-task PII detection model with detailed output capabilities."""

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

        # Load co-reference label mappings
        if "coref" in mappings:
            self.coref_id2label = {
                int(k): v for k, v in mappings["coref"]["id2label"].items()
            }
            logging.info(
                f"✅ Loaded {len(self.coref_id2label)} co-reference label mappings"
            )

        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
        logging.info("✅ Loaded tokenizer")

        # Load model config to get base model name
        config_path = Path(self.model_path) / "config.json"
        if config_path.exists():
            with config_path.open() as f:
                model_config = json.load(f)
            # Try to get base model name from config
            base_model_name = model_config.get("_name_or_path") or model_config.get(
                "model_type", "modernbert"
            )
            # Convert model_type to full model name if needed
            if base_model_name == "modernbert":
                base_model_name = "answerdotai/ModernBERT-base"
        else:
            base_model_name = "answerdotai/ModernBERT-base"
            logging.warning(
                "⚠️  config.json not found, using default: answerdotai/ModernBERT-base"
            )

        # Determine number of labels
        num_pii_labels = len(self.pii_label2id)

        # Determine num_coref_labels from mappings (use max ID + 1 if available)
        if self.coref_id2label:
            num_coref_labels = max(self.coref_id2label.keys()) + 1
        else:
            num_coref_labels = 2

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

        # Load model weights first to determine correct num_coref_labels
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

            # Handle state dict that might have 'model.' prefix
            if any(k.startswith("model.") for k in state_dict.keys()):
                state_dict = {
                    k.replace("model.", ""): v
                    for k, v in state_dict.items()
                    if k.startswith("model.")
                }

            # Infer num_coref_labels from model weights
            for key in state_dict.keys():
                if "coref_classifier.weight" in key:
                    num_coref_labels = state_dict[key].shape[0]
                    logging.info(
                        f"   Detected {num_coref_labels} co-reference labels from model weights"
                    )
                    break

        logging.info("📋 Model configuration:")
        logging.info(f"   Base model: {base_model_name}")
        logging.info(f"   PII labels: {num_pii_labels}")
        logging.info(f"   Co-reference labels: {num_coref_labels}")

        # Create config and load multi-task model
        config = MultiTaskPIIConfig(
            base_model_name=base_model_name,
            num_pii_labels=num_pii_labels,
            num_coref_labels=num_coref_labels,
            id2label_pii=self.pii_id2label,
            id2label_coref=self.coref_id2label or {0: "NO_COREF", 1: "CLUSTER_0"},
        )
        self.model = MultiTaskPIIDetectionModel(config)

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
                # Handle state dict that might have 'model.' prefix
                if any(k.startswith("model.") for k in state_dict.keys()):
                    state_dict = {
                        k.replace("model.", ""): v
                        for k, v in state_dict.items()
                        if k.startswith("model.")
                    }

            self.model.load_state_dict(state_dict, strict=False)
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

    def predict_detailed(
        self, text: str, top_k: int = 3, show_logits: bool = False
    ) -> dict:
        """
        Run inference on input text and return detailed outputs.

        Args:
            text: Input text to analyze
            top_k: Number of top predictions to show per token
            show_logits: Whether to show raw logits

        Returns:
            Dictionary containing:
            - tokens: List of token strings
            - pii_predictions: List of predicted PII labels
            - pii_probabilities: List of probability distributions
            - pii_top_k: List of top-k predictions per token
            - coref_predictions: List of predicted co-reference IDs
            - coref_probabilities: List of probability distributions
            - coref_top_k: List of top-k predictions per token
            - pii_logits: Raw logits for PII (if show_logits=True)
            - coref_logits: Raw logits for co-reference (if show_logits=True)
            - offset_mapping: Token to character position mapping
            - inference_time_ms: Time taken for inference
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
            # Get PII logits and predictions
            pii_logits = outputs["pii_logits"][0]  # [seq_len, num_labels]
            pii_predictions = torch.argmax(pii_logits, dim=-1)  # [seq_len]
            pii_probs = F.softmax(pii_logits, dim=-1)  # [seq_len, num_labels]

            # Get co-reference logits and predictions
            coref_logits = outputs["coref_logits"][0]  # [seq_len, num_coref_labels]
            coref_predictions = torch.argmax(coref_logits, dim=-1)  # [seq_len]
            coref_probs = F.softmax(coref_logits, dim=-1)  # [seq_len, num_coref_labels]

        # Convert to lists
        tokens = self.tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])
        pii_pred_labels = [
            self.pii_id2label.get(p.item(), "O") for p in pii_predictions
        ]
        pii_pred_ids = [p.item() for p in pii_predictions]
        coref_pred_ids = [p.item() for p in coref_predictions]

        # Get top-k predictions for PII
        pii_top_k_list = []
        for i in range(len(tokens)):
            top_k_probs, top_k_indices = torch.topk(
                pii_probs[i], k=min(top_k, len(self.pii_id2label))
            )
            top_k_items = [
                {
                    "label": self.pii_id2label.get(idx.item(), "UNKNOWN"),
                    "label_id": idx.item(),
                    "probability": prob.item(),
                }
                for prob, idx in zip(top_k_probs, top_k_indices, strict=True)
            ]
            pii_top_k_list.append(top_k_items)

        # Get top-k predictions for co-reference
        coref_top_k_list = []
        for i in range(len(tokens)):
            top_k_probs, top_k_indices = torch.topk(
                coref_probs[i], k=min(top_k, len(self.coref_id2label))
            )
            top_k_items = [
                {
                    "label": self.coref_id2label.get(
                        idx.item(), f"CLUSTER_{idx.item()}"
                    ),
                    "label_id": idx.item(),
                    "probability": prob.item(),
                }
                for prob, idx in zip(top_k_probs, top_k_indices, strict=True)
            ]
            coref_top_k_list.append(top_k_items)

        # Convert probabilities to lists
        pii_prob_list = [probs.cpu().tolist() for probs in pii_probs]
        coref_prob_list = [probs.cpu().tolist() for probs in coref_probs]

        end_time = time.perf_counter()
        inference_time_ms = (end_time - start_time) * 1000

        result = {
            "tokens": tokens,
            "pii_predictions": pii_pred_labels,
            "pii_pred_ids": pii_pred_ids,
            "pii_probabilities": pii_prob_list,
            "pii_top_k": pii_top_k_list,
            "coref_predictions": coref_pred_ids,
            "coref_probabilities": coref_prob_list,
            "coref_top_k": coref_top_k_list,
            "offset_mapping": offset_mapping.cpu().tolist(),
            "inference_time_ms": inference_time_ms,
        }

        if show_logits:
            result["pii_logits"] = pii_logits.cpu().tolist()
            result["coref_logits"] = coref_logits.cpu().tolist()

        return result


# =============================================================================
# TEST CASES
# =============================================================================

TEST_CASES = [
    "My name is John Smith, and my email is john.smith@email.com. I was born on March 15, 1985.",
    # "Please contact Sarah Johnson at 555-123-4567 or sarah.j@company.org. She lives in New York.",
    # "The patient's date of birth is 03/15/1985 and their social security number is 123-45-6789.",
    # "I live at 123 Main Street, Springfield, IL 62701. My phone number is 217-555-1234.",
    # "Dr. Emily Chen can be reached at emily.chen@hospital.com or 555-987-6543. Her office is at 789 Medical Center Drive.",
]


# =============================================================================
# DETAILED OUTPUT FORMATTING
# =============================================================================


def print_detailed_results(
    text: str,
    detailed_output: dict,
    case_num: int,
    top_k: int = 3,
    show_logits: bool = False,
    coref_id2label: dict[int, str] | None = None,
    pii_id2label: dict[int, str] | None = None,
):
    """
    Print detailed inference results.

    Args:
        text: Original input text
        detailed_output: Dictionary from predict_detailed()
        case_num: Test case number
        top_k: Number of top predictions to show
        show_logits: Whether to show raw logits
        coref_id2label: Optional mapping from cluster ID to label name
    """
    tokens = detailed_output["tokens"]
    pii_preds = detailed_output["pii_predictions"]
    pii_top_k = detailed_output["pii_top_k"]
    coref_preds = detailed_output["coref_predictions"]
    coref_top_k = detailed_output["coref_top_k"]
    inference_time = detailed_output["inference_time_ms"]

    logging.info(f"\n{'=' * 80}")
    logging.info(f"Test Case {case_num} - Detailed Output")
    logging.info(f"{'=' * 80}")
    logging.info(f"Input Text: {text}")
    logging.info(f"Inference Time: {inference_time:.2f} ms")
    logging.info(f"Total Tokens: {len(tokens)}")

    # Token-by-token breakdown
    logging.info(f"\n{'=' * 80}")
    logging.info("Token-by-Token Predictions")
    logging.info(f"{'=' * 80}")
    logging.info(
        f"{'Token':<20} {'PII Label':<20} {'PII Conf':<10} {'Coref':<15} {'Coref Conf':<10}"
    )
    logging.info("-" * 80)

    for i, (token, pii_label, coref_id) in enumerate(
        zip(tokens, pii_preds, coref_preds, strict=True)
    ):
        # Skip special tokens in main display
        if token in ["[CLS]", "[SEP]", "[PAD]"]:
            continue

        # Get confidence for predicted label
        pii_conf = detailed_output["pii_probabilities"][i][
            detailed_output["pii_pred_ids"][i]
        ]
        coref_conf = detailed_output["coref_probabilities"][i][coref_id]

        # Get coref label
        if coref_id2label and coref_id in coref_id2label:
            coref_label = coref_id2label[coref_id]
        else:
            coref_label = f"CLUSTER_{coref_id}" if coref_id > 0 else "NO_COREF"

        # Truncate token if too long
        token_display = token[:18] + ".." if len(token) > 20 else token

        logging.info(
            f"{token_display:<20} {pii_label:<20} {pii_conf:.4f}     {coref_label:<15} {coref_conf:.4f}"
        )

    # Top-k predictions per token
    logging.info(f"\n{'=' * 80}")
    logging.info(f"Top-{top_k} PII Predictions Per Token")
    logging.info(f"{'=' * 80}")

    for i, (token, top_k_items) in enumerate(zip(tokens, pii_top_k, strict=True)):
        # Skip special tokens
        if token in ["[CLS]", "[SEP]", "[PAD]"]:
            continue

        logging.info(f"\nToken {i}: '{token}'")
        for rank, item in enumerate(top_k_items, 1):
            logging.info(
                f"  {rank}. {item['label']:<20} (ID: {item['label_id']:3d}) "
                f"Probability: {item['probability']:.4f}"
            )

    # Co-reference top-k
    logging.info(f"\n{'=' * 80}")
    logging.info(f"Top-{top_k} Co-reference Predictions Per Token")
    logging.info(f"{'=' * 80}")

    for i, (token, top_k_items) in enumerate(zip(tokens, coref_top_k, strict=True)):
        # Skip special tokens and NO_COREF predictions
        if token in ["[CLS]", "[SEP]", "[PAD]"]:
            continue

        # Only show if there are interesting predictions
        if any(item["label_id"] > 0 for item in top_k_items):
            logging.info(f"\nToken {i}: '{token}'")
            for rank, item in enumerate(top_k_items, 1):
                logging.info(
                    f"  {rank}. {item['label']:<15} (ID: {item['label_id']:3d}) "
                    f"Probability: {item['probability']:.4f}"
                )

    # Show raw logits if requested
    if show_logits:
        logging.info(f"\n{'=' * 80}")
        logging.info("Raw Logits (First 10 tokens, first 10 labels)")
        logging.info(f"{'=' * 80}")

        pii_logits = detailed_output.get("pii_logits", [])
        coref_logits = detailed_output.get("coref_logits", [])

        if pii_logits and pii_id2label:
            logging.info("\nPII Logits (sample):")
            for i in range(min(10, len(tokens))):
                if tokens[i] not in ["[CLS]", "[SEP]", "[PAD]"]:
                    logging.info(f"\nToken {i} '{tokens[i]}':")
                    # Show first 10 label logits
                    label_items = list(pii_id2label.items())[:10]
                    for label_id, label_name in label_items:
                        if label_id < len(pii_logits[i]):
                            logit_val = pii_logits[i][label_id]
                            logging.info(f"  {label_name:<20} logit: {logit_val:.4f}")

        if coref_logits:
            logging.info("\nCo-reference Logits (sample):")
            for i in range(min(10, len(tokens))):
                if tokens[i] not in ["[CLS]", "[SEP]", "[PAD]"]:
                    logging.info(f"\nToken {i} '{tokens[i]}':")
                    for label_id in range(min(10, len(coref_logits[i]))):
                        logit_val = coref_logits[i][label_id]
                        label_name = (
                            coref_id2label.get(label_id, f"CLUSTER_{label_id}")
                            if coref_id2label
                            else f"CLUSTER_{label_id}"
                        )
                        logging.info(f"  {label_name:<15} logit: {logit_val:.4f}")

    # Summary statistics
    logging.info(f"\n{'=' * 80}")
    logging.info("Summary Statistics")
    logging.info(f"{'=' * 80}")

    # Count entities
    pii_entities = sum(1 for label in pii_preds if label.startswith("B-"))
    logging.info(f"PII entities detected: {pii_entities}")

    # Count co-reference clusters
    coref_clusters = set(coref_preds) - {0}
    logging.info(f"Co-reference clusters: {len(coref_clusters)}")

    # Average confidence
    valid_tokens = [
        i for i, token in enumerate(tokens) if token not in ["[CLS]", "[SEP]", "[PAD]"]
    ]
    if valid_tokens:
        avg_pii_conf = sum(
            detailed_output["pii_probabilities"][i][detailed_output["pii_pred_ids"][i]]
            for i in valid_tokens
        ) / len(valid_tokens)
        avg_coref_conf = sum(
            detailed_output["coref_probabilities"][i][coref_preds[i]]
            for i in valid_tokens
        ) / len(valid_tokens)
        logging.info(f"Average PII confidence: {avg_pii_conf:.4f}")
        logging.info(f"Average co-reference confidence: {avg_coref_conf:.4f}")

    logging.info(f"{'=' * 80}\n")


def main():
    """Main execution function."""
    parser = argparse.ArgumentParser(
        description="Detailed evaluation of PII detection model"
    )
    parser.add_argument(
        "--local-model",
        type=str,
        default="./model/trained",
        help="Path to local model directory (default: ./model/trained)",
    )
    parser.add_argument(
        "--num-tests",
        type=int,
        default=5,
        help="Number of test cases to run (default: 5)",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=3,
        help="Number of top predictions to show per token (default: 3)",
    )
    parser.add_argument(
        "--show-logits",
        action="store_true",
        help="Show raw logits (can be verbose)",
    )

    args = parser.parse_args()

    logging.info("=" * 80)
    logging.info("PII Detection Model - Detailed Evaluation")
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
    loader = DetailedPIIModelLoader(model_path)
    loader.load_model()

    # Run inference on test cases
    logging.info(
        f"\n🚀 Running detailed inference on {min(args.num_tests, len(TEST_CASES))} test cases..."
    )

    inference_times = []

    for i, test_text in enumerate(TEST_CASES[: args.num_tests], 1):
        detailed_output = loader.predict_detailed(
            test_text, top_k=args.top_k, show_logits=args.show_logits
        )
        inference_times.append(detailed_output["inference_time_ms"])
        print_detailed_results(
            test_text,
            detailed_output,
            i,
            top_k=args.top_k,
            show_logits=args.show_logits,
            coref_id2label=loader.coref_id2label,
            pii_id2label=loader.pii_id2label,
        )

    # Calculate statistics
    avg_time = sum(inference_times) / len(inference_times) if inference_times else 0
    min_time = min(inference_times) if inference_times else 0
    max_time = max(inference_times) if inference_times else 0
    total_time = sum(inference_times)

    # Summary
    logging.info(f"\n{'=' * 80}")
    logging.info("✅ Detailed Evaluation Complete!")
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
    logging.info(f"{'=' * 80}\n")


if __name__ == "__main__":
    main()
