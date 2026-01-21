"""
Detailed PII Detection Model Evaluation Script

This script provides detailed, token-level outputs from the model including:
1. Token-by-token predictions with confidence scores
2. Top-k predictions per token
3. Raw logits and probabilities
4. Detailed entity extraction breakdown

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

# Add parent directory to path for imports
src_dir = Path(__file__).parent
if str(src_dir) not in sys.path:
    sys.path.insert(0, str(src_dir))

from model import PIIDetectionModel

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
    """Loads and manages PII detection model with detailed output capabilities."""

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
        self.device = get_device()

    def load_model(self):
        """Load model, tokenizer, and label mappings."""
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

        # Load label mappings
        self.label2id = mappings["label2id"]
        self.id2label = {int(k): v for k, v in mappings["id2label"].items()}
        logging.info(f"✅ Loaded {len(self.label2id)} label mappings")

        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
        logging.info("✅ Loaded tokenizer")

        # Use ModernBERT as the base model
        base_model_name = "answerdotai/ModernBERT-base"

        # Determine number of labels
        num_labels = len(self.label2id)

        # Find model weights file
        model_weights_path = Path(self.model_path) / "pytorch_model.bin"
        if not model_weights_path.exists():
            model_weights_path = Path(self.model_path) / "model.safetensors"
            if not model_weights_path.exists():
                bin_files = list(Path(self.model_path).glob("*.bin"))
                if bin_files:
                    model_weights_path = bin_files[0]
                    logging.info(f"   Found weights: {model_weights_path.name}")

        logging.info("📋 Model configuration:")
        logging.info(f"   Base model: {base_model_name}")
        logging.info(f"   Labels: {num_labels}")

        # Load model
        self.model = PIIDetectionModel(
            model_name=base_model_name,
            num_labels=num_labels,
            id2label=self.id2label,
        )

        # Load model weights
        state_dict = None
        if model_weights_path.exists():
            logging.info(f"📦 Loading weights from: {model_weights_path.name}")

            if model_weights_path.suffix == ".safetensors":
                state_dict = {}
                with safe_open(model_weights_path, framework="pt", device="cpu") as f:
                    for key in f.keys():
                        state_dict[key] = f.get_tensor(key)
            else:
                state_dict = torch.load(
                    model_weights_path, map_location="cpu", weights_only=False
                )

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
        Run inference with detailed output including probabilities and top-k predictions.

        Args:
            text: Input text to analyze
            top_k: Number of top predictions to return per token
            show_logits: Whether to include raw logits in output

        Returns:
            Dictionary containing:
            - tokens: List of token strings
            - predictions: List of predicted label IDs
            - probabilities: List of probability distributions
            - top_k: List of top-k predictions per token
            - inference_time_ms: Inference time in milliseconds
            - logits: Raw logits (if show_logits=True)
        """
        if self.model is None or self.tokenizer is None:
            raise ValueError("Model not loaded. Call load_model() first.")

        start_time = time.perf_counter()

        # Tokenize input
        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=4096,
            return_offsets_mapping=True,
        )

        offset_mapping = inputs.pop("offset_mapping")[0]
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        # Run inference
        with torch.no_grad():
            outputs = self.model(**inputs)
            logits = outputs["logits"][0]  # [seq_len, num_labels]
            predictions = torch.argmax(logits, dim=-1)  # [seq_len]
            probs = F.softmax(logits, dim=-1)  # [seq_len, num_labels]

        end_time = time.perf_counter()
        inference_time_ms = (end_time - start_time) * 1000

        # Convert to lists
        tokens = self.tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])
        pred_ids = [p.item() for p in predictions]

        # Get top-k predictions
        top_k_list = []
        for i in range(len(tokens)):
            top_k_probs, top_k_ids = torch.topk(
                probs[i], k=min(top_k, len(self.id2label))
            )
            top_k_items = [
                {
                    "label": self.id2label.get(idx.item(), f"UNK_{idx.item()}"),
                    "probability": prob.item(),
                }
                for idx, prob in zip(top_k_ids, top_k_probs, strict=True)
            ]
            top_k_list.append(top_k_items)

        # Get probability distributions
        prob_list = [probs[i].cpu().tolist() for probs in [probs]]

        result = {
            "tokens": tokens,
            "offset_mapping": offset_mapping.tolist(),
            "predictions": pred_ids,
            "probabilities": [probs[i].cpu().tolist() for i in range(len(tokens))],
            "top_k": top_k_list,
            "inference_time_ms": inference_time_ms,
            "text": text,
        }

        if show_logits:
            result["logits"] = logits.cpu().tolist()

        return result


def print_detailed_results(
    detailed_output: dict,
    id2label: dict[int, str],
    show_logits: bool = False,
):
    """
    Print detailed inference results in a formatted way.

    Args:
        detailed_output: Output from predict_detailed()
        id2label: Mapping from label ID to label name
        show_logits: Whether to show raw logits
    """
    text = detailed_output["text"]
    tokens = detailed_output["tokens"]
    preds = detailed_output["predictions"]
    top_k = detailed_output["top_k"]

    logging.info(f"\n{'=' * 80}")
    logging.info(f"Text: {text}")
    logging.info(f"Inference Time: {detailed_output['inference_time_ms']:.2f} ms")
    logging.info(f"{'=' * 80}")

    # Print token-by-token results
    logging.info(f"\n{'Token':<20} {'Label':<20} {'Confidence':<10}")
    logging.info("-" * 50)

    for i, (token, label_id) in enumerate(zip(tokens, preds, strict=True)):
        # Skip special tokens
        if token in ["[CLS]", "[SEP]", "[PAD]", "<s>", "</s>", "<pad>"]:
            continue

        # Get label and confidence
        label = id2label.get(label_id, f"UNK_{label_id}")
        confidence = detailed_output["probabilities"][i][label_id]

        # Highlight PII tokens
        pii_marker = "◆" if label != "O" else " "

        # Clean token display
        display_token = token.replace("▁", " ").replace("Ġ", " ")
        if len(display_token) > 18:
            display_token = display_token[:15] + "..."

        logging.info(f"{display_token:<20} {pii_marker} {label:<18} {confidence:.4f}")

    # Print top-k predictions for PII tokens
    logging.info("\n📊 Top-K Predictions for PII Tokens:")
    for i, (token, top_k_items) in enumerate(zip(tokens, top_k, strict=True)):
        label_id = preds[i]
        label = id2label.get(label_id, f"UNK_{label_id}")
        if label != "O" and token not in [
            "[CLS]",
            "[SEP]",
            "[PAD]",
            "<s>",
            "</s>",
            "<pad>",
        ]:
            display_token = token.replace("▁", " ").replace("Ġ", " ")
            logging.info(f"\n  Token: '{display_token}'")
            for item in top_k_items:
                logging.info(f"    {item['label']:<20} {item['probability']:.4f}")

    # Print raw logits if requested
    if show_logits:
        logits = detailed_output.get("logits", [])
        if logits:
            logging.info("\n📈 Raw Logits (first 5 tokens, first 10 labels):")
            for i, (token, token_logits) in enumerate(
                zip(tokens[:5], logits[:5], strict=True)
            ):
                display_token = token.replace("▁", " ").replace("Ġ", " ")
                logging.info(f"\n  Token: '{display_token}'")
                for label_id in range(min(10, len(token_logits))):
                    logit_val = token_logits[label_id]
                    label = id2label.get(label_id, f"UNK_{label_id}")
                    logging.info(f"    {label:<20} {logit_val:.4f}")

    # Summary statistics
    logging.info(f"\n{'=' * 80}")
    logging.info("📊 Summary Statistics")
    logging.info(f"{'=' * 80}")

    pii_tokens = sum(1 for p in preds if id2label.get(p, "O") != "O")
    logging.info(f"Total tokens: {len(tokens)}")
    logging.info(f"PII tokens: {pii_tokens}")

    if pii_tokens > 0:
        avg_confidence = (
            sum(
                detailed_output["probabilities"][i][preds[i]]
                for i in range(len(tokens))
                if id2label.get(preds[i], "O") != "O"
            )
            / pii_tokens
        )
        logging.info(f"Average PII confidence: {avg_confidence:.4f}")


# =============================================================================
# TEST CASES
# =============================================================================

TEST_CASES = [
    "My name is John Smith and my email is john.smith@email.com.",
    "Please contact Sarah Johnson at 555-123-4567.",
    "The patient's SSN is 123-45-6789.",
]


def main():
    """Main execution function."""
    parser = argparse.ArgumentParser(
        description="Evaluate PII detection model with detailed output"
    )
    parser.add_argument(
        "--local-model",
        type=str,
        default="./model/trained",
        help="Path to local model directory",
    )
    parser.add_argument(
        "--num-tests",
        type=int,
        default=3,
        help="Number of test cases to run",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=3,
        help="Number of top predictions to show per token",
    )
    parser.add_argument(
        "--show-logits",
        action="store_true",
        help="Show raw logits for each token",
    )

    args = parser.parse_args()

    logging.info("=" * 80)
    logging.info("Detailed PII Detection Model Evaluation")
    logging.info("=" * 80)

    # Determine model path
    model_path = args.local_model

    if not Path(model_path).exists():
        logging.warning(f"⚠️  Model path not found: {model_path}")
        local_paths = ["./model/trained", "../model/trained", "model/trained"]
        for path in local_paths:
            if Path(path).exists():
                model_path = path
                logging.info(f"✅ Found model at: {model_path}")
                break

    if not Path(model_path).exists():
        logging.error("\n❌ No model found! Please specify a valid model path.")
        return

    logging.info(f"\n📁 Using model: {model_path}")

    # Load model
    logging.info("\n📥 Loading model...")
    loader = DetailedPIIModelLoader(model_path)
    loader.load_model()

    # Run detailed inference
    logging.info(
        f"\n🚀 Running detailed inference on {min(args.num_tests, len(TEST_CASES))} test cases..."
    )

    for i, test_text in enumerate(TEST_CASES[: args.num_tests], 1):
        logging.info(f"\n\n{'#' * 80}")
        logging.info(f"TEST CASE {i}")
        logging.info(f"{'#' * 80}")

        detailed_output = loader.predict_detailed(
            test_text,
            top_k=args.top_k,
            show_logits=args.show_logits,
        )

        print_detailed_results(
            detailed_output,
            id2label=loader.id2label,
            show_logits=args.show_logits,
        )

    logging.info(f"\n{'=' * 80}")
    logging.info("✅ Detailed Evaluation Complete!")
    logging.info(f"{'=' * 80}\n")


if __name__ == "__main__":
    main()
