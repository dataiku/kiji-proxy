"""
Test ONNX Model Inference

This script tests the quantized ONNX model to verify:
1. Model loads correctly
2. Tokenizer works
3. Inference produces valid outputs
4. Entities are detected properly
5. Performance metrics

Usage:
    python test_onnx_inference.py
    python test_onnx_inference.py --model-path model/quantized
    python test_onnx_inference.py --verbose
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import onnxruntime as ort
from transformers import AutoTokenizer


class ONNXModelTester:
    """Test ONNX PII detection model."""

    def __init__(self, model_dir: str = "model/quantized"):
        """
        Initialize ONNX model tester.

        Args:
            model_dir: Directory containing ONNX model and tokenizer
        """
        self.model_dir = Path(model_dir)
        self.session = None
        self.tokenizer = None
        self.pii_id2label = None
        self.coref_id2label = None

    def load_model(self):
        """Load ONNX model and tokenizer."""
        print("\n" + "=" * 80)
        print("LOADING MODEL")
        print("=" * 80)

        # Load ONNX model
        model_path = self.model_dir / "model_quantized.onnx"
        print(f"Loading ONNX model: {model_path}")

        if not model_path.exists():
            raise FileNotFoundError(f"Model not found: {model_path}")

        # Create ONNX Runtime session
        self.session = ort.InferenceSession(
            str(model_path),
            providers=["CPUExecutionProvider"],  # Use CPU for compatibility
        )

        print(f"✅ ONNX model loaded successfully")
        print(f"   Providers: {self.session.get_providers()}")

        # Load tokenizer
        print(f"\nLoading tokenizer from: {self.model_dir}")
        self.tokenizer = AutoTokenizer.from_pretrained(str(self.model_dir))
        print(f"✅ Tokenizer loaded")
        print(f"   Type: {self.tokenizer.__class__.__name__}")
        print(f"   Vocab size: {self.tokenizer.vocab_size}")
        print(f"   Model max length: {self.tokenizer.model_max_length}")

        # Load label mappings
        mappings_path = self.model_dir / "label_mappings.json"
        print(f"\nLoading label mappings: {mappings_path}")

        with mappings_path.open() as f:
            mappings = json.load(f)

        self.pii_id2label = {
            int(k): v for k, v in mappings["pii"]["id2label"].items() if k != "-100"
        }
        self.coref_id2label = {
            int(k): v for k, v in mappings["coref"]["id2label"].items()
        }

        print(f"✅ Label mappings loaded")
        print(f"   PII labels: {len(self.pii_id2label)}")
        print(f"   Coref labels: {len(self.coref_id2label)}")

        # Show model info
        print("\n" + "-" * 80)
        print("MODEL INFORMATION")
        print("-" * 80)
        print(f"Inputs:")
        for inp in self.session.get_inputs():
            print(f"  - {inp.name}: {inp.shape} ({inp.type})")
        print(f"\nOutputs:")
        for out in self.session.get_outputs():
            print(f"  - {out.name}: {out.shape} ({out.type})")

    def predict(self, text: str, verbose: bool = False) -> dict:
        """
        Run inference on text.

        Args:
            text: Input text to analyze
            verbose: Print detailed token-level predictions

        Returns:
            Dictionary with entities, timing, and debug info
        """
        if self.session is None:
            raise RuntimeError("Model not loaded. Call load_model() first.")

        # Split text into words - this matches training tokenization
        words = text.split()

        # Tokenize using is_split_into_words=True to match training
        encoding = self.tokenizer(
            words,
            is_split_into_words=True,
            return_tensors="np",
            return_offsets_mapping=True,
            max_length=4096,
            truncation=True,
        )

        input_ids = encoding["input_ids"].astype(np.int64)
        attention_mask = encoding["attention_mask"].astype(np.int64)

        # Get word_ids for mapping tokens back to words
        word_ids = encoding.word_ids(batch_index=0)

        # Build character offsets from words
        # We need to map tokens back to character positions in original text
        char_offsets = self._build_char_offsets(text, words, word_ids)

        if verbose:
            print(f"\n📝 Tokenization:")
            print(f"   Words: {len(words)}")
            print(f"   Tokens: {len(input_ids[0])}")
            print(f"   Input IDs shape: {input_ids.shape}")
            print(f"   Attention mask shape: {attention_mask.shape}")

        # Run inference
        start_time = time.time()

        outputs = self.session.run(
            None,  # Get all outputs
            {
                "input_ids": input_ids,
                "attention_mask": attention_mask,
            },
        )

        inference_time = (time.time() - start_time) * 1000  # Convert to ms

        # Parse outputs
        pii_logits = outputs[0]  # Shape: (1, seq_len, num_pii_labels)
        coref_logits = outputs[1]  # Shape: (1, seq_len, num_coref_labels)

        if verbose:
            print(f"\n🔮 Model Output:")
            print(f"   PII logits shape: {pii_logits.shape}")
            print(f"   Coref logits shape: {coref_logits.shape}")
            print(f"   Inference time: {inference_time:.2f}ms")

        # Get predictions
        pii_predictions = np.argmax(pii_logits[0], axis=-1)  # Shape: (seq_len,)

        # Calculate confidence (softmax)
        pii_probs = self._softmax(pii_logits[0])  # Shape: (seq_len, num_labels)
        pii_confidences = np.max(pii_probs, axis=-1)  # Shape: (seq_len,)

        # Extract entities
        entities = self._extract_entities(
            text, pii_predictions, pii_confidences, char_offsets, verbose=verbose
        )

        return {
            "text": text,
            "entities": entities,
            "num_tokens": len(input_ids[0]),
            "inference_time_ms": inference_time,
            "pii_predictions": pii_predictions.tolist(),
            "pii_confidences": pii_confidences.tolist(),
        }

    def _softmax(self, x: np.ndarray) -> np.ndarray:
        """Apply softmax to logits."""
        exp_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
        return exp_x / np.sum(exp_x, axis=-1, keepdims=True)

    def _extract_entities(
        self,
        text: str,
        predictions: np.ndarray,
        confidences: np.ndarray,
        offsets: np.ndarray,
        confidence_threshold: float = 0.5,
        verbose: bool = False,
    ) -> list:
        """
        Extract entities from token predictions using BIO tagging.

        Args:
            text: Original text
            predictions: Token class predictions
            confidences: Token confidence scores
            offsets: Token offset mappings
            confidence_threshold: Minimum confidence to keep
            verbose: Print token-level details

        Returns:
            List of detected entities
        """
        entities = []
        current_entity = None
        current_tokens = []

        if verbose:
            print(f"\n🔍 Token Predictions (showing non-O and first 10):")

        for i, (pred_id, conf) in enumerate(zip(predictions, confidences)):
            # Get token offset
            start, end = offsets[i]

            # Skip special tokens ([CLS], [SEP]) - they have offset (0, 0)
            is_special_token = start == 0 and end == 0

            # Get label
            label = self.pii_id2label.get(pred_id, "O")

            # Special tokens are always treated as "O"
            if is_special_token:
                label = "O"
            # Apply confidence threshold
            elif conf < confidence_threshold:
                label = "O"

            # Debug output
            if verbose and (label != "O" or i < 10):
                if is_special_token:
                    token_text = "[CLS]" if i == 0 else "[SEP]"
                else:
                    token_text = (
                        text[start:end]
                        if start < len(text) and end <= len(text)
                        else ""
                    )
                special_marker = " (special)" if is_special_token else ""
                print(
                    f"   Token[{i:3d}] {token_text:20s} → {label:20s} (conf: {conf:.3f}){special_marker}"
                )

            # Parse BIO tags
            is_beginning = label.startswith("B-")
            is_inside = label.startswith("I-")

            if is_beginning or is_inside:
                base_label = label[2:]  # Remove B- or I- prefix
            else:
                base_label = label

            # Handle entity states
            if label != "O" and (is_beginning or current_entity is None):
                # Finish previous entity
                if current_entity is not None:
                    self._finalize_entity(
                        current_entity, current_tokens, text, offsets, entities
                    )

                # Start new entity
                current_entity = {
                    "label": base_label,
                    "confidence": conf,
                }
                current_tokens = [i]

            elif label != "O" and is_inside and current_entity is not None:
                if current_entity["label"] == base_label:
                    # Continue current entity
                    current_tokens.append(i)
                    # Update confidence (average)
                    current_entity["confidence"] = (
                        current_entity["confidence"] + conf
                    ) / 2
                else:
                    # Different label - finish current and start new
                    self._finalize_entity(
                        current_entity, current_tokens, text, offsets, entities
                    )
                    current_entity = {
                        "label": base_label,
                        "confidence": conf,
                    }
                    current_tokens = [i]

            else:
                # O label - finish current entity
                if current_entity is not None:
                    self._finalize_entity(
                        current_entity, current_tokens, text, offsets, entities
                    )
                    current_entity = None
                    current_tokens = []

        # Finish last entity
        if current_entity is not None:
            self._finalize_entity(
                current_entity, current_tokens, text, offsets, entities
            )

        return entities

    def _finalize_entity(
        self,
        entity: dict,
        token_indices: list,
        text: str,
        offsets: np.ndarray,
        entities: list,
    ):
        """Extract entity text from token offsets and add to entities list."""
        if not token_indices:
            return

        # Get start and end positions
        start_offset = offsets[token_indices[0]]
        end_offset = offsets[token_indices[-1]]

        start_pos = int(start_offset[0])
        end_pos = int(end_offset[1])

        # Extract text
        entity["text"] = text[start_pos:end_pos]
        entity["start"] = start_pos
        entity["end"] = end_pos

        entities.append(entity)


def run_tests(model_dir: str = "model/quantized", verbose: bool = False):
    """Run test suite on ONNX model."""
    print("\n" + "🧪" * 40)
    print("ONNX MODEL TEST SUITE")
    print("🧪" * 40)

    # Initialize tester
    tester = ONNXModelTester(model_dir)
    tester.load_model()

    # Test cases
    test_cases = [
        {
            "name": "Basic PII Detection",
            "text": "John Smith's email is john.smith@example.com and phone is 555-1234",
            "expected_labels": ["FIRSTNAME", "SURNAME", "EMAIL", "PHONENUMBER"],
        },
        {
            "name": "Address Information",
            "text": "He lives at 123 Main Street, New York, NY 10001",
            "expected_labels": ["BUILDINGNUM", "STREET", "CITY", "STATE", "ZIP"],
        },
        {
            "name": "Personal Identifiers",
            "text": "SSN: 123-45-6789, Driver's License: D1234567",
            "expected_labels": ["SSN", "DRIVERLICENSENUM"],
        },
        {
            "name": "Company and URL",
            "text": "Works at Acme Corporation, website: https://acme.com",
            "expected_labels": ["COMPANYNAME", "URL"],
        },
        {
            "name": "Date of Birth",
            "text": "Born on 01/15/1990, age 35",
            "expected_labels": ["DATEOFBIRTH", "AGE"],
        },
        {
            "name": "No PII (Control)",
            "text": "The weather is nice today and the sky is blue.",
            "expected_labels": [],
        },
    ]

    print("\n" + "=" * 80)
    print("RUNNING TEST CASES")
    print("=" * 80)

    results = []
    total_time = 0

    for i, test_case in enumerate(test_cases, 1):
        print(f"\n{'─' * 80}")
        print(f"Test {i}/{len(test_cases)}: {test_case['name']}")
        print(f"{'─' * 80}")
        print(f"Input: {test_case['text']}")

        # Run prediction
        result = tester.predict(test_case["text"], verbose=verbose)

        # Display results
        print(f"\n📊 Results:")
        print(f"   Tokens: {result['num_tokens']}")
        print(f"   Inference time: {result['inference_time_ms']:.2f}ms")
        print(f"   Entities found: {len(result['entities'])}")

        if result["entities"]:
            print(f"\n   Detected Entities:")
            for entity in result["entities"]:
                print(
                    f'      • {entity["label"]:20s}: "{entity["text"]}" '
                    f"(confidence: {entity['confidence']:.3f})"
                )
        else:
            print(f"   (No entities detected)")

        # Check against expected
        detected_labels = set(e["label"] for e in result["entities"])
        expected_labels = set(test_case["expected_labels"])

        missing = expected_labels - detected_labels
        extra = detected_labels - expected_labels

        if missing or extra:
            print(f"\n   ⚠️  Differences from expected:")
            if missing:
                print(f"      Missing: {missing}")
            if extra:
                print(f"      Extra: {extra}")
        else:
            print(f"   ✅ All expected labels detected!")

        results.append(result)
        total_time += result["inference_time_ms"]

    # Summary
    print("\n" + "=" * 80)
    print("TEST SUMMARY")
    print("=" * 80)

    total_entities = sum(len(r["entities"]) for r in results)
    avg_time = total_time / len(results)

    print(f"Total tests: {len(test_cases)}")
    print(f"Total entities detected: {total_entities}")
    print(f"Average inference time: {avg_time:.2f}ms")
    print(f"Total inference time: {total_time:.2f}ms")

    # Performance rating
    if avg_time < 100:
        rating = "🚀 Excellent"
    elif avg_time < 500:
        rating = "✅ Good"
    elif avg_time < 1000:
        rating = "⚠️  Acceptable"
    else:
        rating = "❌ Slow"

    print(f"Performance: {rating}")

    # Entity type distribution
    print(f"\n📈 Entity Type Distribution:")
    label_counts = {}
    for result in results:
        for entity in result["entities"]:
            label = entity["label"]
            label_counts[label] = label_counts.get(label, 0) + 1

    for label, count in sorted(label_counts.items(), key=lambda x: -x[1]):
        print(f"   {label:20s}: {count}")

    print("\n" + "=" * 80)
    print("✅ Testing complete!")
    print("=" * 80 + "\n")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Test ONNX PII detection model")
    parser.add_argument(
        "--model-path",
        type=str,
        default="model/quantized",
        help="Path to model directory (default: model/quantized)",
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Print detailed token-level predictions",
    )
    parser.add_argument(
        "--text",
        type=str,
        help="Test on custom text instead of test suite",
    )

    args = parser.parse_args()

    if args.text:
        # Single text prediction
        tester = ONNXModelTester(args.model_path)
        tester.load_model()
        result = tester.predict(args.text, verbose=True)

        print(f"\n✅ Detected {len(result['entities'])} entities:")
        for entity in result["entities"]:
            print(
                f'   • {entity["label"]}: "{entity["text"]}" '
                f"(confidence: {entity['confidence']:.3f})"
            )
    else:
        # Run full test suite
        run_tests(args.model_path, verbose=args.verbose)


if __name__ == "__main__":
    main()
