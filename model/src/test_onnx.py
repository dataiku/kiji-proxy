"""
Simple ONNX Model Test Script

Tests the quantized ONNX model for PII detection.

Usage:
    python model/src/test_onnx.py
    python model/src/test_onnx.py --model-path ./model/quantized
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import onnxruntime as ort
from transformers import AutoTokenizer


def load_model(model_path: str):
    """Load ONNX model and tokenizer."""
    model_path = Path(model_path)

    # Find ONNX model file
    onnx_file = model_path / "model_quantized.onnx"
    if not onnx_file.exists():
        onnx_file = model_path / "model.onnx"
    if not onnx_file.exists():
        raise FileNotFoundError(f"No ONNX model found in {model_path}")

    print(f"Loading ONNX model from: {onnx_file}")

    # Load ONNX session
    session = ort.InferenceSession(str(onnx_file), providers=["CPUExecutionProvider"])

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_path)

    # Load label mappings
    mappings_path = model_path / "label_mappings.json"
    if not mappings_path.exists():
        raise FileNotFoundError(f"Label mappings not found at {mappings_path}")

    with mappings_path.open() as f:
        mappings = json.load(f)

    id2label = {int(k): v for k, v in mappings["id2label"].items()}

    print(f"Loaded {len(id2label)} labels")
    print(f"Model inputs: {[i.name for i in session.get_inputs()]}")
    print(f"Model outputs: {[o.name for o in session.get_outputs()]}")

    return session, tokenizer, id2label


def predict(session, tokenizer, id2label, text: str):
    """Run inference on text."""
    # Tokenize
    inputs = tokenizer(
        text,
        return_tensors="np",
        truncation=True,
        max_length=4096,
        return_offsets_mapping=True,
    )

    offset_mapping = inputs.pop("offset_mapping")[0]

    # Run inference
    start_time = time.perf_counter()
    outputs = session.run(
        None,
        {
            "input_ids": inputs["input_ids"].astype(np.int64),
            "attention_mask": inputs["attention_mask"].astype(np.int64),
        },
    )
    inference_time_ms = (time.perf_counter() - start_time) * 1000

    logits = outputs[0][0]  # (seq_len, num_labels)

    # Get predictions
    predictions = np.argmax(logits, axis=-1)

    # Extract entities
    tokens = tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])

    entities = []
    current_entity = None
    current_label = None
    current_start = None
    current_end = None

    for idx, (token, pred_id, offset) in enumerate(
        zip(tokens, predictions, offset_mapping)
    ):
        label = id2label.get(pred_id, "O")

        # Skip special tokens
        if offset[0] == 0 and offset[1] == 0:
            continue

        if label.startswith("B-"):
            # Save previous entity
            if current_entity is not None:
                entities.append(
                    (
                        text[current_start:current_end],
                        current_label,
                        current_start,
                        current_end,
                    )
                )

            # Start new entity
            current_label = label[2:]
            current_start = offset[0]
            current_end = offset[1]
            current_entity = token

        elif label.startswith("I-") and current_entity is not None:
            # Continue entity
            if current_label == label[2:]:
                current_end = offset[1]
            else:
                # Different label, save and start new
                entities.append(
                    (
                        text[current_start:current_end],
                        current_label,
                        current_start,
                        current_end,
                    )
                )
                current_label = label[2:]
                current_start = offset[0]
                current_end = offset[1]

        elif current_entity is not None:
            # End of entity
            entities.append(
                (
                    text[current_start:current_end],
                    current_label,
                    current_start,
                    current_end,
                )
            )
            current_entity = None
            current_label = None

    # Don't forget last entity
    if current_entity is not None:
        entities.append(
            (text[current_start:current_end], current_label, current_start, current_end)
        )

    return entities, inference_time_ms


def main():
    parser = argparse.ArgumentParser(description="Test ONNX PII detection model")
    parser.add_argument(
        "--model-path",
        type=str,
        default="./model/quantized",
        help="Path to quantized model directory",
    )
    args = parser.parse_args()

    print("=" * 60)
    print("ONNX PII Detection Model Test")
    print("=" * 60)

    # Load model
    try:
        session, tokenizer, id2label = load_model(args.model_path)
    except FileNotFoundError as e:
        print(f"\nError: {e}")
        print("\nMake sure you have quantized the model first:")
        print("  python model/src/quantitize.py")
        sys.exit(1)

    # Test cases
    test_cases = [
        "My name is John Smith and my email is john.smith@example.com.",
        "Call me at 555-123-4567 or visit 123 Main Street, Boston, MA 02101.",
        "Patient DOB: 03/15/1985, SSN: 123-45-6789.",
        "Contact Sarah Johnson at sarah.j@company.org for more info.",
    ]

    print("\n" + "=" * 60)
    print("Running inference tests...")
    print("=" * 60)

    total_time = 0
    total_entities = 0

    for i, text in enumerate(test_cases, 1):
        print(f"\n--- Test {i} ---")
        print(f"Text: {text}")

        entities, inference_time = predict(session, tokenizer, id2label, text)
        total_time += inference_time
        total_entities += len(entities)

        print(f"Time: {inference_time:.2f}ms")
        print("Entities:")
        if entities:
            for entity_text, label, start, end in entities:
                print(f"  [{label}] '{entity_text}' ({start}-{end})")
        else:
            print("  (none detected)")

    # Summary
    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)
    print(f"Total tests: {len(test_cases)}")
    print(f"Total entities detected: {total_entities}")
    print(f"Total inference time: {total_time:.2f}ms")
    print(f"Average time per test: {total_time / len(test_cases):.2f}ms")
    print(f"Throughput: {1000 / (total_time / len(test_cases)):.1f} texts/sec")
    print("=" * 60)


if __name__ == "__main__":
    main()
