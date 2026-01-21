"""
Diagnostic script to identify training/inference mismatch in ModernBERT PII detection.

This script checks:
1. Tokenization consistency between training and inference
2. Label alignment correctness
3. Model output distribution
4. Comparison of PyTorch vs ONNX outputs

Run this on the remote machine after training to diagnose issues.

Usage:
    python diagnose_inference.py --trained-model ./model/trained
    python diagnose_inference.py --trained-model ./model/trained --quantized-model ./model/quantized
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from transformers import AutoTokenizer

# Add project root to path
project_root = Path(__file__).parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))


def diagnose_tokenization(tokenizer, text: str, privacy_mask: list):
    """Diagnose tokenization and label alignment."""
    print("\n" + "=" * 80)
    print("TOKENIZATION DIAGNOSIS")
    print("=" * 80)

    print(f"\nInput text: {text}")
    print(f"Privacy mask: {privacy_mask}")

    # Tokenize with offsets
    encoding = tokenizer(
        text,
        return_offsets_mapping=True,
        return_tensors="pt",
        truncation=True,
        max_length=512,
    )

    tokens = tokenizer.convert_ids_to_tokens(encoding["input_ids"][0])
    offsets = encoding["offset_mapping"][0].tolist()

    print(f"\nTokenizer type: {tokenizer.__class__.__name__}")
    print(f"Vocab size: {tokenizer.vocab_size}")
    print(f"Number of tokens: {len(tokens)}")

    print("\n" + "-" * 80)
    print("TOKEN DETAILS (first 30 tokens):")
    print("-" * 80)
    print(f"{'Idx':<5} {'Token':<20} {'Token ID':<10} {'Offset':<15} {'Text Span':<20}")
    print("-" * 80)

    for i, (token, offset) in enumerate(zip(tokens[:30], offsets[:30])):
        start, end = offset
        if start == 0 and end == 0:
            span = "[SPECIAL]"
        else:
            span = repr(text[start:end])

        token_id = encoding["input_ids"][0][i].item()
        print(f"{i:<5} {token:<20} {token_id:<10} {str(offset):<15} {span:<20}")

    # Check for entity alignment
    print("\n" + "-" * 80)
    print("ENTITY ALIGNMENT CHECK:")
    print("-" * 80)

    for item in privacy_mask:
        value = item["value"]
        label = item["label"]

        # Find value in text
        pos = text.find(value)
        if pos == -1:
            print(f"  WARNING: '{value}' ({label}) not found in text!")
            continue

        end_pos = pos + len(value)
        print(f"\n  Entity: '{value}' ({label})")
        print(f"  Position: {pos}-{end_pos}")

        # Find which tokens cover this entity
        covering_tokens = []
        for i, (start, end) in enumerate(offsets):
            if start == 0 and end == 0:
                continue
            # Check overlap
            if start < end_pos and end > pos:
                token = tokens[i]
                covering_tokens.append((i, token, start, end))

        print(f"  Covering tokens:")
        for idx, token, start, end in covering_tokens:
            print(f"    Token[{idx}]: '{token}' ({start}-{end})")


def diagnose_model_outputs(model_path: str, tokenizer, text: str):
    """Diagnose PyTorch model outputs."""
    print("\n" + "=" * 80)
    print("PYTORCH MODEL OUTPUT DIAGNOSIS")
    print("=" * 80)

    from safetensors import safe_open

    try:
        from model.src.model import MultiTaskPIIDetectionModel
    except ImportError:
        from model import MultiTaskPIIDetectionModel

    model_path = Path(model_path)

    # Load label mappings
    mappings_path = model_path / "label_mappings.json"
    with open(mappings_path) as f:
        mappings = json.load(f)

    pii_label2id = mappings["pii"]["label2id"]
    pii_id2label = {
        int(k): v for k, v in mappings["pii"]["id2label"].items() if k != "-100"
    }
    coref_id2label = {int(k): v for k, v in mappings["coref"]["id2label"].items()}

    print(f"\nLoading model from: {model_path}")
    print(f"PII labels: {len(pii_label2id)}")

    # Initialize model
    model = MultiTaskPIIDetectionModel(
        model_name="answerdotai/ModernBERT-base",
        num_pii_labels=len(pii_label2id),
        num_coref_labels=len(coref_id2label),
        id2label_pii=pii_id2label,
        id2label_coref=coref_id2label,
    )

    # Load weights
    weights_path = model_path / "model.safetensors"
    if not weights_path.exists():
        weights_path = model_path / "pytorch_model.bin"

    if weights_path.exists():
        print(f"Loading weights from: {weights_path}")

        if weights_path.suffix == ".safetensors":
            state_dict = {}
            with safe_open(weights_path, framework="pt", device="cpu") as f:
                for key in f.keys():
                    state_dict[key] = f.get_tensor(key)
        else:
            state_dict = torch.load(
                weights_path, map_location="cpu", weights_only=False
            )

        # Print keys to check what's being loaded
        print(f"\nState dict keys (first 20):")
        for i, key in enumerate(list(state_dict.keys())[:20]):
            print(f"  {key}")
        if len(state_dict) > 20:
            print(f"  ... and {len(state_dict) - 20} more")

        # Check for encoder weights
        encoder_keys = [k for k in state_dict.keys() if "encoder" in k]
        classifier_keys = [k for k in state_dict.keys() if "classifier" in k]

        print(f"\nEncoder weight keys: {len(encoder_keys)}")
        print(f"Classifier weight keys: {len(classifier_keys)}")

        if not encoder_keys:
            print("  WARNING: No encoder weights found! Model may have fresh encoder.")

        # Load with strict=False to see what's missing
        missing, unexpected = model.load_state_dict(state_dict, strict=False)

        if missing:
            print(f"\nMISSING KEYS ({len(missing)}):")
            for key in missing[:10]:
                print(f"  {key}")
            if len(missing) > 10:
                print(f"  ... and {len(missing) - 10} more")

        if unexpected:
            print(f"\nUNEXPECTED KEYS ({len(unexpected)}):")
            for key in unexpected[:10]:
                print(f"  {key}")
    else:
        print(f"WARNING: No weights file found at {model_path}")
        return

    model.eval()

    # Tokenize and run inference
    encoding = tokenizer(
        text,
        return_tensors="pt",
        return_offsets_mapping=True,
        truncation=True,
        max_length=512,
    )

    with torch.no_grad():
        outputs = model(
            input_ids=encoding["input_ids"],
            attention_mask=encoding["attention_mask"],
        )

    pii_logits = outputs["pii_logits"][0]  # (seq_len, num_labels)

    # Compute probabilities
    probs = torch.softmax(pii_logits, dim=-1)
    predictions = torch.argmax(probs, dim=-1)
    confidences = torch.max(probs, dim=-1).values

    tokens = tokenizer.convert_ids_to_tokens(encoding["input_ids"][0])
    offsets = encoding["offset_mapping"][0].tolist()

    print("\n" + "-" * 80)
    print("PREDICTIONS (first 30 tokens):")
    print("-" * 80)
    print(f"{'Idx':<5} {'Token':<15} {'Pred':<15} {'Conf':<8} {'Top-3 Labels'}")
    print("-" * 80)

    non_o_count = 0
    for i in range(min(30, len(tokens))):
        token = tokens[i]
        pred_id = predictions[i].item()
        pred_label = pii_id2label.get(pred_id, f"UNK-{pred_id}")
        conf = confidences[i].item()

        # Get top-3 predictions
        top3_idx = torch.topk(probs[i], 3).indices.tolist()
        top3_probs = torch.topk(probs[i], 3).values.tolist()
        top3_str = ", ".join(
            [
                f"{pii_id2label.get(idx, 'UNK')}:{p:.2f}"
                for idx, p in zip(top3_idx, top3_probs)
            ]
        )

        if pred_label != "O":
            non_o_count += 1
            marker = " <-- NON-O"
        else:
            marker = ""

        print(f"{i:<5} {token:<15} {pred_label:<15} {conf:<8.3f} {top3_str}{marker}")

    print(f"\nTotal non-O predictions: {non_o_count} / {len(tokens)}")

    # Check prediction distribution
    print("\n" + "-" * 80)
    print("PREDICTION DISTRIBUTION:")
    print("-" * 80)

    pred_counts = {}
    for pred_id in predictions.tolist():
        label = pii_id2label.get(pred_id, f"UNK-{pred_id}")
        pred_counts[label] = pred_counts.get(label, 0) + 1

    for label, count in sorted(pred_counts.items(), key=lambda x: -x[1]):
        pct = count / len(predictions) * 100
        print(f"  {label:<20}: {count:4d} ({pct:5.1f}%)")

    # Check logit statistics
    print("\n" + "-" * 80)
    print("LOGIT STATISTICS:")
    print("-" * 80)

    print(f"  Shape: {pii_logits.shape}")
    print(f"  Min: {pii_logits.min().item():.4f}")
    print(f"  Max: {pii_logits.max().item():.4f}")
    print(f"  Mean: {pii_logits.mean().item():.4f}")
    print(f"  Std: {pii_logits.std().item():.4f}")

    # Check per-class logit means
    print("\n  Per-class mean logits (top 10 non-O):")
    class_means = pii_logits.mean(dim=0)
    sorted_indices = torch.argsort(class_means, descending=True)

    for idx in sorted_indices[:10]:
        label = pii_id2label.get(idx.item(), f"UNK-{idx}")
        mean_logit = class_means[idx].item()
        print(f"    {label:<20}: {mean_logit:.4f}")


def diagnose_onnx_outputs(onnx_path: str, tokenizer, text: str, pii_id2label: dict):
    """Diagnose ONNX model outputs."""
    print("\n" + "=" * 80)
    print("ONNX MODEL OUTPUT DIAGNOSIS")
    print("=" * 80)

    import onnxruntime as ort

    onnx_path = Path(onnx_path)
    model_file = onnx_path / "model_quantized.onnx"

    if not model_file.exists():
        model_file = onnx_path / "model.onnx"

    if not model_file.exists():
        print(f"ERROR: No ONNX model found at {onnx_path}")
        return

    print(f"Loading ONNX model from: {model_file}")

    session = ort.InferenceSession(str(model_file), providers=["CPUExecutionProvider"])

    # Tokenize
    encoding = tokenizer(
        text,
        return_tensors="np",
        return_offsets_mapping=True,
        truncation=True,
        max_length=512,
    )

    # Run inference
    outputs = session.run(
        None,
        {
            "input_ids": encoding["input_ids"].astype(np.int64),
            "attention_mask": encoding["attention_mask"].astype(np.int64),
        },
    )

    pii_logits = outputs[0][0]  # (seq_len, num_labels)

    # Compute probabilities
    exp_logits = np.exp(pii_logits - np.max(pii_logits, axis=-1, keepdims=True))
    probs = exp_logits / np.sum(exp_logits, axis=-1, keepdims=True)
    predictions = np.argmax(probs, axis=-1)
    confidences = np.max(probs, axis=-1)

    tokens = tokenizer.convert_ids_to_tokens(encoding["input_ids"][0])

    print("\n" + "-" * 80)
    print("ONNX PREDICTIONS (first 30 tokens):")
    print("-" * 80)
    print(f"{'Idx':<5} {'Token':<15} {'Pred':<15} {'Conf':<8}")
    print("-" * 80)

    non_o_count = 0
    for i in range(min(30, len(tokens))):
        token = tokens[i]
        pred_id = predictions[i]
        pred_label = pii_id2label.get(pred_id, f"UNK-{pred_id}")
        conf = confidences[i]

        if pred_label != "O":
            non_o_count += 1
            marker = " <-- NON-O"
        else:
            marker = ""

        print(f"{i:<5} {token:<15} {pred_label:<15} {conf:<8.3f}{marker}")

    print(f"\nTotal non-O predictions: {non_o_count} / {len(tokens)}")

    # Logit statistics
    print("\n" + "-" * 80)
    print("ONNX LOGIT STATISTICS:")
    print("-" * 80)

    print(f"  Shape: {pii_logits.shape}")
    print(f"  Min: {pii_logits.min():.4f}")
    print(f"  Max: {pii_logits.max():.4f}")
    print(f"  Mean: {pii_logits.mean():.4f}")
    print(f"  Std: {pii_logits.std():.4f}")


def main():
    parser = argparse.ArgumentParser(description="Diagnose ModernBERT inference issues")
    parser.add_argument(
        "--trained-model",
        type=str,
        default="./model/trained",
        help="Path to trained model directory",
    )
    parser.add_argument(
        "--quantized-model",
        type=str,
        default="./model/quantized",
        help="Path to quantized ONNX model directory",
    )
    parser.add_argument(
        "--text",
        type=str,
        default="John Smith's email is john.smith@example.com and phone is 555-1234.",
        help="Test text to analyze",
    )
    args = parser.parse_args()

    # Privacy mask for the default test text
    privacy_mask = [
        {"value": "John", "label": "FIRSTNAME"},
        {"value": "Smith", "label": "SURNAME"},
        {"value": "john.smith@example.com", "label": "EMAIL"},
        {"value": "555-1234", "label": "PHONENUMBER"},
    ]

    print("\n" + "=" * 80)
    print("MODERNBERT PII DETECTION - INFERENCE DIAGNOSIS")
    print("=" * 80)
    print(f"\nTest text: {args.text}")
    print(f"Trained model: {args.trained_model}")
    print(f"Quantized model: {args.quantized_model}")

    # Load tokenizer
    tokenizer_path = (
        args.quantized_model
        if Path(args.quantized_model).exists()
        else args.trained_model
    )
    print(f"\nLoading tokenizer from: {tokenizer_path}")
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)

    # 1. Diagnose tokenization
    diagnose_tokenization(tokenizer, args.text, privacy_mask)

    # 2. Diagnose PyTorch model (if weights exist)
    trained_path = Path(args.trained_model)
    if (trained_path / "model.safetensors").exists() or (
        trained_path / "pytorch_model.bin"
    ).exists():
        diagnose_model_outputs(args.trained_model, tokenizer, args.text)
    else:
        print("\n" + "=" * 80)
        print("PYTORCH MODEL - SKIPPED (no weights found)")
        print("=" * 80)
        print(f"  No model weights found at: {args.trained_model}")
        print("  Expected: model.safetensors or pytorch_model.bin")

    # 3. Diagnose ONNX model (if exists)
    quantized_path = Path(args.quantized_model)
    if quantized_path.exists():
        # Load label mappings
        mappings_path = quantized_path / "label_mappings.json"
        if mappings_path.exists():
            with open(mappings_path) as f:
                mappings = json.load(f)
            pii_id2label = {
                int(k): v for k, v in mappings["pii"]["id2label"].items() if k != "-100"
            }
            diagnose_onnx_outputs(
                args.quantized_model, tokenizer, args.text, pii_id2label
            )
        else:
            print(f"\nNo label mappings found at: {mappings_path}")
    else:
        print(f"\nQuantized model path not found: {args.quantized_model}")

    print("\n" + "=" * 80)
    print("DIAGNOSIS COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    main()
