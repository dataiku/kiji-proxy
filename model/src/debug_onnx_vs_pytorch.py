"""
Debug ONNX vs PyTorch inference discrepancy.

This script compares inference results between the trained PyTorch model
and the exported ONNX model to identify where the mismatch occurs.
"""

import json
import sys
from pathlib import Path

import numpy as np
import onnxruntime as ort
import torch
from safetensors import safe_open
from transformers import AutoTokenizer

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

# Also add model/src to path for local imports
sys.path.insert(0, str(Path(__file__).parent))

from model import MultiTaskPIIDetectionModel


def load_pytorch_model(model_path: str):
    """Load the PyTorch model."""
    model_path = Path(model_path)

    # Load label mappings
    with open(model_path / "label_mappings.json") as f:
        mappings = json.load(f)

    pii_label2id = mappings["pii"]["label2id"]
    pii_id2label = {int(k): v for k, v in mappings["pii"]["id2label"].items()}
    coref_id2label = {int(k): v for k, v in mappings["coref"]["id2label"].items()}

    # Create model
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

    print(f"Loading weights from: {weights_path}")

    if weights_path.suffix == ".safetensors":
        state_dict = {}
        with safe_open(weights_path, framework="pt", device="cpu") as f:
            for key in f.keys():
                state_dict[key] = f.get_tensor(key)
    else:
        state_dict = torch.load(weights_path, map_location="cpu", weights_only=False)

    # Check for prefix issues
    model_keys = set(model.state_dict().keys())
    loaded_keys = set(state_dict.keys())

    print(f"\nModel expects {len(model_keys)} keys")
    print(f"Loaded state dict has {len(loaded_keys)} keys")

    # Check if keys have 'model.' prefix
    if any(k.startswith("model.") for k in loaded_keys):
        print("\nLoaded keys have 'model.' prefix - removing...")
        state_dict = {k.replace("model.", ""): v for k, v in state_dict.items()}
        loaded_keys = set(state_dict.keys())

    # Find missing and unexpected keys
    missing = model_keys - loaded_keys
    unexpected = loaded_keys - model_keys

    if missing:
        print(f"\n⚠️  Missing keys ({len(missing)}):")
        for k in sorted(missing)[:10]:
            print(f"   - {k}")
        if len(missing) > 10:
            print(f"   ... and {len(missing) - 10} more")

    if unexpected:
        print(f"\n⚠️  Unexpected keys ({len(unexpected)}):")
        for k in sorted(unexpected)[:10]:
            print(f"   - {k}")
        if len(unexpected) > 10:
            print(f"   ... and {len(unexpected) - 10} more")

    # Load with strict=True to see all issues
    try:
        model.load_state_dict(state_dict, strict=True)
        print("\n✅ All weights loaded successfully (strict=True)")
    except RuntimeError as e:
        print(f"\n❌ Error loading weights: {e}")
        # Fall back to strict=False
        model.load_state_dict(state_dict, strict=False)
        print("   Loaded with strict=False")

    model.eval()
    return model, pii_id2label, coref_id2label


def load_onnx_model(model_path: str):
    """Load the ONNX model."""
    model_path = Path(model_path)
    onnx_path = model_path / "model_quantized.onnx"

    if not onnx_path.exists():
        onnx_path = model_path / "model.onnx"

    print(f"Loading ONNX model: {onnx_path}")
    session = ort.InferenceSession(str(onnx_path))
    return session


def compare_inference(pytorch_model, onnx_session, tokenizer, pii_id2label, text: str):
    """Compare inference between PyTorch and ONNX."""
    print(f"\n{'=' * 80}")
    print(f"Testing: {text!r}")
    print(f"{'=' * 80}")

    # Tokenize
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
    tokens = tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])

    print(f"\nTokens: {tokens}")

    # PyTorch inference
    with torch.no_grad():
        pytorch_outputs = pytorch_model(
            input_ids=inputs["input_ids"], attention_mask=inputs["attention_mask"]
        )

    pytorch_pii_logits = pytorch_outputs["pii_logits"].numpy()
    pytorch_pii_probs = torch.softmax(pytorch_outputs["pii_logits"], dim=-1).numpy()
    pytorch_pii_preds = np.argmax(pytorch_pii_logits, axis=-1)

    # ONNX inference
    onnx_inputs = {
        "input_ids": inputs["input_ids"].numpy(),
        "attention_mask": inputs["attention_mask"].numpy(),
    }
    onnx_outputs = onnx_session.run(None, onnx_inputs)
    onnx_pii_logits = onnx_outputs[0]
    onnx_pii_probs = np.exp(onnx_pii_logits) / np.sum(
        np.exp(onnx_pii_logits), axis=-1, keepdims=True
    )
    onnx_pii_preds = np.argmax(onnx_pii_logits, axis=-1)

    # Compare outputs
    print(f"\n{'─' * 40}")
    print("LOGIT COMPARISON (first 5 classes for each token):")
    print(f"{'─' * 40}")

    for i, tok in enumerate(tokens):
        pt_logits = pytorch_pii_logits[0, i, :5]
        ox_logits = onnx_pii_logits[0, i, :5]
        diff = np.abs(pt_logits - ox_logits).max()

        print(f"\nToken[{i}] {tok:20s}")
        print(f"  PyTorch: {pt_logits}")
        print(f"  ONNX:    {ox_logits}")
        print(f"  Max diff: {diff:.6f}")

    # Compare predictions
    print(f"\n{'─' * 40}")
    print("PREDICTION COMPARISON:")
    print(f"{'─' * 40}")

    for i, tok in enumerate(tokens):
        pt_pred = pytorch_pii_preds[0, i]
        ox_pred = onnx_pii_preds[0, i]
        pt_label = pii_id2label.get(pt_pred, f"UNKNOWN({pt_pred})")
        ox_label = pii_id2label.get(ox_pred, f"UNKNOWN({ox_pred})")
        pt_conf = pytorch_pii_probs[0, i, pt_pred]
        ox_conf = onnx_pii_probs[0, i, ox_pred]

        match = "✅" if pt_pred == ox_pred else "❌"
        print(
            f"{match} Token[{i:2d}] {tok:15s} | PyTorch: {pt_label:15s} ({pt_conf:.3f}) | ONNX: {ox_label:15s} ({ox_conf:.3f})"
        )

    # Overall stats
    logit_diff = np.abs(pytorch_pii_logits - onnx_pii_logits)
    print(f"\n{'─' * 40}")
    print("OVERALL STATISTICS:")
    print(f"{'─' * 40}")
    print(f"Max logit difference: {logit_diff.max():.6f}")
    print(f"Mean logit difference: {logit_diff.mean():.6f}")
    print(
        f"Prediction match rate: {(pytorch_pii_preds == onnx_pii_preds).mean() * 100:.1f}%"
    )


def main():
    trained_path = Path("model/trained")
    quantized_path = Path("model/quantized")

    print("=" * 80)
    print("ONNX vs PyTorch Inference Comparison")
    print("=" * 80)

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(quantized_path)
    print(f"Loaded tokenizer: vocab_size={tokenizer.vocab_size}")

    # Load PyTorch model
    print(f"\n{'─' * 40}")
    print("Loading PyTorch Model:")
    print(f"{'─' * 40}")
    pytorch_model, pii_id2label, coref_id2label = load_pytorch_model(trained_path)

    # Load ONNX model
    print(f"\n{'─' * 40}")
    print("Loading ONNX Model:")
    print(f"{'─' * 40}")
    onnx_session = load_onnx_model(quantized_path)

    # Test cases
    test_texts = [
        "John Smith lives at 123 Main St",
        "Contact me at john@example.com",
        "Visit https://example.com for info",
        "My SSN is 123-45-6789",
    ]

    for text in test_texts:
        compare_inference(pytorch_model, onnx_session, tokenizer, pii_id2label, text)

    print(f"\n{'=' * 80}")
    print("Comparison Complete")
    print(f"{'=' * 80}")


if __name__ == "__main__":
    main()
