"""
Debug script to trace the full pipeline from training sample to model prediction.

This helps identify where the disconnect between training (97% acc) and inference (failing) occurs.
"""

import json
import sys
from pathlib import Path

import torch
from transformers import AutoTokenizer

# Add paths
src_dir = Path(__file__).parent
project_root = src_dir.parent.parent
sys.path.insert(0, str(src_dir))
sys.path.insert(0, str(project_root))

from model import PIIDetectionModel
from model.dataset.label_utils import LabelUtils
from model.dataset.tokenization import TokenizationProcessor


def debug_training_sample():
    """Debug a training sample to see what the model actually sees during training."""
    print("=" * 80)
    print("DEBUG: Training Sample Pipeline")
    print("=" * 80)

    # Load a real training sample
    samples_dir = Path("model/dataset/training_samples")
    if not samples_dir.exists():
        print(f"Training samples dir not found: {samples_dir}")
        return

    json_files = list(samples_dir.glob("*.json"))
    if not json_files:
        print("No training samples found")
        return

    # Load first sample
    sample_file = json_files[0]
    print(f"\n1. Loading sample: {sample_file.name}")

    with open(sample_file) as f:
        ls_sample = json.load(f)

    # Extract text and annotations
    text = ls_sample.get("data", {}).get("text", "")
    print(f"\n2. Text (first 200 chars): {text[:200]}...")

    # Get annotations
    result = None
    if ls_sample.get("annotations") and len(ls_sample["annotations"]) > 0:
        result = ls_sample["annotations"][0].get("result", [])
    elif ls_sample.get("predictions") and len(ls_sample["predictions"]) > 0:
        result = ls_sample["predictions"][0].get("result", [])

    print(f"\n3. Number of annotations: {len(result) if result else 0}")

    # Parse entities
    entities = []
    for item in result or []:
        if "value" in item:
            value = item.get("value", {})
            labels = value.get("labels", [])
            if labels:
                entities.append(
                    {
                        "text": value.get("text", ""),
                        "label": labels[0],
                        "start": value.get("start", 0),
                        "end": value.get("end", 0),
                    }
                )

    print(f"\n4. Parsed entities ({len(entities)}):")
    for e in entities[:10]:
        print(f"   [{e['label']}] '{e['text']}' at {e['start']}-{e['end']}")
    if len(entities) > 10:
        print(f"   ... and {len(entities) - 10} more")

    # Build privacy_mask (what preprocessing creates)
    privacy_mask = []
    seen = set()
    for e in entities:
        if e["label"] not in ("PRONOUN", "REFERENCE") and e["text"] not in seen:
            privacy_mask.append({"value": e["text"], "label": e["label"]})
            seen.add(e["text"])

    print(f"\n5. Privacy mask ({len(privacy_mask)} items):")
    for pm in privacy_mask[:10]:
        print(f"   [{pm['label']}] '{pm['value']}'")

    # Now tokenize like training does
    print("\n6. Tokenization:")
    tokenizer = AutoTokenizer.from_pretrained("answerdotai/ModernBERT-base")
    label2id, id2label = LabelUtils.create_standard_label2id()

    print(f"   Label2id has {len(label2id)} labels")
    print(f"   Sample labels: {list(label2id.keys())[:10]}")

    processor = TokenizationProcessor(tokenizer, label2id, id2label, max_length=512)

    # Tokenize a short portion of text for debugging
    short_text = text[:500] if len(text) > 500 else text
    short_mask = [pm for pm in privacy_mask if pm["value"] in short_text]

    print(
        f"\n7. Tokenizing short text ({len(short_text)} chars, {len(short_mask)} entities)..."
    )

    sample = processor.create_pii_sample(short_text, short_mask)

    tokens = tokenizer.convert_ids_to_tokens(sample["input_ids"])
    labels = sample["labels"]

    print(f"\n8. Tokenized result:")
    print(f"   Tokens: {len(tokens)}")
    print(f"   Labels: {len(labels)}")

    # Count label distribution
    label_counts = {}
    for lid in labels:
        if lid == -100:
            label_name = "IGNORE"
        else:
            label_name = id2label.get(lid, f"UNK-{lid}")
        label_counts[label_name] = label_counts.get(label_name, 0) + 1

    print(f"\n9. Label distribution:")
    for label, count in sorted(label_counts.items(), key=lambda x: -x[1]):
        pct = count / len(labels) * 100
        print(f"   {label:<20}: {count:4d} ({pct:5.1f}%)")

    # Show token-label pairs for entities
    print(f"\n10. Token-label pairs (non-O labels):")
    count = 0
    for i, (token, lid) in enumerate(zip(tokens, labels)):
        if lid != -100 and lid != 0:  # Not IGNORE and not O
            label_name = id2label.get(lid, f"UNK-{lid}")
            print(f"   [{i:3d}] '{token:<20}' -> {label_name}")
            count += 1
            if count >= 30:
                print("   ... (truncated)")
                break

    if count == 0:
        print("   NO NON-O LABELS FOUND! This is the problem.")
        print("\n   Let's check why...")

        # Debug: find where entities should be
        print("\n   Entity positions in text:")
        for pm in short_mask[:5]:
            pos = short_text.find(pm["value"])
            if pos >= 0:
                print(f"   '{pm['value']}' at char {pos}-{pos + len(pm['value'])}")
            else:
                print(f"   '{pm['value']}' NOT FOUND in text!")


def debug_inference_vs_training():
    """Compare inference tokenization with training tokenization."""
    print("\n" + "=" * 80)
    print("DEBUG: Inference vs Training Tokenization")
    print("=" * 80)

    # Simple test text
    test_text = "My name is John Smith and my email is john.smith@example.com."
    test_mask = [
        {"value": "John", "label": "FIRSTNAME"},
        {"value": "Smith", "label": "SURNAME"},
        {"value": "john.smith@example.com", "label": "EMAIL"},
    ]

    print(f"\n1. Test text: {test_text}")
    print(f"2. Privacy mask: {test_mask}")

    tokenizer = AutoTokenizer.from_pretrained("answerdotai/ModernBERT-base")
    label2id, id2label = LabelUtils.create_standard_label2id()

    # Training tokenization
    print("\n3. TRAINING tokenization (using TokenizationProcessor):")
    processor = TokenizationProcessor(tokenizer, label2id, id2label)
    train_sample = processor.create_pii_sample(test_text, test_mask)

    train_tokens = tokenizer.convert_ids_to_tokens(train_sample["input_ids"])
    train_labels = train_sample["labels"]

    print(f"   Tokens: {len(train_tokens)}")
    for i, (token, lid) in enumerate(zip(train_tokens, train_labels)):
        label_name = "IGNORE" if lid == -100 else id2label.get(lid, f"UNK-{lid}")
        marker = "***" if lid not in (-100, 0) else ""
        print(f"   [{i:2d}] '{token:<15}' -> {label_name:<15} {marker}")

    # Inference tokenization
    print("\n4. INFERENCE tokenization (direct tokenizer call):")
    inf_encoding = tokenizer(
        test_text,
        return_tensors="pt",
        return_offsets_mapping=True,
        truncation=True,
        max_length=512,
    )

    inf_tokens = tokenizer.convert_ids_to_tokens(inf_encoding["input_ids"][0])
    inf_offsets = inf_encoding["offset_mapping"][0].tolist()

    print(f"   Tokens: {len(inf_tokens)}")
    for i, (token, offset) in enumerate(zip(inf_tokens, inf_offsets)):
        start, end = offset
        if start == 0 and end == 0:
            span = "[SPECIAL]"
        else:
            span = repr(test_text[start:end])
        print(f"   [{i:2d}] '{token:<15}' offset={offset} -> {span}")

    # Check if tokenizations match
    print("\n5. Comparison:")
    if train_sample["input_ids"] == inf_encoding["input_ids"][0].tolist():
        print("   ✅ Input IDs MATCH")
    else:
        print("   ❌ Input IDs DIFFER!")
        print(f"   Training: {train_sample['input_ids'][:10]}...")
        print(f"   Inference: {inf_encoding['input_ids'][0].tolist()[:10]}...")


def debug_model_prediction():
    """Debug what the trained model actually predicts."""
    print("\n" + "=" * 80)
    print("DEBUG: Model Predictions")
    print("=" * 80)

    model_path = Path("model/trained")
    if not model_path.exists():
        print(f"Model not found at {model_path}")
        return

    # Load model
    print("\n1. Loading model...")

    mappings_path = model_path / "label_mappings.json"
    with open(mappings_path) as f:
        mappings = json.load(f)

    label2id = mappings["label2id"]
    id2label = {int(k): v for k, v in mappings["id2label"].items()}

    print(f"   Labels: {len(label2id)}")

    tokenizer = AutoTokenizer.from_pretrained(model_path)

    model = PIIDetectionModel(
        model_name="answerdotai/ModernBERT-base",
        num_labels=len(label2id),
        id2label=id2label,
    )

    # Load weights
    from safetensors import safe_open

    weights_path = model_path / "model.safetensors"
    if not weights_path.exists():
        weights_path = model_path / "pytorch_model.bin"

    if weights_path.exists():
        if weights_path.suffix == ".safetensors":
            state_dict = {}
            with safe_open(weights_path, framework="pt", device="cpu") as f:
                for key in f.keys():
                    state_dict[key] = f.get_tensor(key)
        else:
            state_dict = torch.load(
                weights_path, map_location="cpu", weights_only=False
            )

        # Fix prefix
        if any(k.startswith("model.") for k in state_dict.keys()):
            state_dict = {
                (k[6:] if k.startswith("model.") else k): v
                for k, v in state_dict.items()
            }

        model.load_state_dict(state_dict, strict=False)
        print("   Weights loaded")

    model.eval()

    # Test prediction
    test_text = "My name is John Smith and my email is john.smith@example.com."
    print(f"\n2. Test text: {test_text}")

    inputs = tokenizer(test_text, return_tensors="pt", return_offsets_mapping=True)
    offset_mapping = inputs.pop("offset_mapping")

    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs["logits"][0]
        probs = torch.softmax(logits, dim=-1)
        predictions = torch.argmax(logits, dim=-1)

    tokens = tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])

    print("\n3. Predictions:")
    print(f"   {'Idx':<4} {'Token':<15} {'Pred':<15} {'Conf':<8} {'Top-3'}")
    print("   " + "-" * 70)

    for i, (token, pred_id) in enumerate(zip(tokens, predictions)):
        pred_label = id2label.get(pred_id.item(), f"UNK-{pred_id.item()}")
        conf = probs[i][pred_id].item()

        # Top 3
        top3_idx = torch.topk(probs[i], 3).indices.tolist()
        top3_probs = torch.topk(probs[i], 3).values.tolist()
        top3 = ", ".join(
            [
                f"{id2label.get(idx, 'UNK')}:{p:.2f}"
                for idx, p in zip(top3_idx, top3_probs)
            ]
        )

        marker = "***" if pred_label != "O" else ""
        print(
            f"   [{i:2d}] '{token:<15}' {pred_label:<15} {conf:.4f}   {top3} {marker}"
        )

    # Check label distribution in predictions
    print("\n4. Prediction distribution:")
    pred_counts = {}
    for p in predictions.tolist():
        label = id2label.get(p, f"UNK-{p}")
        pred_counts[label] = pred_counts.get(label, 0) + 1

    for label, count in sorted(pred_counts.items(), key=lambda x: -x[1]):
        print(f"   {label:<20}: {count}")


if __name__ == "__main__":
    debug_training_sample()
    debug_inference_vs_training()
    debug_model_prediction()
