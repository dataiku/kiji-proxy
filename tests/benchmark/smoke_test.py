"""Quick smoke test to verify the ONNX model detects basic PII.

Usage:
    uv run python tests/benchmark/smoke_test.py
    uv run python tests/benchmark/smoke_test.py --model-path ./model/quantized
"""

from __future__ import annotations

import argparse
import hashlib
import os
import sys
from pathlib import Path

import numpy as np
import onnxruntime as ort
from transformers import AutoTokenizer

TESTS = [
    (
        "My name is John Smith and my email is john.smith@email.com.",
        {"FIRSTNAME": ["John"], "SURNAME": ["Smith"], "EMAIL": ["john.smith@email.com"]},
    ),
    (
        "Call Sarah Johnson at 555-123-4567. She was born on March 15, 1985.",
        {"FIRSTNAME": ["Sarah"], "SURNAME": ["Johnson"], "PHONENUMBER": ["555-123-4567"]},
    ),
    (
        "I live at 123 Main Street, Springfield, IL 62701.",
        {"STREET": ["Main Street"], "CITY": ["Springfield"], "STATE": ["IL"], "ZIP": ["62701"]},
    ),
]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-path", default="./model/quantized")
    args = ap.parse_args()

    model_dir = Path(args.model_path)
    onnx_file = model_dir / "model_quantized.onnx"
    if not onnx_file.exists():
        onnx_file = model_dir / "model.onnx"

    # Print model metadata
    print(f"Model directory: {model_dir.resolve()}")
    print(f"ONNX file:       {onnx_file.name}")
    size_mb = onnx_file.stat().st_size / (1024 * 1024)
    print(f"ONNX size:       {size_mb:.1f} MB")
    mod_time = os.path.getmtime(onnx_file)
    from datetime import datetime
    print(f"Last modified:   {datetime.fromtimestamp(mod_time)}")

    # Hash full file to fingerprint the model
    h = hashlib.sha256()
    with open(onnx_file, "rb") as f:
        while chunk := f.read(8192):
            h.update(chunk)
    print(f"SHA256:          {h.hexdigest()[:16]}...")

    # Load model
    import json
    with (model_dir / "label_mappings.json").open() as f:
        mappings = json.load(f)
    id2label = {int(k): v for k, v in mappings["pii"]["id2label"].items()}
    print(f"Labels:          {len(id2label)}")

    tokenizer = AutoTokenizer.from_pretrained(str(model_dir))
    session = ort.InferenceSession(
        str(onnx_file),
        providers=["CPUExecutionProvider"],
    )

    print(f"\n{'='*60}")
    passed = 0
    failed = 0

    for text, expected in TESTS:
        inputs = tokenizer(text, return_tensors="np", truncation=True, max_length=512)
        ort_inputs = {
            "input_ids": inputs["input_ids"],
            "attention_mask": inputs["attention_mask"],
        }
        logits = session.run(None, ort_inputs)[0]
        preds = np.argmax(logits, axis=-1)[0]

        # Collect predicted labels (ignore O and special tokens)
        detected_labels = set()
        for p in preds:
            label = id2label.get(int(p), "O")
            if label not in ("O", "IGNORE"):
                base = label[2:] if label.startswith(("B-", "I-")) else label
                detected_labels.add(base)

        print(f"\nText: {text[:70]}...")
        print(f"  Expected labels: {sorted(expected.keys())}")
        print(f"  Detected labels: {sorted(detected_labels)}")

        missing = set(expected.keys()) - detected_labels
        if missing:
            print(f"  FAIL - missing: {sorted(missing)}")
            failed += 1
        else:
            print(f"  PASS")
            passed += 1

    print(f"\n{'='*60}")
    print(f"Results: {passed} passed, {failed} failed out of {len(TESTS)}")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
