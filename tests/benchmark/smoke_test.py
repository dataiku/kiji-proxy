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

from tests.benchmark.run import (
    DEFAULT_ENTITY_CONFIDENCE_THRESHOLD,
    OnnxPIIModel,
)

TESTS = [
    (
        "My name is John Smith and my email is john.smith@email.com.",
        {
            "FIRSTNAME": ["John"],
            "SURNAME": ["Smith"],
            "EMAIL": ["john.smith@email.com"],
        },
    ),
    (
        "Call Sarah Johnson at 555-123-4567. She was born on March 15, 1985.",
        {
            "FIRSTNAME": ["Sarah"],
            "SURNAME": ["Johnson"],
            "PHONENUMBER": ["555-123-4567"],
        },
    ),
    (
        "I live at 123 Main Street, Springfield, IL 62701.",
        {
            "STREET": ["Main Street"],
            "CITY": ["Springfield"],
            "STATE": ["IL"],
            "ZIP": ["62701"],
        },
    ),
]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-path", default="./model/quantized")
    ap.add_argument(
        "--confidence-threshold",
        type=float,
        default=DEFAULT_ENTITY_CONFIDENCE_THRESHOLD,
        help="Minimum token confidence before treating a label as O.",
    )
    args = ap.parse_args()

    model_dir = Path(args.model_path)
    onnx_file = model_dir / "model.onnx"
    if not onnx_file.exists():
        onnx_file = model_dir / "model_quantized.onnx"

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

    import json

    with (model_dir / "label_mappings.json").open() as f:
        mappings = json.load(f)
    id2label = {int(k): v for k, v in mappings["pii"]["id2label"].items()}
    print(f"Labels:          {len(id2label)}")

    model = OnnxPIIModel(
        str(model_dir),
        entity_confidence_threshold=args.confidence_threshold,
    )
    print(f"CRF decoding:    {'enabled' if model.uses_crf else 'disabled'}")

    print(f"\n{'=' * 60}")
    passed = 0
    failed = 0

    for text, expected in TESTS:
        spans = model.predict(text)
        detected_labels = {label for _, _, label in spans}

        print(f"\nText: {text[:70]}...")
        print(f"  Expected labels: {sorted(expected.keys())}")
        print(f"  Detected labels: {sorted(detected_labels)}")
        for start, end, label in spans:
            print(f"    [{label:<20s}] {text[start:end]!r}")

        missing = set(expected.keys()) - detected_labels
        if missing:
            print(f"  FAIL - missing: {sorted(missing)}")
            failed += 1
        else:
            print("  PASS")
            passed += 1

    print(f"\n{'=' * 60}")
    print(f"Results: {passed} passed, {failed} failed out of {len(TESTS)}")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
