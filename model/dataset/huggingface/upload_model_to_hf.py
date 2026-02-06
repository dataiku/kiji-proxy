#!/usr/bin/env python3
"""Upload trained or quantized PII detection model to HuggingFace Hub with a model card."""

import json
import os
import sys
from pathlib import Path

from huggingface_hub import create_repo, upload_folder

# Add project root to path for imports
project_root = Path(__file__).parent.parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from model.dataset.label_utils import LabelUtils

# Files to upload per variant
_TRAINED_MODEL_FILES = [
    "model.safetensors",
    "label_mappings.json",
    "tokenizer_config.json",
    "tokenizer.json",
    "vocab.txt",
    "special_tokens_map.json",
]

_QUANTIZED_MODEL_FILES = [
    "model_quantized.onnx",
    "model.onnx.data",
    "ort_config.json",
    "label_mappings.json",
    "model_manifest.json",
    "tokenizer_config.json",
    "tokenizer.json",
    "vocab.txt",
    "special_tokens_map.json",
]


def _load_label_info(model_dir: str) -> tuple[int, int, list[str]]:
    """Load label mappings and return (num_pii_labels, num_coref_labels, pii_entities)."""
    mappings_path = Path(model_dir) / "label_mappings.json"
    if not mappings_path.exists():
        return 0, 0, []
    with mappings_path.open() as f:
        mappings = json.load(f)
    pii_label2id = mappings.get("pii", {}).get("label2id", {})
    num_pii_labels = len(pii_label2id)
    pii_entities = sorted({k[2:] for k in pii_label2id if k.startswith("B-")})
    coref_id2label = mappings.get("coref", {}).get("id2label", {})
    num_coref_labels = len(coref_id2label)
    return num_pii_labels, num_coref_labels, pii_entities


def _pii_label_rows(pii_entities: list[str]) -> str:
    """Build markdown table rows for PII entity labels."""
    rows = ""
    for label in pii_entities:
        desc = LabelUtils.LABEL_DESCRIPTIONS.get(label, {})
        name = desc.get("name", label)
        rows += f"| `{label}` | {name} |\n"
    return rows


def _yaml_language_block() -> str:
    """Build YAML language list for frontmatter."""
    lang_codes = sorted(LabelUtils.LANGUAGE_CODES.values())
    return "\n".join(f"- {code}" for code in lang_codes)


def _generate_trained_model_card(
    repo_id: str,
    model_dir: str,
    base_model: str,
    dataset_repo_id: str | None,
    quantized_repo_id: str | None,
) -> str:
    """Generate a model card for the trained (SafeTensors) model."""
    num_pii_labels, num_coref_labels, pii_entities = _load_label_info(model_dir)

    model_file = Path(model_dir) / "model.safetensors"
    model_size_mb = (
        model_file.stat().st_size / (1024 * 1024) if model_file.exists() else 0
    )

    label_rows = _pii_label_rows(pii_entities)
    yaml_lang_block = _yaml_language_block()

    dataset_section = ""
    if dataset_repo_id:
        dataset_section = f"""
## Training Data

Trained on the [{dataset_repo_id}](https://huggingface.co/datasets/{dataset_repo_id}) dataset — a synthetic multilingual PII dataset with entity annotations and coreference resolution.
"""

    derived_section = ""
    if quantized_repo_id:
        derived_section = f"""
## Derived Models

| Variant | Format | Repository |
|---------|--------|------------|
| Quantized (INT8) | ONNX | [{quantized_repo_id}](https://huggingface.co/{quantized_repo_id}) |
"""

    card = f"""---
language:
{yaml_lang_block}
license: apache-2.0
library_name: transformers
pipeline_tag: token-classification
tags:
- pii
- privacy
- ner
- coreference-resolution
- distilbert
- multi-task
base_model: {base_model}
---

# Kiji PII Detection Model

Multi-task DistilBERT model for detecting Personally Identifiable Information (PII) in text with coreference resolution. Fine-tuned from [`{base_model}`](https://huggingface.co/{base_model}).

## Model Summary

| | |
|---|---|
| **Base model** | [{base_model}](https://huggingface.co/{base_model}) |
| **Architecture** | Shared DistilBERT encoder + two linear classification heads |
| **Parameters** | ~66M |
| **Model size** | {model_size_mb:.0f} MB (SafeTensors) |
| **Tasks** | PII token classification ({num_pii_labels} labels) + coreference detection ({num_coref_labels} labels) |
| **PII entity types** | {len(pii_entities)} |
| **Max sequence length** | 512 tokens |

## Architecture

```
Input (input_ids, attention_mask)
        |
  DistilBERT Encoder (shared, hidden_size=768)
        |
   +----+----+
   |         |
PII Head  Coref Head
(768->{num_pii_labels})  (768->{num_coref_labels})
```

The model uses multi-task learning: a shared DistilBERT encoder feeds into two independent linear classification heads. Both tasks are trained simultaneously with equal loss weighting, which acts as regularization and improves PII detection generalization.

## Usage

```python
import torch
from transformers import AutoTokenizer

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained("{repo_id}")

# The model uses a custom MultiTaskPIIDetectionModel architecture.
# Load weights manually:
from safetensors.torch import load_file
weights = load_file("{repo_id}/model.safetensors")  # or local path

# Tokenize
text = "Contact John Smith at john.smith@example.com or call +1-555-123-4567."
inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512)

# See the label_mappings.json file for PII label definitions
```

## PII Labels (BIO tagging)

The model uses BIO tagging with {len(pii_entities)} entity types:

| Label | Description |
|-------|-------------|
{label_rows}

Each entity type has `B-` (beginning) and `I-` (inside) variants, plus `O` for non-PII tokens.

## Coreference Labels

| Label | Description |
|-------|-------------|
| `NO_COREF` | Token is not part of a coreference cluster |
| `CLUSTER_0`-`CLUSTER_3` | Token belongs to coreference cluster 0-3 |

## Training

| | |
|---|---|
| **Epochs** | 15 (with early stopping) |
| **Batch size** | 16 |
| **Learning rate** | 3e-5 |
| **Weight decay** | 0.01 |
| **Warmup steps** | 200 |
| **Early stopping** | patience=3, threshold=1% |
| **Loss** | Multi-task: PII cross-entropy + coreference cross-entropy (equal weights) |
| **Optimizer** | AdamW |
| **Metric** | Weighted F1 (PII task) |
{dataset_section}{derived_section}
## Limitations

- Trained on **synthetically generated** data — may not generalize to all real-world text
- Coreference head supports up to 4 clusters per sequence
- Optimized for the 6 languages in the training data ({", ".join(LabelUtils.LANGUAGE_CODES.keys())})
- Max sequence length is 512 tokens
"""
    return card.strip() + "\n"


def _generate_quantized_model_card(
    repo_id: str,
    model_dir: str,
    trained_repo_id: str | None,
    dataset_repo_id: str | None,
) -> str:
    """Generate a model card for the quantized (ONNX INT8) model."""
    num_pii_labels, num_coref_labels, pii_entities = _load_label_info(model_dir)
    label_rows = _pii_label_rows(pii_entities)
    yaml_lang_block = _yaml_language_block()

    # Compute file sizes for the manifest table
    model_path = Path(model_dir)
    file_rows = ""
    for f in _QUANTIZED_MODEL_FILES:
        fpath = model_path / f
        if fpath.exists():
            size = fpath.stat().st_size
            if size > 1024 * 1024:
                size_str = f"{size / (1024 * 1024):.1f} MB"
            else:
                size_str = f"{size / 1024:.1f} KB"
            file_rows += f"| `{f}` | {size_str} |\n"

    # Load quantization config for details
    ort_config_path = model_path / "ort_config.json"
    quant_details = ""
    if ort_config_path.exists():
        with ort_config_path.open() as f:
            ort_config = json.load(f)
        q = ort_config.get("quantization", {})
        operators = ", ".join(q.get("operators_to_quantize", []))
        quant_details = f"""
## Quantization Details

| | |
|---|---|
| **Method** | Dynamic quantization (ONNX Runtime / Optimum) |
| **Weights** | {q.get("weights_dtype", "QInt8")} (symmetric, per-channel) |
| **Activations** | {q.get("activations_dtype", "QUInt8")} (asymmetric, per-tensor) |
| **Mode** | {q.get("mode", "IntegerOps")} |
| **Format** | {q.get("format", "QOperator")} |
| **Operators quantized** | {operators} |
"""

    # Base model reference (the trained model)
    base_model_value = trained_repo_id or "distilbert-base-cased"

    trained_section = ""
    if trained_repo_id:
        trained_section = f"""
## Source Model

This is a quantized version of [{trained_repo_id}](https://huggingface.co/{trained_repo_id}) — a multi-task DistilBERT model fine-tuned for PII detection with coreference resolution.
"""

    dataset_section = ""
    if dataset_repo_id:
        dataset_section = f"""
## Training Data

The source model was trained on the [{dataset_repo_id}](https://huggingface.co/datasets/{dataset_repo_id}) dataset — a synthetic multilingual PII dataset with entity annotations and coreference resolution.
"""

    lineage_section = ""
    if dataset_repo_id or trained_repo_id:
        lineage_rows = ""
        if dataset_repo_id:
            lineage_rows += f"| Dataset | [{dataset_repo_id}](https://huggingface.co/datasets/{dataset_repo_id}) |\n"
        if trained_repo_id:
            lineage_rows += f"| Trained model | [{trained_repo_id}](https://huggingface.co/{trained_repo_id}) |\n"
        lineage_rows += f"| **Quantized model** | **{repo_id}** (this repo) |\n"
        lineage_section = f"""
## Lineage

| Stage | Repository |
|-------|------------|
{lineage_rows}"""

    card = f"""---
language:
{yaml_lang_block}
license: apache-2.0
library_name: onnx
pipeline_tag: token-classification
tags:
- pii
- privacy
- ner
- coreference-resolution
- distilbert
- multi-task
- onnx
- quantized
- int8
base_model: {base_model_value}
---

# Kiji PII Detection Model (ONNX Quantized)

INT8-quantized ONNX version of the Kiji PII detection model for efficient CPU inference. Detects Personally Identifiable Information (PII) in text with coreference resolution.
{trained_section}
## Model Summary

| | |
|---|---|
| **Format** | ONNX (INT8 quantized) |
| **Architecture** | Shared DistilBERT encoder + two classification heads |
| **Tasks** | PII token classification ({num_pii_labels} labels) + coreference detection ({num_coref_labels} labels) |
| **PII entity types** | {len(pii_entities)} |
| **Max sequence length** | 512 tokens |
| **Runtime** | ONNX Runtime |

## Files

| File | Size |
|------|------|
{file_rows}{quant_details}
## Usage

```python
import numpy as np
from onnxruntime import InferenceSession
from transformers import AutoTokenizer

# Load tokenizer and model
tokenizer = AutoTokenizer.from_pretrained("{repo_id}")
session = InferenceSession("{repo_id}/model_quantized.onnx")  # or local path

# Tokenize
text = "Contact John Smith at john.smith@example.com or call +1-555-123-4567."
inputs = tokenizer(text, return_tensors="np", truncation=True, max_length=512)

# Run inference
outputs = session.run(None, dict(inputs))
pii_logits, coref_logits = outputs  # (1, seq_len, {num_pii_labels}), (1, seq_len, {num_coref_labels})

# Decode PII predictions
pii_predictions = np.argmax(pii_logits, axis=-1)[0]

# See label_mappings.json for label ID -> label name mapping
```

## PII Labels (BIO tagging)

The model uses BIO tagging with {len(pii_entities)} entity types:

| Label | Description |
|-------|-------------|
{label_rows}

Each entity type has `B-` (beginning) and `I-` (inside) variants, plus `O` for non-PII tokens.

## Coreference Labels

| Label | Description |
|-------|-------------|
| `NO_COREF` | Token is not part of a coreference cluster |
| `CLUSTER_0`-`CLUSTER_3` | Token belongs to coreference cluster 0-3 |
{dataset_section}{lineage_section}
## Limitations

- Trained on **synthetically generated** data — may not generalize to all real-world text
- Coreference head supports up to 4 clusters per sequence
- Optimized for the 6 languages in the training data ({", ".join(LabelUtils.LANGUAGE_CODES.keys())})
- Max sequence length is 512 tokens
- Quantization may slightly reduce accuracy compared to the full-precision model
"""
    return card.strip() + "\n"


def upload_model_to_huggingface(
    variant: str = "trained",
    model_dir: str | None = None,
    repo_id: str | None = None,
    private: bool = True,
    base_model: str = "distilbert-base-cased",
    dataset_repo_id: str | None = None,
    trained_repo_id: str | None = None,
    quantized_repo_id: str | None = None,
):
    """
    Upload a PII detection model to HuggingFace Hub.

    Supports both trained (SafeTensors) and quantized (ONNX) variants.

    Args:
        variant: Model variant to upload ("trained" or "quantized")
        model_dir: Path to the model directory (defaults based on variant)
        repo_id: HuggingFace repo ID (e.g., "username/kiji-pii-model")
        private: Whether to make the repo private
        base_model: Name of the base model used for fine-tuning
        dataset_repo_id: Optional HF dataset repo ID to link in the model card
        trained_repo_id: Optional HF trained model repo ID (for quantized variant lineage)
        quantized_repo_id: Optional HF quantized model repo ID (for trained variant cross-link)
    """
    token = os.environ.get("HF_TOKEN")
    if not token:
        raise ValueError("HF_TOKEN environment variable not set")

    if not repo_id:
        raise ValueError("repo_id is required (e.g., 'username/kiji-pii-model')")

    if variant not in ("trained", "quantized"):
        raise ValueError(
            f"Unknown variant: {variant}. Must be 'trained' or 'quantized'"
        )

    # Default model directories
    if model_dir is None:
        model_dir = "model/trained" if variant == "trained" else "model/quantized"

    model_path = Path(model_dir)
    if not model_path.exists():
        raise ValueError(f"Model directory not found: {model_dir}")

    # Select files for this variant
    model_files = (
        _TRAINED_MODEL_FILES if variant == "trained" else _QUANTIZED_MODEL_FILES
    )

    # Verify required files exist
    missing = [f for f in model_files if not (model_path / f).exists()]
    if missing:
        raise ValueError(f"Missing required model files: {missing}")

    print(f"Uploading {variant} model from {model_dir}")
    for f in model_files:
        size = (model_path / f).stat().st_size
        print(
            f"  {f}: {size / (1024 * 1024):.1f} MB"
            if size > 1024 * 1024
            else f"  {f}: {size / 1024:.1f} KB"
        )

    # Generate model card for the appropriate variant
    print("Generating model card...")
    if variant == "trained":
        card = _generate_trained_model_card(
            repo_id=repo_id,
            model_dir=model_dir,
            base_model=base_model,
            dataset_repo_id=dataset_repo_id,
            quantized_repo_id=quantized_repo_id,
        )
    else:
        card = _generate_quantized_model_card(
            repo_id=repo_id,
            model_dir=model_dir,
            trained_repo_id=trained_repo_id,
            dataset_repo_id=dataset_repo_id,
        )

    # Write model card to the model directory temporarily
    readme_path = model_path / "README.md"
    readme_existed = readme_path.exists()
    readme_path.write_text(card)

    # Create the repo if it doesn't exist yet
    allow_patterns = model_files + ["README.md"]
    print(f"\nPushing to {repo_id} (private={private})...")
    create_repo(
        repo_id=repo_id,
        repo_type="model",
        token=token,
        private=private,
        exist_ok=True,
    )
    upload_folder(
        folder_path=model_dir,
        repo_id=repo_id,
        repo_type="model",
        token=token,
        create_pr=False,
        allow_patterns=allow_patterns,
    )

    # Clean up the temporary README if it didn't exist before
    if not readme_existed:
        readme_path.unlink()

    print(f"Done! Model available at: https://huggingface.co/{repo_id}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Upload trained or quantized PII model to HuggingFace Hub"
    )
    parser.add_argument(
        "--variant",
        choices=["trained", "quantized"],
        default="trained",
        help="Model variant to upload (default: trained)",
    )
    parser.add_argument(
        "--model-dir",
        default=None,
        help="Directory containing model files (default: model/trained or model/quantized)",
    )
    parser.add_argument(
        "--repo-id",
        required=True,
        help="HuggingFace repo ID (e.g., 'username/kiji-pii-model')",
    )
    parser.add_argument(
        "--public",
        action="store_true",
        help="Make the model public (default: private)",
    )
    parser.add_argument(
        "--base-model",
        default="distilbert-base-cased",
        help="Base model name for fine-tuning (default: distilbert-base-cased)",
    )
    parser.add_argument(
        "--dataset-repo-id",
        default=None,
        help="HuggingFace dataset repo ID to link in the model card",
    )
    parser.add_argument(
        "--trained-repo-id",
        default=None,
        help="HuggingFace trained model repo ID (for quantized variant lineage)",
    )
    parser.add_argument(
        "--quantized-repo-id",
        default=None,
        help="HuggingFace quantized model repo ID (for trained variant cross-link)",
    )

    args = parser.parse_args()

    upload_model_to_huggingface(
        variant=args.variant,
        model_dir=args.model_dir,
        repo_id=args.repo_id,
        private=not args.public,
        base_model=args.base_model,
        dataset_repo_id=args.dataset_repo_id,
        trained_repo_id=args.trained_repo_id,
        quantized_repo_id=args.quantized_repo_id,
    )
