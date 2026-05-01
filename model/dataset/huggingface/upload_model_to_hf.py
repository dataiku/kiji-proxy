#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.13"
# dependencies = []
# ///
"""Upload trained or quantized PII detection model to HuggingFace Hub with a model card."""

import json
import os
import struct
import sys
import tomllib
from pathlib import Path

from huggingface_hub import create_repo, upload_folder

# Add project root to path for imports
PROJECT_ROOT = Path(__file__).parent.parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from model.dataset.label_utils import LabelUtils  # noqa: E402

# Files to upload per variant. "required" must exist; "optional" is uploaded
# if present (tokenizer file naming varies between BERT-style `vocab.txt` and
# SentencePiece-style `spm.model`).
_OPTIONAL_TOKENIZER_FILES = ["vocab.txt", "spm.model", "added_tokens.json"]

_TRAINED_REQUIRED_FILES = [
    "model.safetensors",
    "config.json",
    "label_mappings.json",
    "tokenizer_config.json",
    "tokenizer.json",
    "special_tokens_map.json",
]
_TRAINED_OPTIONAL_FILES = _OPTIONAL_TOKENIZER_FILES

# The quantized ONNX file is named `model_quantized.onnx` by Optimum's dynamic
# quantization, but custom export pipelines may produce `model.onnx`. Either
# satisfies the requirement.
_QUANTIZED_MODEL_FILE_CANDIDATES = ["model_quantized.onnx", "model.onnx"]
_QUANTIZED_REQUIRED_FILES = [
    "label_mappings.json",
    "tokenizer_config.json",
    "tokenizer.json",
    "special_tokens_map.json",
]
_QUANTIZED_OPTIONAL_FILES = [
    "config.json",
    "ort_config.json",
    "model.onnx.data",
    "model_manifest.json",
    "crf_transitions.json",
    *_OPTIONAL_TOKENIZER_FILES,
]


def _load_label_info(model_dir: str) -> tuple[int, list[str]]:
    """Load label mappings and return (num_pii_labels, pii_entities)."""
    mappings_path = Path(model_dir) / "label_mappings.json"
    if not mappings_path.exists():
        return 0, []
    with mappings_path.open() as f:
        mappings = json.load(f)
    pii_label2id = mappings.get("pii", {}).get("label2id", {})
    num_pii_labels = len(pii_label2id)
    pii_entities = sorted({k[2:] for k in pii_label2id if k.startswith("B-")})
    return num_pii_labels, pii_entities


def _load_model_config(model_dir: str) -> dict:
    """Load HuggingFace `config.json` from the model directory (or empty dict)."""
    cfg_path = Path(model_dir) / "config.json"
    if not cfg_path.exists():
        return {}
    with cfg_path.open() as f:
        return json.load(f)


def _load_training_config() -> dict:
    """Load training hyperparameters from the project's training_config.toml."""
    cfg_path = PROJECT_ROOT / "model" / "flows" / "training_config.toml"
    if not cfg_path.exists():
        return {}
    with cfg_path.open("rb") as f:
        return tomllib.load(f)


def _count_safetensors_params(path: Path) -> int:
    """Count tensor elements declared in a safetensors header without loading weights."""
    if not path.exists():
        return 0
    with path.open("rb") as f:
        header_size = struct.unpack("<Q", f.read(8))[0]
        header = json.loads(f.read(header_size))
    total = 0
    for key, info in header.items():
        if key == "__metadata__":
            continue
        n = 1
        for dim in info.get("shape", []):
            n *= dim
        total += n
    return total


def _format_param_count(n: int) -> str:
    if n >= 1_000_000_000:
        return f"{n / 1e9:.2f}B"
    if n >= 1_000_000:
        return f"{n / 1e6:.0f}M"
    if n >= 1_000:
        return f"{n / 1e3:.0f}K"
    return str(n)


_MODEL_TYPE_DISPLAY = {
    "deberta-v2": "DeBERTa-v3",
    "deberta": "DeBERTa",
    "distilbert": "DistilBERT",
    "roberta": "RoBERTa",
    "bert": "BERT",
    "xlm-roberta": "XLM-RoBERTa",
}


def _encoder_display_name(config: dict, base_model: str) -> str:
    model_type = config.get("model_type", "")
    if model_type in _MODEL_TYPE_DISPLAY:
        return _MODEL_TYPE_DISPLAY[model_type]
    if "deberta-v3" in base_model:
        return "DeBERTa-v3"
    if "distilbert" in base_model:
        return "DistilBERT"
    return model_type or "Transformer"


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
    num_pii_labels, pii_entities = _load_label_info(model_dir)
    config = _load_model_config(model_dir)
    training = _load_training_config()

    model_file = Path(model_dir) / "model.safetensors"
    model_size_mb = (
        model_file.stat().st_size / (1024 * 1024) if model_file.exists() else 0
    )
    param_count = _count_safetensors_params(model_file)
    param_str = _format_param_count(param_count) if param_count else "—"

    encoder_name = _encoder_display_name(config, base_model)
    hidden_size = config.get("hidden_size", 768)
    bottleneck_size = hidden_size // 2
    model_type_tag = config.get("model_type") or (
        "deberta-v2" if "deberta" in base_model else "transformer"
    )

    label_rows = _pii_label_rows(pii_entities)
    yaml_lang_block = _yaml_language_block()

    train_cfg = training.get("training", {}) if training else {}
    num_epochs = train_cfg.get("num_epochs", "—")
    batch_size = train_cfg.get("batch_size", "—")
    learning_rate = train_cfg.get("learning_rate", "—")
    weight_decay = train_cfg.get("weight_decay", "—")
    warmup_steps = train_cfg.get("warmup_steps", "—")
    bf16 = train_cfg.get("bf16", False)
    aux_ce_weight = train_cfg.get("auxiliary_ce_loss_weight", 0.0)
    es_patience = train_cfg.get("early_stopping_patience", "—")
    es_threshold = train_cfg.get("early_stopping_threshold")
    es_threshold_str = (
        f"{es_threshold * 100:.2f}%" if isinstance(es_threshold, (int, float)) else "—"
    )
    precision_str = "bf16 mixed precision" if bf16 else "fp32"

    dataset_section = ""
    if dataset_repo_id:
        dataset_section = f"""
## Training Data

Trained on the [{dataset_repo_id}](https://huggingface.co/datasets/{dataset_repo_id}) dataset — a synthetic multilingual PII dataset with entity annotations.
"""

    derived_section = ""
    if quantized_repo_id:
        derived_section = f"""
## Derived Models

| Variant | Format | Repository |
|---------|--------|------------|
| Quantized | ONNX | [{quantized_repo_id}](https://huggingface.co/{quantized_repo_id}) |
"""

    aux_loss_clause = (
        f" + {aux_ce_weight}×class-weighted token cross-entropy"
        if aux_ce_weight
        else ""
    )

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
- token-classification
- crf
- {model_type_tag}
base_model: {base_model}
---

# Kiji PII Detection Model

Token classification model for detecting Personally Identifiable Information (PII) in text. Fine-tuned from [`{base_model}`](https://huggingface.co/{base_model}) and decoded with a CRF layer for valid BIO sequence prediction.

## Model Summary

| | |
|---|---|
| **Base model** | [{base_model}](https://huggingface.co/{base_model}) |
| **Architecture** | {encoder_name} encoder + MLP token classifier + CRF |
| **Parameters** | {param_str} |
| **Model size** | {model_size_mb:.0f} MB (SafeTensors) |
| **Hidden size** | {hidden_size} |
| **Task** | PII token classification ({num_pii_labels} BIO labels) |
| **PII entity types** | {len(pii_entities)} |
| **Decoder** | CRF (Viterbi) |
| **Max sequence length** | 512 tokens |

## Architecture

```
Input (input_ids, attention_mask)
        │
  {encoder_name} encoder (hidden_size={hidden_size})
        │
  Dropout → Linear({hidden_size} → {bottleneck_size}) → GELU → Dropout
        │
  Linear({bottleneck_size} → {num_pii_labels})        [BIO emission scores]
        │
  CRF                                  [valid BIO transitions]
        │
  Predicted label sequence
```

The token classifier emits per-token BIO scores; a learned CRF layer enforces valid transitions (e.g., an `I-EMAIL` cannot follow a `B-PHONENUMBER`). The training loss is the CRF negative log-likelihood{aux_loss_clause}. At inference time, predictions are produced by Viterbi decoding.

## Usage

The repository contains the encoder weights, MLP head, and CRF parameters in a single SafeTensors file. The architecture is custom (`PIIDetectionModel`) and is not loadable via `AutoModelForTokenClassification` — see `model/src/model.py` in the source repository for the head + CRF wiring.

```python
from transformers import AutoTokenizer
from safetensors.torch import load_file

tokenizer = AutoTokenizer.from_pretrained("{repo_id}")
weights = load_file("model.safetensors")  # downloaded from this repo

text = "Contact John Smith at john.smith@example.com or call +1-555-123-4567."
inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
# See label_mappings.json for the BIO label set.
```

## PII Labels (BIO tagging)

The model uses BIO tagging with {len(pii_entities)} entity types:

| Label | Description |
|-------|-------------|
{label_rows}

Each entity type has `B-` (beginning) and `I-` (inside) variants, plus `O` for non-PII tokens.

## Training

| | |
|---|---|
| **Epochs** | {num_epochs} (with early stopping) |
| **Batch size** | {batch_size} |
| **Learning rate** | {learning_rate} |
| **Weight decay** | {weight_decay} |
| **Warmup steps** | {warmup_steps} |
| **Precision** | {precision_str} |
| **Early stopping** | patience={es_patience}, threshold={es_threshold_str} |
| **Loss** | CRF NLL{aux_loss_clause} |
| **Optimizer** | AdamW |
| **Metric** | Weighted F1 (token-level) |
{dataset_section}{derived_section}
## Limitations

- Trained on **synthetically generated** data — may not generalize perfectly to all real-world text
- Optimized for the {len(LabelUtils.LANGUAGE_CODES)} languages in the training data ({", ".join(LabelUtils.LANGUAGE_CODES.keys())})
- Max sequence length is 512 tokens
- CRF transitions are learned from training data — rare BIO transitions may be underweighted
"""
    return card.strip() + "\n"


def _generate_quantized_model_card(
    repo_id: str,
    model_dir: str,
    trained_repo_id: str | None,
    dataset_repo_id: str | None,
    onnx_filename: str = "model_quantized.onnx",
) -> str:
    """Generate a model card for the quantized (ONNX) model."""
    num_pii_labels, pii_entities = _load_label_info(model_dir)
    config = _load_model_config(model_dir)
    label_rows = _pii_label_rows(pii_entities)
    yaml_lang_block = _yaml_language_block()

    model_path = Path(model_dir)
    file_rows = ""
    candidate_files = [
        *_QUANTIZED_MODEL_FILE_CANDIDATES,
        *_QUANTIZED_REQUIRED_FILES,
        *_QUANTIZED_OPTIONAL_FILES,
    ]
    seen: set[str] = set()
    for f in candidate_files:
        if f in seen:
            continue
        seen.add(f)
        fpath = model_path / f
        if fpath.exists():
            size = fpath.stat().st_size
            if size > 1024 * 1024:
                size_str = f"{size / (1024 * 1024):.1f} MB"
            else:
                size_str = f"{size / 1024:.1f} KB"
            file_rows += f"| `{f}` | {size_str} |\n"

    # Detect whether the ONNX file is actually quantized (vs plain export).
    is_quantized = "quantized" in onnx_filename
    has_crf_json = (model_path / "crf_transitions.json").exists()

    # Load quantization config for details (only when the uploaded ONNX is the
    # quantized variant; ort_config.json may exist alongside an unquantized
    # model.onnx in mixed-export pipelines).
    ort_config_path = model_path / "ort_config.json"
    quant_details = ""
    if "quantized" in onnx_filename and ort_config_path.exists():
        with ort_config_path.open() as f:
            ort_config = json.load(f)
        q = ort_config.get("quantization", {})
        if q:
            operators = ", ".join(q.get("operators_to_quantize", []))
            quant_details = f"""
## Quantization Details

| | |
|---|---|
| **Method** | Dynamic quantization (ONNX Runtime / Optimum) |
| **Weights** | {q.get("weights_dtype", "QInt8")} |
| **Activations** | {q.get("activations_dtype", "QUInt8")} |
| **Mode** | {q.get("mode", "IntegerOps")} |
| **Format** | {q.get("format", "QOperator")} |
| **Operators quantized** | {operators} |
"""

    # Pick a base_model value; the trained repo if known, else fall back to config or v3-base
    base_model_value = (
        trained_repo_id or config.get("_name_or_path") or "microsoft/deberta-v3-base"
    )

    encoder_name = _encoder_display_name(config, base_model_value)
    hidden_size = config.get("hidden_size", 768)
    model_type_tag = config.get("model_type") or (
        "deberta-v2" if "deberta" in base_model_value else "transformer"
    )

    title_suffix = "ONNX (Quantized)" if is_quantized else "ONNX"
    format_label = "ONNX (INT8 dynamic quantization)" if is_quantized else "ONNX"
    quant_tags = "\n- quantized\n- int8" if is_quantized else ""

    trained_section = ""
    if trained_repo_id:
        trained_section = f"""
## Source Model

ONNX export of [{trained_repo_id}](https://huggingface.co/{trained_repo_id}) — the {encoder_name} encoder fine-tuned for PII token classification with a CRF decoder.
"""

    dataset_section = ""
    if dataset_repo_id:
        dataset_section = f"""
## Training Data

The source model was trained on the [{dataset_repo_id}](https://huggingface.co/datasets/{dataset_repo_id}) dataset — a synthetic multilingual PII dataset with entity annotations.
"""

    lineage_section = ""
    if dataset_repo_id or trained_repo_id:
        lineage_rows = ""
        if dataset_repo_id:
            lineage_rows += f"| Dataset | [{dataset_repo_id}](https://huggingface.co/datasets/{dataset_repo_id}) |\n"
        if trained_repo_id:
            lineage_rows += f"| Trained model | [{trained_repo_id}](https://huggingface.co/{trained_repo_id}) |\n"
        lineage_rows += f"| **ONNX model** | **{repo_id}** (this repo) |\n"
        lineage_section = f"""
## Lineage

| Stage | Repository |
|-------|------------|
{lineage_rows}"""

    crf_paragraph = (
        " The CRF transition parameters are exported separately as `crf_transitions.json`; "
        "downstream code should perform Viterbi decoding using these transitions to convert "
        "emission logits into a valid BIO label sequence."
        if has_crf_json
        else ""
    )

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
- token-classification
- crf
- {model_type_tag}
- onnx{quant_tags}
base_model: {base_model_value}
---

# Kiji PII Detection Model ({title_suffix})

ONNX export of the Kiji PII detection model for efficient CPU inference. The ONNX graph contains the encoder and the token classification head; it emits per-token BIO emission scores.{crf_paragraph}
{trained_section}
## Model Summary

| | |
|---|---|
| **Format** | {format_label} |
| **Architecture** | {encoder_name} encoder + MLP token classifier (CRF transitions exported as JSON) |
| **Hidden size** | {hidden_size} |
| **Task** | PII token classification ({num_pii_labels} BIO labels) |
| **PII entity types** | {len(pii_entities)} |
| **ONNX outputs** | `pii_logits` — shape `(batch, seq_len, {num_pii_labels})` |
| **Max sequence length** | 512 tokens |
| **Runtime** | ONNX Runtime |

## Files

| File | Size |
|------|------|
{file_rows}{quant_details}
## Usage

```python
import json
import numpy as np
from onnxruntime import InferenceSession
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("{repo_id}")
session = InferenceSession("{onnx_filename}")  # downloaded from this repo

text = "Contact John Smith at john.smith@example.com or call +1-555-123-4567."
inputs = tokenizer(text, return_tensors="np", truncation=True, max_length=512)

# ONNX session expects int64 input_ids and attention_mask.
ort_inputs = {{
    "input_ids": inputs["input_ids"].astype("int64"),
    "attention_mask": inputs["attention_mask"].astype("int64"),
}}
(pii_logits,) = session.run(["pii_logits"], ort_inputs)

# For best accuracy, decode `pii_logits` with the CRF transitions:
#   crf = json.load(open("crf_transitions.json"))
#   labels = viterbi_decode(pii_logits[0], crf)   # implement using start/end/transitions
# A simple argmax baseline (no CRF) is also available:
labels = np.argmax(pii_logits, axis=-1)[0]
# See label_mappings.json for label ID -> label name.
```

## PII Labels (BIO tagging)

The model uses BIO tagging with {len(pii_entities)} entity types:

| Label | Description |
|-------|-------------|
{label_rows}

Each entity type has `B-` (beginning) and `I-` (inside) variants, plus `O` for non-PII tokens.
{dataset_section}{lineage_section}
## Limitations

- Trained on **synthetically generated** data — may not generalize perfectly to all real-world text
- Optimized for the {len(LabelUtils.LANGUAGE_CODES)} languages in the training data ({", ".join(LabelUtils.LANGUAGE_CODES.keys())})
- Max sequence length is 512 tokens
- The ONNX graph emits emission logits only; valid BIO sequences require Viterbi decoding with the exported CRF transitions
{"- Dynamic quantization may slightly reduce accuracy compared to the full-precision model" if is_quantized else ""}
"""
    return card.strip() + "\n"


def upload_model_to_huggingface(
    variant: str = "trained",
    model_dir: str | None = None,
    repo_id: str | None = None,
    private: bool = True,
    base_model: str = "microsoft/deberta-v3-small",
    dataset_repo_id: str | None = None,
    trained_repo_id: str | None = None,
    quantized_repo_id: str | None = None,
    do_create_repo: bool = False,
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
        do_create_repo: Whether to create the repo if it doesn't exist (requires create permissions)
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
    if variant == "trained":
        required_files = _TRAINED_REQUIRED_FILES
        optional_files = _TRAINED_OPTIONAL_FILES
    else:
        required_files = _QUANTIZED_REQUIRED_FILES
        optional_files = _QUANTIZED_OPTIONAL_FILES

    # Verify required files exist
    missing = [f for f in required_files if not (model_path / f).exists()]
    if missing:
        raise ValueError(f"Missing required model files: {missing}")

    # For the quantized variant, exactly one of the candidate ONNX filenames
    # must be present.
    quantized_model_file: str | None = None
    if variant == "quantized":
        present_candidates = [
            f for f in _QUANTIZED_MODEL_FILE_CANDIDATES if (model_path / f).exists()
        ]
        if not present_candidates:
            raise ValueError(
                f"Missing quantized ONNX model file (expected one of {_QUANTIZED_MODEL_FILE_CANDIDATES})"
            )
        quantized_model_file = present_candidates[0]

    present_optional = [f for f in optional_files if (model_path / f).exists()]
    model_files = required_files + present_optional
    if quantized_model_file:
        model_files = [quantized_model_file, *model_files]

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
            onnx_filename=quantized_model_file or "model_quantized.onnx",
        )

    # Write model card to the model directory temporarily
    readme_path = model_path / "README.md"
    readme_existed = readme_path.exists()
    readme_path.write_text(card)

    # Optionally create the repo (requires create permissions on the org)
    allow_patterns = model_files + ["README.md"]
    if do_create_repo:
        create_repo(
            repo_id=repo_id,
            repo_type="model",
            token=token,
            private=private,
            exist_ok=True,
        )
        print(f"Created/verified repo: {repo_id}")

    print(f"\nPushing to {repo_id}...")
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
        "--create-repo",
        action="store_true",
        help="Create the repo if it doesn't exist (requires create permissions on the org)",
    )
    parser.add_argument(
        "--base-model",
        default="microsoft/deberta-v3-small",
        help="Base model name for fine-tuning (default: microsoft/deberta-v3-small)",
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
        do_create_repo=args.create_repo,
    )
