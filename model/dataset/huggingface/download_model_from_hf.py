#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.13"
# dependencies = []
# ///
"""Download trained or quantized PII detection model from HuggingFace Hub."""

import os
import sys
from pathlib import Path

from huggingface_hub import HfApi, snapshot_download
from huggingface_hub.errors import GatedRepoError, RepositoryNotFoundError

DEFAULT_TRAINED_REPO_ID = "DataikuNLP/kiji-pii-model"
DEFAULT_QUANTIZED_REPO_ID = "DataikuNLP/kiji-pii-model-onnx"

# Mirrors the file lists from upload_model_to_hf.py so the download targets the
# artifacts the training/inference pipeline actually consumes.
_OPTIONAL_TOKENIZER_FILES = ["vocab.txt", "spm.model", "added_tokens.json"]

_TRAINED_ALLOW_PATTERNS = [
    "model.safetensors",
    "config.json",
    "label_mappings.json",
    "tokenizer_config.json",
    "tokenizer.json",
    "special_tokens_map.json",
    "model.onnx",
    "crf_transitions.json",
    *_OPTIONAL_TOKENIZER_FILES,
]

_QUANTIZED_ALLOW_PATTERNS = [
    "model_quantized.onnx",
    "model.onnx",
    "model.onnx.data",
    "label_mappings.json",
    "tokenizer_config.json",
    "tokenizer.json",
    "special_tokens_map.json",
    "config.json",
    "ort_config.json",
    "model_manifest.json",
    "crf_transitions.json",
    *_OPTIONAL_TOKENIZER_FILES,
]


def download_model_from_huggingface(
    variant: str = "trained",
    repo_id: str | None = None,
    output_dir: str | None = None,
    revision: str | None = None,
):
    """
    Download a PII detection model from HuggingFace Hub.

    Supports both trained (SafeTensors) and quantized (ONNX) variants.

    Args:
        variant: Model variant to download ("trained" or "quantized")
        repo_id: HuggingFace repo ID (defaults based on variant)
        output_dir: Destination directory (defaults based on variant)
        revision: Optional git revision (branch, tag, or commit) to pin
    """
    token = os.environ.get("HF_TOKEN")  # required only for private/gated repos

    if variant not in ("trained", "quantized"):
        raise ValueError(
            f"Unknown variant: {variant}. Must be 'trained' or 'quantized'"
        )

    if repo_id is None:
        repo_id = (
            DEFAULT_TRAINED_REPO_ID
            if variant == "trained"
            else DEFAULT_QUANTIZED_REPO_ID
        )

    if output_dir is None:
        output_dir = "model/trained" if variant == "trained" else "model/quantized"

    allow_patterns = (
        _TRAINED_ALLOW_PATTERNS if variant == "trained" else _QUANTIZED_ALLOW_PATTERNS
    )

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Probe the repo first: public repos download without a token; private or
    # gated repos surface a clear error if HF_TOKEN is missing.
    try:
        HfApi().model_info(repo_id, token=token)
    except (RepositoryNotFoundError, GatedRepoError) as e:
        if not token:
            raise ValueError(
                f"Cannot access {repo_id}. If the repo is private or gated, "
                "set the HF_TOKEN environment variable."
            ) from e
        raise

    print(f"Downloading {variant} model from {repo_id} to {output_dir}...")
    snapshot_download(
        repo_id=repo_id,
        repo_type="model",
        local_dir=str(output_path),
        token=token,
        revision=revision,
        allow_patterns=allow_patterns,
    )

    downloaded = sorted(p for p in output_path.iterdir() if p.is_file())
    for f in downloaded:
        size = f.stat().st_size
        print(
            f"  {f.name}: {size / (1024 * 1024):.1f} MB"
            if size > 1024 * 1024
            else f"  {f.name}: {size / 1024:.1f} KB"
        )

    print(f"\nDone! Model downloaded to: {output_dir}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Download trained or quantized PII model from HuggingFace Hub"
    )
    parser.add_argument(
        "--variant",
        choices=["trained", "quantized"],
        default="trained",
        help="Model variant to download (default: trained)",
    )
    parser.add_argument(
        "--repo-id",
        default=None,
        help=(
            "HuggingFace repo ID "
            f"(defaults: {DEFAULT_TRAINED_REPO_ID} / {DEFAULT_QUANTIZED_REPO_ID})"
        ),
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Directory to download model into (default: model/trained or model/quantized)",
    )
    parser.add_argument(
        "--revision",
        default=None,
        help="Git revision (branch, tag, or commit) to pin (default: main)",
    )

    args = parser.parse_args()

    try:
        download_model_from_huggingface(
            variant=args.variant,
            repo_id=args.repo_id,
            output_dir=args.output_dir,
            revision=args.revision,
        )
    except ValueError as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)
