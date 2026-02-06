#!/usr/bin/env python3
"""Download a PII dataset from HuggingFace Hub and save as local Label Studio JSON files."""

import hashlib
import json
import os
import sys
from datetime import datetime
from pathlib import Path

from datasets import load_dataset

# Add project root to path for imports
project_root = Path(__file__).parent.parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from model.dataset.labelstudio.labelstudio_format import convert_to_labelstudio


def download_from_huggingface(
    repo_id: str,
    output_dir: str = "model/dataset/data_samples/training_samples",
    splits: list[str] | None = None,
):
    """
    Download a PII dataset from HuggingFace Hub and save as local Label Studio JSON files.

    This produces the same file format that the training pipeline expects:
    individual JSON files in Label Studio format in the training_samples directory.

    Args:
        repo_id: HuggingFace repo ID (e.g., "username/kiji-pii-training-data")
        output_dir: Directory to save JSON files (default: training_samples)
        splits: Which splits to download (default: all available)
    """
    token = os.environ.get("HF_TOKEN")

    print(f"Loading dataset from {repo_id}...")
    ds = load_dataset(repo_id, token=token)

    if splits is None:
        splits = list(ds.keys())

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    total_saved = 0
    for split_name in splits:
        if split_name not in ds:
            print(f"  Split '{split_name}' not found, skipping")
            continue

        split_data = ds[split_name]
        print(f"  Processing split '{split_name}': {len(split_data)} samples")

        for row in split_data:
            # Build the clean training format dict expected by convert_to_labelstudio
            sample = {
                "text": row["text"],
                "privacy_mask": row.get("privacy_mask", []),
                "coreferences": row.get("coreferences", []),
                "language": row.get("language"),
                "country": row.get("country"),
            }

            # Convert to Label Studio format
            ls_sample = convert_to_labelstudio(sample)

            # Generate filename: timestamp + hash of text
            text_hash = hashlib.sha256(sample["text"].encode()).hexdigest()
            timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
            file_name = f"{timestamp}_{text_hash}.json"

            # Add file_name to data section
            ls_sample["data"]["file_name"] = file_name

            file_path = output_path / file_name
            with file_path.open("w", encoding="utf-8") as f:
                json.dump(ls_sample, f, indent=2, ensure_ascii=False)

            total_saved += 1

    print(f"\nSaved {total_saved} samples to {output_dir}")
    print(
        f"Ready for training with: uv run python model/flows/training_pipeline.py run"
    )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Download PII dataset from HuggingFace Hub to local JSON files"
    )
    parser.add_argument(
        "--repo-id",
        required=True,
        help="HuggingFace repo ID (e.g., 'username/kiji-pii-training-data')",
    )
    parser.add_argument(
        "--output-dir",
        default="model/dataset/data_samples/training_samples",
        help="Directory to save JSON files",
    )
    parser.add_argument(
        "--splits",
        nargs="*",
        default=None,
        help="Which splits to download (default: all)",
    )

    args = parser.parse_args()

    download_from_huggingface(
        repo_id=args.repo_id,
        output_dir=args.output_dir,
        splits=args.splits,
    )
