#!/usr/bin/env python3
"""Upload generated PII samples to HuggingFace Hub."""

import json
import os
from pathlib import Path

from huggingface_hub import HfApi, create_repo


def upload_samples(
    samples_dir: str = "model/dataset/reviewed_samples",
    repo_id: str = None,
    private: bool = True,
):
    """
    Upload samples directory to HuggingFace Hub.

    Args:
        samples_dir: Path to directory containing JSON samples
        repo_id: HuggingFace repo ID (e.g., "username/pii-training-data")
        private: Whether to make the repo private
    """
    token = os.environ.get("HF_TOKEN")
    if not token:
        raise ValueError("HF_TOKEN environment variable not set")

    if not repo_id:
        raise ValueError("repo_id is required (e.g., 'username/pii-training-data')")

    samples_path = Path(samples_dir)
    if not samples_path.exists():
        raise ValueError(f"Samples directory not found: {samples_dir}")

    # Count samples
    json_files = list(samples_path.glob("*.json"))
    print(f"Found {len(json_files)} samples in {samples_dir}")

    if len(json_files) == 0:
        print("No samples to upload!")
        return

    # Create repo if it doesn't exist
    api = HfApi(token=token)
    try:
        create_repo(repo_id, repo_type="dataset", private=private, token=token)
        print(f"Created new dataset repo: {repo_id}")
    except Exception as e:
        if "already exists" in str(e).lower() or "409" in str(e):
            print(f"Repo {repo_id} already exists, will update")
        else:
            raise

    # Option 1: Upload as individual files (good for smaller datasets)
    # Option 2: Combine into single JSONL and upload (better for larger datasets)

    # Using Option 2: Combine into JSONL for efficiency
    jsonl_path = samples_path / "samples.jsonl"
    print(f"Combining samples into {jsonl_path}...")

    with open(jsonl_path, "w") as f:
        for json_file in sorted(json_files):
            if json_file.name == "samples.jsonl":
                continue
            try:
                with open(json_file) as jf:
                    sample = json.load(jf)
                    f.write(json.dumps(sample) + "\n")
            except json.JSONDecodeError as e:
                print(f"  Skipping {json_file.name}: {e}")

    # Upload the JSONL file
    print(f"Uploading to {repo_id}...")
    api.upload_file(
        path_or_fileobj=str(jsonl_path),
        path_in_repo="data/samples.jsonl",
        repo_id=repo_id,
        repo_type="dataset",
        token=token,
    )

    # Create a simple README
    readme_content = f"""# PII Training Data

Synthetic PII training samples for token classification.

## Dataset Info
- **Samples**: {len(json_files)}
- **Format**: JSONL

## Sample Structure
```json
{{
  "text": "The text containing PII...",
  "privacy_mask": [
    {{"value": "John", "label": "FIRSTNAME"}},
    {{"value": "Doe", "label": "SURNAME"}}
  ],
  "coreferences": [
    {{"cluster_id": 0, "mentions": ["John Doe", "He", "his"], "entity_type": "person"}}
  ]
}}
```

## Usage
```python
from datasets import load_dataset
ds = load_dataset("{repo_id}")
```
"""

    readme_path = samples_path / "README.md"
    with open(readme_path, "w") as f:
        f.write(readme_content)

    api.upload_file(
        path_or_fileobj=str(readme_path),
        path_in_repo="README.md",
        repo_id=repo_id,
        repo_type="dataset",
        token=token,
    )

    print(f"Done! Dataset available at: https://huggingface.co/datasets/{repo_id}")

    # Cleanup temp files
    jsonl_path.unlink()
    readme_path.unlink()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Upload PII samples to HuggingFace Hub"
    )
    parser.add_argument(
        "--samples-dir",
        default="model/dataset/reviewed_samples",
        help="Directory containing JSON samples",
    )
    parser.add_argument(
        "--repo-id",
        required=True,
        help="HuggingFace repo ID (e.g., 'username/pii-training-data')",
    )
    parser.add_argument(
        "--public",
        action="store_true",
        help="Make the dataset public (default: private)",
    )

    args = parser.parse_args()

    upload_samples(
        samples_dir=args.samples_dir,
        repo_id=args.repo_id,
        private=not args.public,
    )
