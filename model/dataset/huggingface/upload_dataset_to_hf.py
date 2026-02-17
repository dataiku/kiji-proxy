#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.13"
# dependencies = [
#   "kiji-proxy[training]"
# ]
# ///
"""Upload PII training samples to HuggingFace Hub as a Parquet dataset."""

import json
import os
import sys
from collections import Counter
from pathlib import Path

from datasets import Dataset, DatasetDict
from huggingface_hub import HfApi

# Add project root to path for imports
project_root = Path(__file__).parent.parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from model.dataset.label_utils import LabelUtils  # noqa: E402
from model.src.preprocessing import DatasetProcessor  # noqa: E402

# PII labels that are part of the core annotation schema (exclude coreference-only labels)
_CORE_PII_LABELS = set(LabelUtils.STANDARD_PII_LABELS)


def load_and_convert_samples(samples_dir: str) -> list[dict]:
    """
    Load Label Studio JSON files and convert to clean training format.

    Args:
        samples_dir: Path to directory containing Label Studio JSON samples

    Returns:
        List of samples in clean training format
    """
    samples_path = Path(samples_dir)
    if not samples_path.exists():
        raise ValueError(f"Samples directory not found: {samples_dir}")

    json_files = sorted(samples_path.glob("*.json"))
    print(f"Found {len(json_files)} JSON files in {samples_dir}")

    if not json_files:
        raise ValueError("No JSON files found!")

    # Reuse the Label Studio → training format conversion from DatasetProcessor.
    # The method doesn't use self, so we pass None.
    convert = DatasetProcessor.convert_labelstudio_to_training_format

    samples = []
    skipped = 0
    for json_file in json_files:
        try:
            with json_file.open() as f:
                ls_sample = json.load(f)

            converted = convert(None, ls_sample, file_name=json_file.name)
            if converted:
                samples.append(converted)
            else:
                skipped += 1
        except (json.JSONDecodeError, Exception) as e:
            print(f"  Skipping {json_file.name}: {e}")
            skipped += 1

    print(f"Converted {len(samples)} samples ({skipped} skipped)")
    return samples


def _compute_dataset_stats(samples: list[dict]) -> dict:
    """Compute statistics from converted samples for the dataset card."""
    languages = Counter()
    countries = Counter()
    label_counts = Counter()
    total_entities = 0
    total_coref_clusters = 0
    text_lengths = []
    entities_per_sample = []
    samples_with_coref = 0

    for s in samples:
        languages[s.get("language", "Unknown")] += 1
        countries[s.get("country", "Unknown")] += 1
        text_lengths.append(len(s["text"]))

        pm = s.get("privacy_mask", [])
        entities_per_sample.append(len(pm))
        total_entities += len(pm)
        for entity in pm:
            label_counts[entity["label"]] += 1

        corefs = s.get("coreferences", [])
        total_coref_clusters += len(corefs)
        if corefs:
            samples_with_coref += 1

    return {
        "total_samples": len(samples),
        "total_entities": total_entities,
        "avg_entities": total_entities / len(samples) if samples else 0,
        "min_entities": min(entities_per_sample) if entities_per_sample else 0,
        "max_entities": max(entities_per_sample) if entities_per_sample else 0,
        "total_coref_clusters": total_coref_clusters,
        "samples_with_coref": samples_with_coref,
        "coref_pct": 100 * samples_with_coref / len(samples) if samples else 0,
        "text_len_min": min(text_lengths) if text_lengths else 0,
        "text_len_max": max(text_lengths) if text_lengths else 0,
        "text_len_avg": sum(text_lengths) / len(text_lengths) if text_lengths else 0,
        "languages": languages,
        "countries": countries,
        "label_counts": label_counts,
    }


def _generate_dataset_card(
    repo_id: str,
    stats: dict,
    train_count: int,
    test_count: int,
) -> str:
    """Generate a HuggingFace dataset card (README.md) with YAML frontmatter."""
    # Build language list for YAML frontmatter
    yaml_langs = []
    for lang in stats["languages"]:
        code = LabelUtils.LANGUAGE_CODES.get(lang, lang.lower()[:2])
        if code not in yaml_langs:
            yaml_langs.append(code)

    yaml_lang_block = "\n".join(f"- {code}" for code in sorted(yaml_langs))

    # Language distribution table
    lang_rows = ""
    for lang, count in stats["languages"].most_common():
        pct = 100 * count / stats["total_samples"]
        lang_rows += f"| {lang} | {count:,} | {pct:.1f}% |\n"

    # Country distribution table (top 15)
    country_rows = ""
    for country, count in stats["countries"].most_common(15):
        pct = 100 * count / stats["total_samples"]
        country_rows += f"| {country} | {count:,} | {pct:.1f}% |\n"
    remaining_countries = len(stats["countries"]) - 15
    if remaining_countries > 0:
        remaining_count = sum(c for _, c in stats["countries"].most_common()[15:])
        country_rows += f"| *({remaining_countries} more)* | {remaining_count:,} | {100 * remaining_count / stats['total_samples']:.1f}% |\n"

    # PII label distribution table (core labels only, sorted by count)
    label_rows = "| Label | Count |\n|-------|------:|\n"
    for label, count in stats["label_counts"].most_common():
        if label in _CORE_PII_LABELS:
            label_rows += f"| `{label}` | {count:,} |\n"

    card = f"""---
language:
{yaml_lang_block}
license: apache-2.0
task_categories:
- token-classification
task_ids:
- named-entity-recognition
tags:
- pii
- privacy
- ner
- coreference-resolution
- synthetic
pretty_name: Kiji PII Detection Training Data
size_categories:
- 10K<n<100K
---

# Kiji PII Detection Training Data

Synthetic multilingual dataset for training PII (Personally Identifiable Information) detection models with token-level entity annotations and coreference resolution.

## Dataset Summary

| | |
|---|---|
| **Samples** | {stats["total_samples"]:,} (train: {train_count:,}, test: {test_count:,}) |
| **Languages** | {len(stats["languages"])} ({", ".join(stats["languages"].keys())}) |
| **Countries** | {len(stats["countries"])} |
| **PII entity types** | {sum(1 for lbl in stats["label_counts"] if lbl in _CORE_PII_LABELS)} |
| **Total entity annotations** | {stats["total_entities"]:,} (avg {stats["avg_entities"]:.1f} per sample) |
| **Coreference clusters** | {stats["total_coref_clusters"]:,} ({stats["coref_pct"]:.0f}% of samples) |
| **Text length** | {stats["text_len_min"]:,}–{stats["text_len_max"]:,} chars (avg {stats["text_len_avg"]:.0f}) |

## Usage

```python
from datasets import load_dataset

ds = load_dataset("{repo_id}")

# Access a sample
sample = ds["train"][0]
print(sample["text"])
print(sample["privacy_mask"])   # PII entity annotations
print(sample["coreferences"])   # Coreference clusters
print(sample["language"])       # e.g. "English"
print(sample["country"])        # e.g. "United States"
```

## Schema

Each sample contains:

| Column | Type | Description |
|--------|------|-------------|
| `text` | `string` | Natural language text with embedded PII |
| `privacy_mask` | `list[{{"value": str, "label": str}}]` | PII entities with their text span and label |
| `coreferences` | `list[{{"mentions": list[str], "entity_type": str, "cluster_id": int}}]` | Coreference clusters linking mentions of the same entity |
| `language` | `string` | Language of the text |
| `country` | `string` | Country context for the PII (affects address/ID formats) |

### Example sample

```json
{{
  "text": "Contact Dr. Maria Santos at maria.santos@hospital.org or call +1-555-123-4567.",
  "privacy_mask": [
    {{"value": "Maria", "label": "FIRSTNAME"}},
    {{"value": "Santos", "label": "SURNAME"}},
    {{"value": "maria.santos@hospital.org", "label": "EMAIL"}},
    {{"value": "+1-555-123-4567", "label": "PHONENUMBER"}}
  ],
  "coreferences": [
    {{
      "mentions": ["Dr. Maria Santos", "maria.santos"],
      "entity_type": "FIRSTNAME",
      "cluster_id": 0
    }}
  ],
  "language": "English",
  "country": "United States"
}}
```

## PII Labels

{label_rows}

## Language Distribution

| Language | Samples | % |
|----------|--------:|--:|
{lang_rows}

## Country Distribution

| Country | Samples | % |
|---------|--------:|--:|
{country_rows}

## Data Generation

Samples are synthetically generated using LLMs with structured outputs. The generation pipeline:

1. **NER generation** — LLM produces text with embedded PII and entity annotations
2. **Coreference generation** — second pass links pronouns and references to their antecedent entities
3. **Review (optional)** — additional LLM pass validates and corrects annotations
4. **Format conversion** — samples are converted to a clean, standardized schema

## Intended Use

This dataset is designed for training token-classification models that detect and classify PII in text. The coreference annotations enable training models that can also resolve entity mentions (e.g., linking "he" back to "John Smith").

## Limitations

- All data is **synthetically generated** — entity distributions may not match real-world text
- Coreference annotations are LLM-generated and may contain errors
- Address and ID formats are country-specific but may not cover all regional variations
"""
    return card.strip() + "\n"


def upload_to_huggingface(
    samples_dir: str = "model/dataset/data_samples/training_samples",
    repo_id: str | None = None,
    private: bool = True,
    test_split_ratio: float = 0.1,
    create_repo: bool = False,
):
    """
    Convert Label Studio samples to clean format and push to HuggingFace Hub.

    Produces a Parquet-backed dataset with train/test splits that can be loaded with:
        ds = load_dataset("repo_id")

    Args:
        samples_dir: Path to directory containing Label Studio JSON samples
        repo_id: HuggingFace repo ID (e.g., "username/kiji-pii-training-data")
        private: Whether to make the repo private
        test_split_ratio: Fraction of data for the test split (default 0.1)
        create_repo: Whether to create the repo if it doesn't exist (requires create permissions)
    """
    token = os.environ.get("HF_TOKEN")
    if not token:
        raise ValueError("HF_TOKEN environment variable not set")

    if not repo_id:
        raise ValueError(
            "repo_id is required (e.g., 'username/kiji-pii-training-data')"
        )

    # Load and convert all samples
    samples = load_and_convert_samples(samples_dir)
    if not samples:
        raise ValueError("No samples could be converted!")

    # Build a flat list of dicts for Dataset.from_list
    rows = []
    for s in samples:
        rows.append(
            {
                "text": s["text"],
                "privacy_mask": s.get("privacy_mask", []),
                "coreferences": s.get("coreferences", []),
                "language": s.get("language"),
                "country": s.get("country"),
            }
        )

    # Create HuggingFace Dataset and split
    full_dataset = Dataset.from_list(rows)
    split = full_dataset.train_test_split(test_size=test_split_ratio, seed=42)
    dataset_dict = DatasetDict({"train": split["train"], "test": split["test"]})

    print("\nDataset splits:")
    print(f"  train: {len(dataset_dict['train'])} samples")
    print(f"  test:  {len(dataset_dict['test'])} samples")

    # Optionally create the repo (requires create permissions on the org)
    api = HfApi(token=token)
    if create_repo:
        api.create_repo(repo_id, repo_type="dataset", private=private, exist_ok=True)
        print(f"Created/verified repo: {repo_id}")

    # Push to Hub (creates Parquet files automatically)
    print(f"\nPushing to {repo_id} (private={private})...")
    dataset_dict.push_to_hub(repo_id, token=token)

    # Generate and upload dataset card
    print("Generating dataset card...")
    stats = _compute_dataset_stats(samples)
    card = _generate_dataset_card(
        repo_id=repo_id,
        stats=stats,
        train_count=len(dataset_dict["train"]),
        test_count=len(dataset_dict["test"]),
    )
    api.upload_file(
        path_or_fileobj=card.encode(),
        path_in_repo="README.md",
        repo_id=repo_id,
        repo_type="dataset",
    )

    print(f"Done! Dataset available at: https://huggingface.co/datasets/{repo_id}")
    print("\nUsage:")
    print("  from datasets import load_dataset")
    print(f'  ds = load_dataset("{repo_id}")')


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Upload PII training samples to HuggingFace Hub"
    )
    parser.add_argument(
        "--samples-dir",
        default="model/dataset/data_samples/training_samples",
        help="Directory containing Label Studio JSON samples",
    )
    parser.add_argument(
        "--repo-id",
        required=True,
        help="HuggingFace repo ID (e.g., 'username/kiji-pii-training-data')",
    )
    parser.add_argument(
        "--public",
        action="store_true",
        help="Make the dataset public (default: private)",
    )
    parser.add_argument(
        "--create-repo",
        action="store_true",
        help="Create the repo if it doesn't exist (requires create permissions on the org)",
    )
    parser.add_argument(
        "--test-split-ratio",
        type=float,
        default=0.1,
        help="Fraction of data for test split (default: 0.1)",
    )

    args = parser.parse_args()

    upload_to_huggingface(
        samples_dir=args.samples_dir,
        repo_id=args.repo_id,
        private=not args.public,
        test_split_ratio=args.test_split_ratio,
        create_repo=args.create_repo,
    )
