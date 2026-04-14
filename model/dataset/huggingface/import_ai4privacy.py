#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.13"
# dependencies = []
# ///
"""Import English samples from ai4privacy/pii-masking-200k into Kiji training format.

Downloads the dataset from HuggingFace, maps AI4Privacy entity labels to Kiji's
standard PII labels, and saves as Label Studio JSON files ready for training.

Usage:
    uv run python model/dataset/huggingface/import_ai4privacy.py
    uv run python model/dataset/huggingface/import_ai4privacy.py --max-samples 5000
    uv run python model/dataset/huggingface/import_ai4privacy.py --output-dir model/dataset/data_samples/training_samples
"""

import hashlib
import json
import sys
from datetime import datetime
from pathlib import Path

from datasets import load_dataset

# Add project root to path for imports
project_root = Path(__file__).parent.parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from model.dataset.labelstudio.labelstudio_format import (  # noqa: E402
    convert_to_labelstudio,
)

# AI4Privacy labels → Kiji standard PII labels.
# Labels mapped to None are dropped (no Kiji equivalent).
AI4PRIVACY_TO_KIJI: dict[str, str | None] = {
    # Names
    "FIRSTNAME": "FIRSTNAME",
    "LASTNAME": "SURNAME",
    "MIDDLENAME": "FIRSTNAME",
    # Contact
    "EMAIL": "EMAIL",
    "PHONENUMBER": "PHONENUMBER",
    "URL": "URL",
    "USERNAME": "USERNAME",
    # Address
    "BUILDINGNUMBER": "BUILDINGNUM",
    "STREET": "STREET",
    "SECONDARYADDRESS": "STREET",
    "CITY": "CITY",
    "COUNTY": "STATE",
    "STATE": "STATE",
    "ZIPCODE": "ZIP",
    # Identity & dates
    "DOB": "DATEOFBIRTH",
    "AGE": "AGE",
    "SSN": "SSN",
    # Financial
    "CREDITCARDNUMBER": "CREDITCARDNUMBER",
    "IBAN": "IBAN",
    # Other
    "COMPANYNAME": "COMPANYNAME",
    "PASSWORD": "PASSWORD",
    "VEHICLEVRM": "LICENSEPLATENUM",
    # --- Unmapped (dropped) ---
    "PREFIX": None,
    "DATE": None,
    "TIME": None,
    "PHONEIMEI": None,
    "GENDER": None,
    "SEX": None,
    "JOBAREA": None,
    "JOBTYPE": None,
    "JOBTITLE": None,
    "USERAGENT": None,
    "ACCOUNTNAME": None,
    "ACCOUNTNUMBER": None,
    "CURRENCYSYMBOL": None,
    "AMOUNT": None,
    "CREDITCARDISSUER": None,
    "CREDITCARDCVV": None,
    "IP": None,
    "IPV4": None,
    "IPV6": None,
    "MAC": None,
    "ETHEREUMADDRESS": None,
    "BITCOINADDRESS": None,
    "LITECOINADDRESS": None,
    "CURRENCY": None,
    "CURRENCYNAME": None,
    "CURRENCYCODE": None,
    "ORDINALDIRECTION": None,
    "MASKEDNUMBER": None,
    "BIC": None,
    "NEARBYGPSCOORDINATE": None,
    "VEHICLEVIN": None,
    "EYECOLOR": None,
    "HEIGHT": None,
    "PIN": None,
}


def convert_ai4privacy_sample(row: dict) -> dict | None:
    """Convert a single AI4Privacy row to Kiji's internal training format.

    Args:
        row: A row from the ai4privacy/pii-masking-200k dataset with keys
             ``source_text``, ``privacy_mask``, and ``language``.

    Returns:
        A dict with ``text``, ``privacy_mask``, ``coreferences``, ``language``,
        and ``country`` ready for ``convert_to_labelstudio()``, or ``None`` if
        the sample has no mappable entities after label conversion.
    """
    text = row.get("source_text", "")
    if not text:
        return None

    # Build privacy_mask with character offsets resolved from the source text.
    # ai4privacy only provides entity value + label, not offsets.
    privacy_mask = []
    search_from = 0
    for entity in row.get("privacy_mask", []):
        external_label = entity.get("label", "")
        kiji_label = AI4PRIVACY_TO_KIJI.get(external_label)
        if kiji_label is None:
            continue
        value = entity["value"]
        # Find the entity in the text starting from last match position
        idx = text.find(value, search_from)
        if idx == -1:
            # Retry from the beginning (entity may appear before last match)
            idx = text.find(value)
        if idx == -1:
            # Entity text not found in source — skip it
            continue
        privacy_mask.append(
            {
                "value": value,
                "label": kiji_label,
                "start": idx,
                "end": idx + len(value),
            }
        )
        search_from = idx + len(value)

    if not privacy_mask:
        return None

    return {
        "text": text,
        "privacy_mask": privacy_mask,
        "coreferences": [],
        "language": "English",
        "country": None,
    }


def import_ai4privacy(
    output_dir: str = "model/dataset/data_samples/training_samples",
    max_samples: int = 0,
):
    """Download English AI4Privacy samples and save as Label Studio JSON files.

    Args:
        output_dir: Directory to write Label Studio JSON files.
        max_samples: Maximum number of samples to import (0 = all).
    """
    print("Loading ai4privacy/pii-masking-200k from HuggingFace...")
    ds = load_dataset("ai4privacy/pii-masking-200k", split="train")

    # Filter to English only
    ds = ds.filter(lambda row: row.get("language") == "en")
    print(f"  English samples: {len(ds)}")

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    saved = 0
    skipped = 0
    label_counts: dict[str, int] = {}

    for row in ds:
        if max_samples > 0 and saved >= max_samples:
            break

        sample = convert_ai4privacy_sample(row)
        if sample is None:
            skipped += 1
            continue

        # Track label distribution
        for entity in sample["privacy_mask"]:
            label = entity["label"]
            label_counts[label] = label_counts.get(label, 0) + 1

        # Convert to Label Studio format
        ls_sample = convert_to_labelstudio(sample)

        # Generate deterministic filename from text hash
        text_hash = hashlib.sha256(sample["text"].encode()).hexdigest()[:16]
        timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
        file_name = f"ai4p_{timestamp}_{text_hash}.json"
        ls_sample["data"]["file_name"] = file_name

        file_path = output_path / file_name
        with file_path.open("w", encoding="utf-8") as f:
            json.dump(ls_sample, f, indent=2, ensure_ascii=False)

        saved += 1
        if saved % 1000 == 0:
            print(f"  Saved {saved} samples...")

    print("\nImport complete:")
    print(f"  Saved:   {saved}")
    print(f"  Skipped: {skipped} (no mappable entities)")
    print(f"  Output:  {output_dir}")
    print("\nLabel distribution (mapped to Kiji):")
    for label, count in sorted(label_counts.items(), key=lambda x: -x[1]):
        print(f"  {label:25s} {count:>6,}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Import AI4Privacy dataset into Kiji training format"
    )
    parser.add_argument(
        "--output-dir",
        default="model/dataset/data_samples/training_samples",
        help="Directory to save Label Studio JSON files",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=0,
        help="Max samples to import (0 = all)",
    )

    args = parser.parse_args()
    import_ai4privacy(output_dir=args.output_dir, max_samples=args.max_samples)
