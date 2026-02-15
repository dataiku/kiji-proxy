#!/usr/bin/env python3
"""Analyze annotated PII datasets and print comprehensive statistics.

Works with both simplified format (samples/, reviewed_samples/) and
Label Studio format (annotation_samples/, training_samples/).

Usage:
    python model/dataset/analyze_dataset.py --samples-dir model/dataset/data_samples/reviewed_samples
"""

import argparse
import json
import sys
from collections import Counter
from itertools import combinations
from pathlib import Path

from tqdm import tqdm

# Add project root to path for imports
project_root = Path(__file__).parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from model.dataset.label_utils import LabelUtils  # noqa: E402

_CORE_PII_LABELS = set(LabelUtils.STANDARD_PII_LABELS)


def detect_format(sample: dict) -> str:
    """Detect whether a sample is in simplified or Label Studio format.

    Returns:
        "simplified" or "labelstudio"
    """
    if "data" in sample and "predictions" in sample:
        return "labelstudio"
    if "text" in sample and "privacy_mask" in sample:
        return "simplified"
    if "data" in sample:
        return "labelstudio"
    raise ValueError(f"Unknown sample format. Top-level keys: {list(sample.keys())}")


def extract_from_simplified(sample: dict) -> dict:
    """Extract normalized record from a simplified-format sample."""
    privacy_mask = sample.get("privacy_mask", []) or []
    coreferences = sample.get("coreferences", []) or []

    labels = [e["label"] for e in privacy_mask if e.get("label") in _CORE_PII_LABELS]
    all_labels = [e["label"] for e in privacy_mask if e.get("label")]
    entities = [
        (e["label"], e["value"])
        for e in privacy_mask
        if e.get("label") in _CORE_PII_LABELS and e.get("value")
    ]

    return {
        "text": sample.get("text", ""),
        "language": sample.get("language", "unknown"),
        "country": sample.get("country", "unknown"),
        "labels": labels,
        "all_labels": all_labels,
        "entities": entities,
        "entity_count": len(labels),
        "coref_cluster_count": len(coreferences),
        "has_coreferences": len(coreferences) > 0,
    }


def extract_from_labelstudio(sample: dict) -> dict:
    """Extract normalized record from a Label Studio format sample."""
    data = sample.get("data", {})
    text = data.get("text", "")
    language = data.get("language", "unknown")
    country = data.get("country", "unknown")

    # Get results from first prediction (or first annotation)
    results = []
    predictions = sample.get("predictions", [])
    annotations = sample.get("annotations", [])
    source = predictions or annotations
    if source:
        results = source[0].get("result", []) or []

    labels = []
    all_labels = []
    entities = []
    relation_to_ids = set()

    for entry in results:
        entry_type = entry.get("type")
        if entry_type == "labels":
            value = entry.get("value", {})
            entry_labels = value.get("labels", [])
            entry_text = value.get("text", "")
            for lbl in entry_labels:
                all_labels.append(lbl)
                if lbl in _CORE_PII_LABELS:
                    labels.append(lbl)
                    if entry_text:
                        entities.append((lbl, entry_text))
        elif entry_type == "relation":
            to_id = entry.get("to_id")
            if to_id:
                relation_to_ids.add(to_id)

    return {
        "text": text,
        "language": language,
        "country": country,
        "labels": labels,
        "all_labels": all_labels,
        "entities": entities,
        "entity_count": len(labels),
        "coref_cluster_count": len(relation_to_ids),
        "has_coreferences": len(relation_to_ids) > 0,
    }


def load_and_extract(samples_dir: Path) -> list[dict]:
    """Load all JSON files from a directory and extract normalized records."""
    json_files = sorted(samples_dir.glob("*.json"))
    if not json_files:
        print(f"No JSON files found in {samples_dir}")
        sys.exit(1)

    print(f"Found {len(json_files):,} JSON files in {samples_dir}\n")

    records = []
    errors = 0
    fmt = None

    for f in tqdm(json_files, desc="Loading samples"):
        try:
            with open(f, encoding="utf-8") as fh:
                sample = json.load(fh)

            if fmt is None:
                fmt = detect_format(sample)
                print(f"Detected format: {fmt}\n")

            if fmt == "simplified":
                records.append(extract_from_simplified(sample))
            else:
                records.append(extract_from_labelstudio(sample))
        except Exception:
            errors += 1

    if errors:
        print(f"\nSkipped {errors:,} files due to errors\n")

    return records


def _median(values: list[int | float]) -> float:
    """Compute the median of a list of numbers."""
    sorted_vals = sorted(values)
    n = len(sorted_vals)
    mid = n // 2
    if n % 2 == 0:
        return (sorted_vals[mid - 1] + sorted_vals[mid]) / 2
    return sorted_vals[mid]


def compute_stats(records: list[dict]) -> dict:
    """Compute comprehensive statistics from normalized records."""
    total = len(records)

    languages: Counter = Counter()
    countries: Counter = Counter()
    label_counts: Counter = Counter()
    unknown_labels: Counter = Counter()
    pair_counts: Counter = Counter()
    # Per-label value tracking: label -> Counter of values
    label_value_counts: dict[str, Counter] = {
        lbl: Counter() for lbl in _CORE_PII_LABELS
    }

    entity_counts = []
    text_lengths = []
    total_coref_clusters = 0
    samples_with_coref = 0
    unannotated = 0

    for rec in records:
        languages[rec["language"]] += 1
        countries[rec["country"]] += 1

        for lbl in rec["labels"]:
            label_counts[lbl] += 1

        # Track entity values per label
        for lbl, val in rec["entities"]:
            if lbl in label_value_counts:
                label_value_counts[lbl][val] += 1

        # Track unknown labels
        for lbl in rec["all_labels"]:
            if lbl not in _CORE_PII_LABELS:
                unknown_labels[lbl] += 1

        entity_counts.append(rec["entity_count"])
        text_lengths.append(len(rec["text"]))

        total_coref_clusters += rec["coref_cluster_count"]
        if rec["has_coreferences"]:
            samples_with_coref += 1

        if rec["entity_count"] == 0:
            unannotated += 1

        # Co-occurring label pairs
        unique_labels = sorted(set(rec["labels"]))
        if len(unique_labels) >= 2:
            for pair in combinations(unique_labels, 2):
                pair_counts[pair] += 1

    return {
        "total_samples": total,
        "languages": languages,
        "countries": countries,
        "label_counts": label_counts,
        "unknown_labels": unknown_labels,
        "entity_total": sum(entity_counts),
        "entity_avg": sum(entity_counts) / total if total else 0,
        "entity_min": min(entity_counts) if entity_counts else 0,
        "entity_max": max(entity_counts) if entity_counts else 0,
        "text_len_min": min(text_lengths) if text_lengths else 0,
        "text_len_max": max(text_lengths) if text_lengths else 0,
        "text_len_avg": sum(text_lengths) / total if total else 0,
        "text_len_median": _median(text_lengths) if text_lengths else 0,
        "total_coref_clusters": total_coref_clusters,
        "samples_with_coref": samples_with_coref,
        "coref_pct": 100 * samples_with_coref / total if total else 0,
        "unannotated": unannotated,
        "top_label_pairs": pair_counts.most_common(10),
        "label_value_counts": label_value_counts,
    }


def print_report(stats: dict, samples_dir: str) -> None:
    """Print a formatted statistics report to stdout."""
    total = stats["total_samples"]
    sep = "=" * 70

    print(f"\n{sep}")
    print("DATASET ANALYSIS REPORT")
    print(sep)
    print(f"  Directory:        {samples_dir}")
    print(f"  Total samples:    {total:,}")
    print(f"  Unannotated:      {stats['unannotated']:,}")

    # Entity statistics
    print(f"\n{'— Entity Statistics ':-<70}")
    print(f"  Total entities:   {stats['entity_total']:,}")
    print(f"  Avg per sample:   {stats['entity_avg']:.1f}")
    print(f"  Min per sample:   {stats['entity_min']:,}")
    print(f"  Max per sample:   {stats['entity_max']:,}")

    # Text length statistics
    print(f"\n{'— Text Length (chars) ':-<70}")
    print(f"  Min:              {stats['text_len_min']:,}")
    print(f"  Max:              {stats['text_len_max']:,}")
    print(f"  Avg:              {stats['text_len_avg']:.0f}")
    print(f"  Median:           {stats['text_len_median']:.0f}")

    # Coreference statistics
    print(f"\n{'— Coreference Statistics ':-<70}")
    print(f"  Total clusters:   {stats['total_coref_clusters']:,}")
    print(
        f"  Samples w/ coref: {stats['samples_with_coref']:,} ({stats['coref_pct']:.1f}%)"
    )

    # Language distribution
    print(f"\n{'— Language Distribution ':-<70}")
    print(f"  {'Language':<25} {'Count':>8}  {'%':>6}")
    print(f"  {'-' * 25} {'-' * 8}  {'-' * 6}")
    for lang, count in stats["languages"].most_common():
        pct = 100 * count / total
        print(f"  {lang:<25} {count:>8,}  {pct:>5.1f}%")

    # Country distribution
    print(f"\n{'— Country Distribution ':-<70}")
    print(f"  {'Country':<25} {'Count':>8}  {'%':>6}")
    print(f"  {'-' * 25} {'-' * 8}  {'-' * 6}")
    for country, count in stats["countries"].most_common():
        pct = 100 * count / total
        print(f"  {country:<25} {count:>8,}  {pct:>5.1f}%")

    # PII label distribution
    print(f"\n{'— PII Label Distribution ':-<70}")
    print(f"  {'Label':<25} {'Count':>8}  {'%':>6}")
    print(f"  {'-' * 25} {'-' * 8}  {'-' * 6}")
    total_entities = stats["entity_total"] or 1
    for label, count in stats["label_counts"].most_common():
        pct = 100 * count / total_entities
        print(f"  {label:<25} {count:>8,}  {pct:>5.1f}%")

    # Unknown labels
    if stats["unknown_labels"]:
        print(f"\n{'— Non-Standard Labels ':-<70}")
        print(f"  {'Label':<25} {'Count':>8}")
        print(f"  {'-' * 25} {'-' * 8}")
        for label, count in stats["unknown_labels"].most_common():
            print(f"  {label:<25} {count:>8,}")

    # Top co-occurring label pairs
    if stats["top_label_pairs"]:
        print(f"\n{'— Top Co-occurring Label Pairs ':-<70}")
        print(f"  {'Pair':<40} {'Samples':>8}")
        print(f"  {'-' * 40} {'-' * 8}")
        for (a, b), count in stats["top_label_pairs"]:
            print(f"  {a + ' + ' + b:<40} {count:>8,}")

    # Per-label entity value diversity
    label_value_counts = stats.get("label_value_counts", {})
    if label_value_counts:
        print(f"\n{'— Entity Value Diversity ':-<70}")
        print(f"  {'Label':<25} {'Unique':>8}")
        print(f"  {'-' * 25} {'-' * 8}")
        for lbl in LabelUtils.STANDARD_PII_LABELS:
            vc = label_value_counts.get(lbl)
            if vc:
                print(f"  {lbl:<25} {len(vc):>8,}")

        # Top 5 most frequent values per label
        print(f"\n{'— Top 5 Most Frequent Values Per Label ':-<70}")
        for lbl in LabelUtils.STANDARD_PII_LABELS:
            vc = label_value_counts.get(lbl)
            if not vc:
                continue
            top5 = vc.most_common(5)
            print(f"\n  {lbl}:")
            for val, count in top5:
                display = val if len(val) <= 40 else val[:37] + "..."
                print(f"    {display:<42} {count:>6,}")

    print(f"\n{sep}\n")


def main() -> None:
    """CLI entry point."""
    default_dir = Path(__file__).parent / "data_samples" / "reviewed_samples"

    parser = argparse.ArgumentParser(
        description="Analyze annotated PII datasets and print statistics."
    )
    parser.add_argument(
        "--samples-dir",
        default=str(default_dir),
        help="Path to directory containing JSON samples (default: %(default)s)",
    )
    args = parser.parse_args()

    samples_dir = Path(args.samples_dir)
    if not samples_dir.exists():
        print(f"Error: directory not found: {samples_dir}")
        sys.exit(1)

    records = load_and_extract(samples_dir)
    stats = compute_stats(records)
    print_report(stats, str(samples_dir))


if __name__ == "__main__":
    main()
