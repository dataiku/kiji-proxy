#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.13"
# dependencies = ["httpx", "python-dotenv", "tqdm"]
# ///
"""Audit training samples using an LLM to find noisy labels.

Loads training samples (Label Studio or simplified format), sends each to an LLM
for independent PII annotation, diffs against ground truth, and generates a report.
Optionally writes corrected Label Studio JSON files.

Usage:
    # Audit custom dataset (dry run — report only)
    uv run python model/dataset/audit_dataset.py \
        --samples-dir model/dataset/data_samples/training_samples

    # Audit and write corrected files
    uv run python model/dataset/audit_dataset.py \
        --samples-dir model/dataset/data_samples/training_samples \
        --fix

    # Audit ai4privacy samples with a sample limit
    uv run python model/dataset/audit_dataset.py \
        --samples-dir model/dataset/data_samples/training_samples \
        --max-samples 200

    # Use a custom API endpoint (e.g., local vLLM or OpenRouter)
    uv run python model/dataset/audit_dataset.py \
        --samples-dir model/dataset/data_samples/training_samples \
        --api-url http://localhost:8000/v1/chat/completions \
        --api-model meta-llama/Llama-3-70B
"""

import argparse
import json
import sys
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv
from tqdm import tqdm

# Add project root to path for imports
project_root = Path(__file__).parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from model.dataset.label_utils import LabelUtils  # noqa: E402
from model.dataset.openai.api_clients import OpenAIClient  # noqa: E402

# Load .env from project root
load_dotenv(project_root / ".env")

STANDARD_LABELS = set(LabelUtils.STANDARD_PII_LABELS)


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


@dataclass
class EntityAnnotation:
    """A single PII entity annotation."""

    value: str
    label: str
    start: int | None = None
    end: int | None = None

    def key(self) -> tuple[str, str]:
        """Identity key for comparison (value, label)."""
        return (self.value.strip(), self.label)


@dataclass
class AuditIssue:
    """A single discrepancy found during audit."""

    issue_type: str  # MISSING, EXTRA, WRONG_LABEL, WRONG_BOUNDARY, INVALID_LABEL
    original: EntityAnnotation | None = None
    suggested: EntityAnnotation | None = None
    detail: str = ""


@dataclass
class SampleAuditResult:
    """Audit result for one sample."""

    file_name: str
    text: str
    issues: list[AuditIssue] = field(default_factory=list)
    original_entities: list[EntityAnnotation] = field(default_factory=list)
    llm_entities: list[EntityAnnotation] = field(default_factory=list)
    corrected_sample: dict | None = None

    @property
    def is_clean(self) -> bool:
        return len(self.issues) == 0


# ---------------------------------------------------------------------------
# Sample loading (supports both Label Studio and simplified formats)
# ---------------------------------------------------------------------------


def load_sample(path: Path) -> dict | None:
    """Load a single JSON sample and normalize to simplified format."""
    try:
        with path.open(encoding="utf-8") as f:
            raw = json.load(f)
    except (json.JSONDecodeError, OSError):
        return None

    # Label Studio format
    if "data" in raw:
        text = raw.get("data", {}).get("text", "")
        language = raw.get("data", {}).get("language", "English")
        country = raw.get("data", {}).get("country")

        results = []
        for source_key in ("annotations", "predictions"):
            source = raw.get(source_key, [])
            if source:
                results = source[0].get("result", []) or []
                break

        entities = []
        for entry in results:
            if entry.get("type") != "labels":
                continue
            value = entry.get("value", {})
            labels = value.get("labels", [])
            if labels:
                entities.append(
                    {
                        "value": value.get("text", ""),
                        "label": labels[0],
                        "start": value.get("start"),
                        "end": value.get("end"),
                    }
                )

        return {
            "text": text,
            "privacy_mask": entities,
            "language": language,
            "country": country,
            "file_name": path.name,
            "_raw": raw,
        }

    # Simplified format
    if "text" in raw and "privacy_mask" in raw:
        raw.setdefault("language", "English")
        raw.setdefault("file_name", path.name)
        raw["_raw"] = raw.copy()
        return raw

    return None


def extract_entities(privacy_mask: list[dict]) -> list[EntityAnnotation]:
    """Convert privacy_mask dicts to EntityAnnotation objects."""
    return [
        EntityAnnotation(
            value=e.get("value", ""),
            label=e.get("label", ""),
            start=e.get("start"),
            end=e.get("end"),
        )
        for e in privacy_mask
        if e.get("label") in STANDARD_LABELS
    ]


# ---------------------------------------------------------------------------
# LLM audit prompt
# ---------------------------------------------------------------------------

AUDIT_PROMPT_TEMPLATE = """You are an expert data quality auditor for a PII (Personally Identifiable Information) detection dataset.

Your task: independently annotate ALL PII entities in the given text, then compare against the existing annotations.

**Valid PII labels (use ONLY these):**
{label_list}

**Critical annotation rules:**
- Names MUST be split: "John Smith" → FIRSTNAME("John") + SURNAME("Smith")
- Every PII entity must have `value` (exact substring from text) and `label`
- Only annotate values that actually appear verbatim in the text
- Do NOT annotate generic words like "the customer" or pronouns as PII
- Include ALL instances of PII, even if mentioned only once

**Text to audit:**
\"\"\"{text}\"\"\"

**Existing annotations:**
{existing_annotations}

**Your task:**
1. Independently identify all PII entities in the text
2. Compare your annotations against the existing ones
3. For each entity, determine: CORRECT, WRONG_LABEL, MISSING (you found it but it's not in existing), or EXTRA (in existing but shouldn't be)

Return your analysis as JSON."""


def get_audit_response_schema() -> dict:
    """JSON schema for the audit response."""
    return {
        "type": "object",
        "properties": {
            "entities": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "value": {"type": "string"},
                        "label": {"type": "string"},
                    },
                    "required": ["value", "label"],
                    "additionalProperties": False,
                },
                "description": "All PII entities you found in the text",
            },
            "issues": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "issue_type": {
                            "type": "string",
                            "enum": [
                                "MISSING",
                                "EXTRA",
                                "WRONG_LABEL",
                            ],
                        },
                        "value": {"type": "string"},
                        "original_label": {"type": "string"},
                        "suggested_label": {"type": "string"},
                        "detail": {"type": "string"},
                    },
                    "required": [
                        "issue_type",
                        "value",
                        "original_label",
                        "suggested_label",
                        "detail",
                    ],
                    "additionalProperties": False,
                },
                "description": "List of discrepancies between your annotations and the existing ones",
            },
        },
        "required": ["entities", "issues"],
        "additionalProperties": False,
    }


# ---------------------------------------------------------------------------
# Diffing logic (rule-based, runs after LLM audit)
# ---------------------------------------------------------------------------


def diff_entities(
    original: list[EntityAnnotation],
    llm: list[EntityAnnotation],
) -> list[AuditIssue]:
    """Compare original and LLM entities, return discrepancies."""
    issues: list[AuditIssue] = []

    orig_by_value: dict[str, list[EntityAnnotation]] = {}
    for e in original:
        orig_by_value.setdefault(e.value.strip(), []).append(e)

    llm_by_value: dict[str, list[EntityAnnotation]] = {}
    for e in llm:
        llm_by_value.setdefault(e.value.strip(), []).append(e)

    # Check for WRONG_LABEL and EXTRA (in original but not in LLM)
    matched_llm_values: set[str] = set()
    for value, orig_ents in orig_by_value.items():
        if value in llm_by_value:
            matched_llm_values.add(value)
            llm_ents = llm_by_value[value]
            orig_labels = {e.label for e in orig_ents}
            llm_labels = {e.label for e in llm_ents}
            if orig_labels != llm_labels:
                for oe in orig_ents:
                    if oe.label not in llm_labels:
                        # Find the LLM's suggested label
                        suggested = llm_ents[0] if llm_ents else None
                        issues.append(
                            AuditIssue(
                                issue_type="WRONG_LABEL",
                                original=oe,
                                suggested=suggested,
                                detail=f'"{value}": labeled {oe.label}, LLM suggests {suggested.label if suggested else "?"}',
                            )
                        )
        else:
            # In original but LLM didn't find it — could be an EXTRA annotation
            for oe in orig_ents:
                issues.append(
                    AuditIssue(
                        issue_type="EXTRA",
                        original=oe,
                        detail=f'"{value}" ({oe.label}): in ground truth but LLM did not annotate',
                    )
                )

    # Check for MISSING (LLM found it but not in original)
    for value, llm_ents in llm_by_value.items():
        if value not in orig_by_value:
            for le in llm_ents:
                issues.append(
                    AuditIssue(
                        issue_type="MISSING",
                        suggested=le,
                        detail=f'"{value}" ({le.label}): LLM found PII not in ground truth',
                    )
                )

    return issues


# ---------------------------------------------------------------------------
# Build corrected Label Studio sample
# ---------------------------------------------------------------------------


def build_corrected_sample(
    original_raw: dict,
    llm_entities: list[EntityAnnotation],
    text: str,
) -> dict:
    """Build a corrected Label Studio JSON from LLM entities."""
    results = []
    entity_id = 1
    for ent in llm_entities:
        if ent.label not in STANDARD_LABELS:
            continue
        # Find the entity in the text to get character offsets
        start = text.find(ent.value)
        if start == -1:
            continue
        end = start + len(ent.value)
        results.append(
            {
                "id": f"ent-{entity_id}",
                "from_name": "entities",
                "to_name": "text",
                "type": "labels",
                "value": {
                    "start": start,
                    "end": end,
                    "text": ent.value,
                    "labels": [ent.label],
                },
            }
        )
        entity_id += 1

    # Preserve metadata from original
    data = original_raw.get("data", {}).copy()
    data["text"] = text

    return {
        "data": data,
        "predictions": [
            {
                "model_version": "llm-audit-corrected",
                "score": 1.0,
                "result": results,
            }
        ],
    }


# ---------------------------------------------------------------------------
# Core audit function
# ---------------------------------------------------------------------------


def audit_sample(
    sample: dict,
    client: OpenAIClient,
) -> SampleAuditResult:
    """Audit a single sample using the LLM."""
    text = sample["text"]
    file_name = sample.get("file_name", "unknown")
    original_entities = extract_entities(sample.get("privacy_mask", []))

    # Build prompt
    label_list = ", ".join(sorted(STANDARD_LABELS))
    existing_json = json.dumps(
        [{"value": e.value, "label": e.label} for e in original_entities],
        indent=2,
    )
    prompt = AUDIT_PROMPT_TEMPLATE.format(
        label_list=label_list,
        text=text,
        existing_annotations=existing_json,
    )

    result = SampleAuditResult(
        file_name=file_name,
        text=text,
        original_entities=original_entities,
    )

    try:
        response = client.review(prompt, get_audit_response_schema())
    except Exception as e:
        result.issues.append(
            AuditIssue(issue_type="ERROR", detail=f"LLM call failed: {e}")
        )
        return result

    # Parse LLM entities
    llm_raw = response.get("entities", [])
    llm_entities = [
        EntityAnnotation(value=e["value"], label=e["label"])
        for e in llm_raw
        if e.get("label") in STANDARD_LABELS
    ]
    result.llm_entities = llm_entities

    # Combine LLM-reported issues with our own diff
    diff_issues = diff_entities(original_entities, llm_entities)
    result.issues = diff_issues

    # Build corrected sample from LLM entities
    if diff_issues:
        result.corrected_sample = build_corrected_sample(
            sample.get("_raw", {}), llm_entities, text
        )

    return result


# ---------------------------------------------------------------------------
# Audit ledger (TSV persistence across runs)
# ---------------------------------------------------------------------------

LEDGER_HEADER = "file_name\tstatus\tissue_count\tissues"


def load_ledger(ledger_path: Path) -> dict[str, str]:
    """Load existing ledger and return {filename: status} for already-reviewed files."""
    reviewed: dict[str, str] = {}
    if not ledger_path.exists():
        return reviewed
    for line in ledger_path.read_text().splitlines():
        if line.startswith("file_name\t"):
            continue
        parts = line.split("\t")
        if len(parts) >= 2:
            reviewed[parts[0]] = parts[1]
    return reviewed


def _update_ledger(ledger_path: Path, results: list["SampleAuditResult"]) -> None:
    """Append new audit results to the ledger, preserving previous entries."""
    # Read existing lines (preserving full TSV content)
    existing_lines: dict[str, str] = {}  # filename -> full line
    if ledger_path.exists():
        for line in ledger_path.read_text().splitlines():
            if line.startswith("file_name\t") or not line.strip():
                continue
            parts = line.split("\t", 1)
            if parts:
                existing_lines[parts[0]] = line

    # Build new lines from results (overwrite if re-audited)
    new_filenames: set[str] = set()
    for r in results:
        status = "CLEAN" if r.is_clean else "DIRTY"
        issue_summary = "; ".join(i.detail[:80] for i in r.issues[:5])
        existing_lines[r.file_name] = (
            f"{r.file_name}\t{status}\t{len(r.issues)}\t{issue_summary}"
        )
        new_filenames.add(r.file_name)

    # Write sorted ledger
    ledger_path.parent.mkdir(parents=True, exist_ok=True)
    with ledger_path.open("w", encoding="utf-8") as f:
        f.write(LEDGER_HEADER + "\n")
        for fname in sorted(existing_lines):
            f.write(existing_lines[fname] + "\n")


# ---------------------------------------------------------------------------
# Report generation
# ---------------------------------------------------------------------------


def print_report(results: list[SampleAuditResult], output_path: Path | None) -> None:
    """Print a summary report of the audit."""
    total = len(results)
    dirty = [r for r in results if not r.is_clean]
    clean = total - len(dirty)

    issue_type_counts: Counter = Counter()
    label_issue_counts: Counter = Counter()
    for r in dirty:
        for issue in r.issues:
            issue_type_counts[issue.issue_type] += 1
            if issue.original:
                label_issue_counts[issue.original.label] += 1
            elif issue.suggested:
                label_issue_counts[issue.suggested.label] += 1

    sep = "=" * 70
    print(f"\n{sep}")
    print("DATASET AUDIT REPORT")
    print(sep)
    print(f"  Samples audited:   {total:,}")
    print(f"  Clean:             {clean:,} ({100 * clean / total:.1f}%)")
    print(f"  With issues:       {len(dirty):,} ({100 * len(dirty) / total:.1f}%)")

    print(f"\n{'— Issue Breakdown ':-<70}")
    for issue_type, count in issue_type_counts.most_common():
        print(f"  {issue_type:<20} {count:>6,}")

    print(f"\n{'— Labels Most Affected ':-<70}")
    for label, count in label_issue_counts.most_common(15):
        print(f"  {label:<25} {count:>6,}")

    # Show worst samples
    dirty_sorted = sorted(dirty, key=lambda r: len(r.issues), reverse=True)
    print(f"\n{'— Top 20 Noisiest Samples ':-<70}")
    for r in dirty_sorted[:20]:
        types = Counter(i.issue_type for i in r.issues)
        type_str = ", ".join(f"{t}:{c}" for t, c in types.most_common())
        print(f"  {r.file_name:<45} {len(r.issues):>3} issues  ({type_str})")

    # Show example issues
    print(f"\n{'— Example Issues (first 30) ':-<70}")
    shown = 0
    for r in dirty_sorted:
        for issue in r.issues:
            if shown >= 30:
                break
            print(f"  [{issue.issue_type}] {r.file_name}")
            print(f"    {issue.detail}")
            shown += 1
        if shown >= 30:
            break

    # Write/update audit ledger (TSV: filename, status, issue_count)
    # This ledger is appended to across runs so already-reviewed files are skipped.
    ledger_path = output_path.parent / "audit_ledger.tsv" if output_path else None
    if ledger_path:
        _update_ledger(ledger_path, results)
        clean_in_ledger = sum(
            1 for line in ledger_path.read_text().splitlines()
            if line.split("\t")[1:2] == ["CLEAN"]
        )
        total_in_ledger = sum(
            1 for line in ledger_path.read_text().splitlines()
            if not line.startswith("file_name\t")
        )
        print(f"\n  Ledger updated: {ledger_path}")
        print(f"  Total reviewed: {total_in_ledger:,}  (CLEAN: {clean_in_ledger:,})")
        print("  Set audit_allowlist to this path in training config.")

    print(f"\n{sep}")

    # Write detailed JSON report
    if output_path:
        report = {
            "summary": {
                "total_samples": total,
                "clean_samples": clean,
                "dirty_samples": len(dirty),
                "issue_counts": dict(issue_type_counts),
                "label_issue_counts": dict(label_issue_counts),
            },
            "dirty_samples": [
                {
                    "file_name": r.file_name,
                    "text": r.text[:200] + "..." if len(r.text) > 200 else r.text,
                    "issue_count": len(r.issues),
                    "issues": [
                        {
                            "type": i.issue_type,
                            "detail": i.detail,
                        }
                        for i in r.issues
                    ],
                }
                for r in dirty_sorted
            ],
        }
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w", encoding="utf-8") as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        print(f"\nDetailed report written to: {output_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Audit PII training samples using an LLM"
    )
    parser.add_argument(
        "--samples-dir",
        required=True,
        help="Directory containing JSON training samples",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=0,
        help="Max samples to audit (0 = all)",
    )
    parser.add_argument(
        "--fix",
        action="store_true",
        help="Write corrected Label Studio JSON files to --output-dir",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Directory for corrected files (default: <samples-dir>_corrected)",
    )
    parser.add_argument(
        "--report",
        default=None,
        help="Path for detailed JSON report (default: <samples-dir>/audit_report.json)",
    )
    parser.add_argument(
        "--api-url",
        default=None,
        help="LLM API URL (default: OpenAI)",
    )
    parser.add_argument(
        "--api-model",
        default="gpt-4.1-mini",
        help="LLM model name (default: gpt-4.1-mini)",
    )
    parser.add_argument(
        "--max-workers",
        type=int,
        default=8,
        help="Parallel workers for API calls (default: 8)",
    )
    args = parser.parse_args()

    samples_dir = Path(args.samples_dir)
    if not samples_dir.exists():
        print(f"Error: directory not found: {samples_dir}")
        sys.exit(1)

    # Load existing ledger to skip already-reviewed files
    report_path = (
        Path(args.report) if args.report else samples_dir / "audit_report.json"
    )
    ledger_path = report_path.parent / "audit_ledger.tsv"
    already_reviewed = load_ledger(ledger_path)
    if already_reviewed:
        print(f"Found {len(already_reviewed):,} already-reviewed files in {ledger_path}")

    # Load samples, skipping already-reviewed ones
    json_files = sorted(samples_dir.glob("*.json"))
    if not json_files:
        print(f"No JSON files found in {samples_dir}")
        sys.exit(1)

    # Filter out already-reviewed files
    json_files = [f for f in json_files if f.name not in already_reviewed]
    print(f"  {len(json_files):,} files remaining to audit")

    if args.max_samples > 0:
        json_files = json_files[: args.max_samples]

    if not json_files:
        print("All files have already been reviewed. Nothing to do.")
        sys.exit(0)

    print(f"Loading {len(json_files):,} JSON files from {samples_dir}")

    samples = []
    for f in json_files:
        s = load_sample(f)
        if s:
            samples.append(s)

    print(f"Loaded {len(samples):,} samples for audit\n")

    # Initialize LLM client
    client = OpenAIClient(model=args.api_model, api_url=args.api_url)

    # Audit samples in parallel
    results: list[SampleAuditResult] = []

    def _audit(sample: dict) -> SampleAuditResult:
        return audit_sample(sample, client)

    with ThreadPoolExecutor(max_workers=args.max_workers) as executor:
        futures = {executor.submit(_audit, s): s for s in samples}
        with tqdm(total=len(samples), desc="Auditing samples") as pbar:
            for future in as_completed(futures):
                try:
                    result = future.result()
                    results.append(result)
                except Exception as e:
                    sample = futures[future]
                    results.append(
                        SampleAuditResult(
                            file_name=sample.get("file_name", "unknown"),
                            text=sample.get("text", ""),
                            issues=[
                                AuditIssue(
                                    issue_type="ERROR",
                                    detail=str(e),
                                )
                            ],
                        )
                    )
                pbar.update(1)

    # Print report (report_path computed above when loading ledger)
    print_report(results, report_path)

    # Write corrected files if --fix
    if args.fix:
        output_dir = (
            Path(args.output_dir)
            if args.output_dir
            else Path(str(samples_dir) + "_corrected")
        )
        output_dir.mkdir(parents=True, exist_ok=True)

        written = 0
        for r in results:
            if r.corrected_sample:
                out_path = output_dir / r.file_name
                with out_path.open("w", encoding="utf-8") as f:
                    json.dump(r.corrected_sample, f, indent=2, ensure_ascii=False)
                written += 1

        print(f"\nWrote {written:,} corrected files to: {output_dir}")
        print(
            "Review the corrected files before replacing originals in your training pipeline."
        )


if __name__ == "__main__":
    main()
