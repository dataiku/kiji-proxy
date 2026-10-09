"""Build Label Studio import tasks from Clef-labeled samples."""

import json
import re
from pathlib import Path
from typing import Any

from ..label_utils import LabelUtils
from .config import GENERATION_LANGUAGES, NONE_LABEL

ENTITIES_FROM_NAME = "entities"
TEXT_TO_NAME = "text"


def _is_word_char(char: str) -> bool:
    return char.isalnum() or char == "_"


def find_span_offsets(
    text: str, value: str, script: str = "latin"
) -> list[tuple[int, int]]:
    """Return ``(start, end)`` for every occurrence of ``value`` in ``text``.

    For Latin-script text a value that starts and ends with a word character is
    only matched at word boundaries (so "Ann" does not match inside "Anna").
    CJK text has no word delimiters, so plain substring matching is used.
    """
    if not value:
        return []
    use_boundaries = (
        script != "cjk" and _is_word_char(value[0]) and _is_word_char(value[-1])
    )
    if use_boundaries:
        pattern = r"(?<![\w])" + re.escape(value) + r"(?![\w])"
        offsets = [(m.start(), m.end()) for m in re.finditer(pattern, text)]
        if offsets:
            return offsets
    offsets = []
    start = 0
    while True:
        pos = text.find(value, start)
        if pos == -1:
            return offsets
        offsets.append((pos, pos + len(value)))
        start = pos + 1


def build_regions(
    text: str, privacy_mask: list[dict[str, Any]], script: str = "latin"
) -> list[dict[str, Any]]:
    """Create non-overlapping Label Studio ``labels`` regions for ``privacy_mask``.

    Longer values are placed first so that "Harbour Lane" wins over "Lane".
    """
    ordered = sorted(privacy_mask, key=lambda item: len(item["value"]), reverse=True)
    taken: list[tuple[int, int]] = []
    regions: list[dict[str, Any]] = []
    for item in ordered:
        label = item["label"]
        if label == NONE_LABEL:
            continue
        for start, end in find_span_offsets(text, item["value"], script):
            if any(start < t_end and end > t_start for t_start, t_end in taken):
                continue
            taken.append((start, end))
            region: dict[str, Any] = {
                "from_name": ENTITIES_FROM_NAME,
                "to_name": TEXT_TO_NAME,
                "type": "labels",
                "value": {
                    "start": start,
                    "end": end,
                    "text": text[start:end],
                    "labels": [label],
                },
            }
            if item.get("score") is not None:
                region["score"] = float(item["score"])
            regions.append(region)
    regions.sort(key=lambda region: region["value"]["start"])
    for index, region in enumerate(regions, start=1):
        region["id"] = f"ent-{index}"
    return regions


def build_labelstudio_task(
    sample: dict[str, Any],
    min_score: float = 0.0,
    model_version: str | None = None,
) -> dict[str, Any]:
    """Convert a labeled sample into one Label Studio task with predictions.

    Args:
        sample: Sample produced by the pipeline (``text``, ``privacy_mask`` with
            per-entity ``score``, and metadata).
        min_score: Entities whose Clef probability is below this are left out of
            the predictions (they stay in the sample file for inspection).
        model_version: Overrides the prediction ``model_version``.
    """
    text = sample["text"]
    language = sample.get("language")
    script = sample.get("script") or GENERATION_LANGUAGES.get(language, {}).get(
        "script", "latin"
    )
    kept = [
        item
        for item in sample.get("privacy_mask", [])
        if item.get("label") != NONE_LABEL
        and float(item.get("score", 1.0)) >= min_score
    ]
    regions = build_regions(text, kept, script)
    scores = [r["score"] for r in regions if "score" in r]
    prediction_score = round(sum(scores) / len(scores), 4) if scores else 1.0

    version = (
        model_version
        or "+".join(
            part
            for part in (sample.get("generator_model"), sample.get("labeler_model"))
            if part
        )
        or "qwen+clef"
    )

    data = {"text": text}
    for key in ("language", "language_code", "country", "text_type", "file_name"):
        if sample.get(key) is not None:
            data[key] = sample[key]

    agreement = None
    decided = [
        item for item in sample.get("privacy_mask", []) if item.get("author_label")
    ]
    if decided:
        agreement = round(
            sum(1 for i in decided if i["author_label"] == i["label"]) / len(decided),
            4,
        )

    meta = {
        "generator_model": sample.get("generator_model"),
        "labeler_model": sample.get("labeler_model"),
        "sample_index": sample.get("sample_index"),
        "missing_pii_probability": sample.get("missing_pii_probability"),
        "author_agreement": agreement,
        "rejected_candidates": [
            item["value"] for item in sample.get("rejected_candidates", [])
        ],
    }

    return {
        "data": data,
        "meta": {k: v for k, v in meta.items() if v is not None},
        "predictions": [
            {"model_version": version, "score": prediction_score, "result": regions}
        ],
    }


def write_labelstudio_import(tasks: list[dict[str, Any]], path: str | Path) -> Path:
    """Write ``tasks`` as a single JSON array that Label Studio imports directly."""
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with open(output, "w", encoding="utf-8") as f:
        json.dump(tasks, f, indent=2, ensure_ascii=False)
    return output


def write_labeling_config(path: str | Path, granularity: str = "symbol") -> Path:
    """Write a labeling config whose granularity suits CJK text (default: symbol)."""
    output = Path(path)
    LabelUtils.generate_labelstudio_config(output_path=output, granularity=granularity)
    return output
