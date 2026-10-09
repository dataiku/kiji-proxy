"""Prompt and schema for Qwen3.8-27B text generation with candidate spans."""

import json

from ..label_utils import LabelUtils
from .config import GENERATION_LANGUAGES, NONE_LABEL, TEXT_TYPES

SENTENCE_RANGES = [(2, 4), (3, 6), (4, 8), (5, 10)]

EXAMPLE = {
    "text": "Hi Priya, your parcel ships to 14 Harbour Lane, Galway today. "
    "Call 087 123 4567 if the courier from Nolan Logistics cannot reach you.",
    "candidates": [
        {"value": "Priya", "author_label": "FIRSTNAME"},
        {"value": "14", "author_label": "BUILDINGNUM"},
        {"value": "Harbour Lane", "author_label": "STREET"},
        {"value": "Galway", "author_label": "CITY"},
        {"value": "087 123 4567", "author_label": "PHONENUMBER"},
        {"value": "Nolan Logistics", "author_label": "COMPANYNAME"},
        {"value": "today", "author_label": NONE_LABEL},
    ],
}


def get_generation_schema() -> dict:
    """JSON schema for a generated sample: text plus candidate spans."""
    return {
        "type": "object",
        "properties": {
            "text": {"type": "string"},
            "candidates": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "value": {"type": "string"},
                        "author_label": {"type": "string"},
                    },
                    "required": ["value", "author_label"],
                    "additionalProperties": False,
                },
            },
        },
        "required": ["text", "candidates"],
        "additionalProperties": False,
    }


def build_generation_prompt(
    language: str,
    country: str,
    text_type: str,
    labels: dict[str, dict],
    sample_index: int = 0,
) -> str:
    """Build the Qwen prompt for one sample.

    Args:
        language: Key of ``GENERATION_LANGUAGES``
        country: Country the text should be realistic for
        text_type: Key of ``TEXT_TYPES``
        labels: PII labels (subset of ``LabelUtils.LABEL_DESCRIPTIONS``) to embed
        sample_index: Used to vary length and tone between samples
    """
    language_info = GENERATION_LANGUAGES[language]
    type_hint = TEXT_TYPES[text_type]
    low, high = SENTENCE_RANGES[sample_index % len(SENTENCE_RANGES)]

    label_lines = []
    for code, info in labels.items():
        line = f"- {code}: {info.get('description') or info.get('name') or code}."
        hints = "; ".join(hint.rstrip(". ") for hint in info.get("hints", []))
        if hints:
            line += f" {hints}."
        label_lines.append(line)
    label_block = "\n".join(label_lines)

    locale_hints = "\n".join(f"- {hint}" for hint in language_info["hints"])
    if locale_hints:
        locale_hints = f"\nLocale conventions:\n{locale_hints}\n"

    all_labels = ", ".join(LabelUtils.STANDARD_PII_LABELS)
    example = json.dumps(EXAMPLE, ensure_ascii=False, indent=2)

    return f"""Write {type_hint}, entirely in {language}, that reads as if it was written in {country}.

Requirements:
1. The text must contain {low} to {high} sentences (or equivalent lines for forms and lists) and feel natural for a {text_type.replace("_", " ")}.
2. Embed realistic, fictional personal data of these types, written the way people in {country} write them:
{label_block}
3. Vary where the personal data appears; do not open with a name every time.
4. Use native script for the whole text, including names and places. Do not add translations or transliterations.
5. Do not use markdown formatting.
{locale_hints}
Candidate spans:
- List EVERY span that could possibly be personal data or looks like an identifier: all person names, organisation names, places, address parts, dates, ages, numbers, codes, emails, URLs, usernames and phone numbers.
- Each "value" must be copied verbatim from the text (same characters, case and spacing). It must appear in the text exactly as written.
- Split full names into separate first-name and surname spans, and addresses into their components (building number, street, city, state, zip, country).
- Set "author_label" to the label you intended from this list: {all_labels}. Use "{NONE_LABEL}" for spans that are not personal data (generic organisations, product names, relative dates such as "today").
- Return each distinct value once.

Return only JSON with the keys "text" and "candidates", like this example:
{example}
"""
