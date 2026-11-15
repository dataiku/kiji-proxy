"""German and English evaluation sets for the offline PII benchmark.

Every loader returns samples in the benchmark format::

    {"text": str, "entities": [(start, end, kiji_label), ...], "values": [str, ...]}

``values`` holds the source's own entity strings (parallel to ``entities``) so
offsets can be checked with ``text[start:end] == value``; the benchmark ignores
it. A sample may also carry ``unscored``: spans the source annotates with a
label Kiji does not have (IP addresses, times, ...). Predictions on them are
neither true nor false positives. See ``tests/benchmark/GERMAN_EVAL.md`` for
licences, label mappings and contamination notes.

Usage (sanity check, no model inference):
    uv run python -m tests.benchmark.german_eval --check --num 5
    uv run python -m tests.benchmark.german_eval --stats --source nobody-de
"""

from __future__ import annotations

import argparse
import json
import random
import re
import sys
from collections import Counter
from collections.abc import Callable
from dataclasses import dataclass
from functools import partial
from pathlib import Path

Span = tuple[int, int, str]

PROBES_PATH = Path(__file__).parent / "data" / "german_probes.jsonl"

NOBODY_REPO = "naeyn/nobody-pii-synth-de"
# Revision whose Parquet hashes are published on the dataset card.
NOBODY_REVISION = "23eba3fadef50ef4c01738bea0b0ee1c392426e6"
NYM_REPO = "Wismut/nym-pii-multilingual-data"
AI4PRIVACY_REPO = "ai4privacy/pii-masking-300k"
AI4PRIVACY_FILES = {
    "de": {
        "train": "data/train/german_openpii_30k.jsonl",
        "validation": "data/validation/german_openpii_8k.jsonl",
    },
    "en": {
        "train": "data/train/1english_openpii_30k.jsonl",
        "validation": "data/validation/1english_openpii_8k.jsonl",
    },
}

# ---------------------------------------------------------------------------
# Label mappings
# ---------------------------------------------------------------------------

# naeyn/nobody-pii-synth-de has coarse labels: one span per full name and one
# per full postal address. Gold keeps those coarse spans; Kiji predictions are
# projected onto the same scheme with KIJI_TO_COARSE + merge_coarse_spans.
NOBODY_TO_COARSE: dict[str, str] = {
    "person": "PERSON",
    "address": "ADDRESS",
    "email": "EMAIL",
    "phone number": "PHONENUMBER",
    "iban": "IBAN",
    "date of birth": "DATEOFBIRTH",
    "organization": "COMPANYNAME",
}

KIJI_TO_COARSE: dict[str, str] = {
    "FIRSTNAME": "PERSON",
    "SURNAME": "PERSON",
    "STREET": "ADDRESS",
    "BUILDINGNUM": "ADDRESS",
    "ZIP": "ADDRESS",
    "CITY": "ADDRESS",
    "STATE": "ADDRESS",
    # nobody-de address spans include a trailing "Deutschland" when present.
    "COUNTRY": "ADDRESS",
}

# Text allowed between two adjacent predicted spans that should be merged into
# one coarse span ("Max Müller", "Hauptstraße 12, 80331 München").
COARSE_JOINERS: dict[str, re.Pattern[str]] = {
    "PERSON": re.compile(r"[ \t.\-]*"),
    "ADDRESS": re.compile(r"[\s,·/\-]*"),
}

# Wismut/nym-pii-multilingual-data → Kiji. None = dropped (no Kiji equivalent).
NYM_TO_KIJI: dict[str, str | None] = {
    "GIVEN_NAME": "FIRSTNAME",
    "SURNAME": "SURNAME",
    "EMAIL": "EMAIL",
    "PHONE": "PHONENUMBER",
    "FAX_NUMBER": "PHONENUMBER",
    "USERNAME": "USERNAME",
    "URL": "URL",
    "COMPANY_NAME": "COMPANYNAME",
    # STREET_ADDRESS is "street + number"; split by split_street_address().
    "STREET_ADDRESS": "STREET",
    "STREET_NAME": "STREET",
    "SECONDARY_ADDRESS": "STREET",
    "BUILDING_NUMBER": "BUILDINGNUM",
    "CITY": "CITY",
    "STATE": "STATE",
    "ZIP_CODE": "ZIP",
    "COUNTRY": "COUNTRY",
    "DATE_OF_BIRTH": "DATEOFBIRTH",
    "AGE": "AGE",
    "SSN": "SSN",
    "GOVERNMENT_ID": "NATIONALID",
    "PASSPORT": "PASSPORTID",
    "DRIVERS_LICENSE": "DRIVERLICENSENUM",
    "TAX_ID": "TAXNUM",
    "LICENSE_PLATE": "LICENSEPLATENUM",
    "IBAN": "IBAN",
    "CREDIT_DEBIT_CARD": "CREDITCARDNUMBER",
    "PASSWORD": "PASSWORD",
    "API_KEY": "SECURITYTOKEN",
    # --- Unmapped (dropped) ---
    "DATE": None,
    "TIME": None,
    "GENDER": None,
    "CUSTOMER_ID": None,
    "EMPLOYEE_ID": None,
    "ACCOUNT_NUMBER": None,
    "ROUTING_NUMBER": None,
    "SWIFT_BIC": None,
    "CVV": None,
    "PIN": None,
    "MEDICAL_RECORD_NUMBER": None,
    "MAC_ADDRESS": None,
}

# Dropped labels that are still personal identifiers. A Kiji mask on one of
# them is right to mask, so it is left out of scoring instead of counted as a
# false positive. Dates, times, gender, titles and card issuers stay scored:
# masking an ordinary date or "Mr" is over-masking.
UNSCORED_NYM_LABELS = frozenset(
    {
        "CUSTOMER_ID",
        "EMPLOYEE_ID",
        "ACCOUNT_NUMBER",
        "ROUTING_NUMBER",
        "SWIFT_BIC",
        "CVV",
        "PIN",
        "MEDICAL_RECORD_NUMBER",
        "MAC_ADDRESS",
    }
)
UNSCORED_AI4PRIVACY_LABELS = frozenset({"IP", "GEOCOORD"})

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_STREET_THEN_NUMBER = re.compile(
    r"^(?P<street>.*\S)\s+(?P<num>\d+\s?[a-zA-Z]?(?:\s?[-/]\s?\d+\s?[a-zA-Z]?)?)$"
)
_NUMBER_THEN_STREET = re.compile(r"^(?P<num>\d+[a-zA-Z]?)\s+(?P<street>\S.*)$")


def split_street_address(text: str, start: int, end: int) -> list[Span]:
    """Split a "street + house number" span into Kiji STREET and BUILDINGNUM."""
    value = text[start:end]
    for pattern in (_STREET_THEN_NUMBER, _NUMBER_THEN_STREET):
        m = pattern.match(value)
        if m:
            return sorted(
                [
                    (start + m.start("street"), start + m.end("street"), "STREET"),
                    (start + m.start("num"), start + m.end("num"), "BUILDINGNUM"),
                ]
            )
    return [(start, end, "STREET")]


def merge_coarse_spans(text: str, spans: list[Span]) -> list[Span]:
    """Project Kiji spans onto coarse labels and merge adjacent fragments."""
    projected = sorted((s, e, KIJI_TO_COARSE.get(lbl, lbl)) for s, e, lbl in spans)
    merged: list[Span] = []
    for span in projected:
        if merged:
            last = merged[-1]
            joiner = COARSE_JOINERS.get(span[2])
            if (
                joiner is not None
                and last[2] == span[2]
                and last[1] <= span[0]
                and joiner.fullmatch(text[last[1] : span[0]])
            ):
                merged[-1] = (last[0], span[1], last[2])
                continue
        merged.append(span)
    return merged


_DE_WORDS = re.compile(
    r"\b(und|der|die|das|nicht|ist|mit|für|sie|ich|bitte|wir|ihre|ihr|auf|den|dem"
    r"|von|zu|eine|ein|einen|bei|unter|wurde|sind|haben|hat|wird|können|kann"
    r"|meine|mein|zum|zur|ihnen|herr|frau|sehr|geehrte|grüße|werden|auch|oder)\b",
    re.IGNORECASE,
)
_OTHER_LANG_WORDS = [
    re.compile(rf"\b({words})\b", re.IGNORECASE)
    for words in (
        "the|and|is|for|with|please|your|of|to|this|that|are|was|be",  # en
        "het|een|niet|voor|met|wij|uw|zijn|van|naar|ik|je|deze|dat|wordt",  # nl
        "og|ikke|jeg|er|til|af|på|med|det|den|som|har",  # da
        "och|inte|jag|är|till|av|på|med|det|som|har|för",  # sv
        "le|la|les|et|est|pour|avec|vous|nous|une|des|du|au",  # fr
        "il|la|che|è|per|con|non|una|sono|di|del|della",  # it
        "el|la|que|es|para|con|por|una|los|las|del",  # es
        "o|que|não|para|com|uma|os|do|da|em|é",  # pt
    )
]


def looks_german(text: str) -> bool:
    """Function-word heuristic; nym rows carry no language field."""
    de_words = [w.lower() for w in _DE_WORDS.findall(text)]
    other = max(len(p.findall(text)) for p in _OTHER_LANG_WORDS)
    return len(set(de_words)) >= 2 and len(de_words) >= 3 and len(de_words) > 2 * other


def looks_english(text: str) -> bool:
    """Same heuristic as looks_german, for English function words."""
    en_words = [w.lower() for w in _OTHER_LANG_WORDS[0].findall(text)]
    other = max(len(p.findall(text)) for p in [_DE_WORDS, *_OTHER_LANG_WORDS[1:]])
    return len(set(en_words)) >= 2 and len(en_words) >= 4 and len(en_words) > 2 * other


def token_char_offsets(text: str, tokens: list[str]) -> list[tuple[int, int]] | None:
    """Align whitespace/punctuation tokens back to character offsets."""
    offsets: list[tuple[int, int]] = []
    cursor = 0
    for token in tokens:
        idx = text.find(token, cursor)
        if idx < 0 or text[cursor:idx].strip():
            return None
        offsets.append((idx, idx + len(token)))
        cursor = idx + len(token)
    return offsets


def _load_dataset(*args, **kwargs):
    """load_dataset with a fallback to anonymous Hub access (expired tokens)."""
    from datasets import load_dataset
    from datasets.exceptions import DatasetNotFoundError
    from huggingface_hub.errors import HfHubHTTPError

    try:
        return load_dataset(*args, **kwargs)
    # An expired token surfaces as a Hub error, or as a ValueError when
    # `datasets` falls back to a cache that lacks this config.
    except (DatasetNotFoundError, HfHubHTTPError, ValueError):
        print("  Authenticated Hub lookup failed; retrying anonymously ...")
        return load_dataset(*args, token=False, **kwargs)


def _trim(text: str, start: int, end: int) -> tuple[int, int]:
    # Some sources include a trailing space in the gold span ("Löwer "); the
    # model trims spans, so such gold could never match exactly.
    while start < end and text[start].isspace():
        start += 1
    while end > start and text[end - 1].isspace():
        end -= 1
    return start, end


def _sample(text: str, entities: list[tuple[int, int, str, str]]) -> dict:
    trimmed = []
    for start, end, label, value in entities:
        s, e = _trim(text, start, end)
        if s < e:
            trimmed.append((s, e, label, value.strip()))
    entities = sorted(trimmed)
    return {
        "text": text,
        "entities": [(s, e, lbl) for s, e, lbl, _ in entities],
        "values": [v for _, _, _, v in entities],
    }


def _take(rows: list, num: int, seed: int) -> list:
    if num <= 0 or num >= len(rows):
        return rows
    return random.Random(seed).sample(rows, num)


# ---------------------------------------------------------------------------
# Source converters
# ---------------------------------------------------------------------------


def convert_nobody_row(row: dict) -> dict | None:
    """naeyn/nobody-pii-synth-de row → sample (inclusive token spans → chars)."""
    text = row["text"]
    offsets = token_char_offsets(text, list(row["tokenized_text"]))
    if offsets is None:
        return None
    entities = []
    for ent in row["ner"]:
        start = offsets[ent["start"]][0]
        end = offsets[ent["end"]][1]
        entities.append((start, end, NOBODY_TO_COARSE[ent["label"]], text[start:end]))
    return _sample(text, entities)


def convert_nym_row(row: dict) -> dict:
    """Wismut/nym-pii-multilingual-data row → sample with Kiji labels."""
    text = row["text"]
    entities = []
    unscored = []
    for ent in row["entities"]:
        kiji = NYM_TO_KIJI.get(ent["label"])
        if kiji is None:
            if ent["label"] in UNSCORED_NYM_LABELS:
                unscored.append((ent["start"], ent["end"]))
            continue
        start, end = ent["start"], ent["end"]
        if ent["label"] == "STREET_ADDRESS":
            parts = split_street_address(text, start, end)
        else:
            parts = [(start, end, kiji)]
        entities.extend((s, e, lbl, text[s:e]) for s, e, lbl in parts)
    sample = _sample(text, entities)
    sample["unscored"] = unscored
    return sample


def convert_probe_row(row: dict) -> dict:
    """Hand-written probe → sample; entity values are located in text order."""
    text = row["text"]
    entities = []
    cursor = 0
    for ent in row["entities"]:
        idx = text.find(ent["value"], cursor)
        if idx < 0:
            raise ValueError(f"probe {row['id']}: {ent['value']!r} not found")
        end = idx + len(ent["value"])
        entities.append((idx, end, ent["label"], ent["value"]))
        cursor = end
    sample = _sample(text, entities)
    sample["id"] = row["id"]
    sample["category"] = row["category"]
    return sample


# ---------------------------------------------------------------------------
# Source loaders
# ---------------------------------------------------------------------------


def load_nobody(num: int, seed: int, split: str, lang: str = "de") -> list[dict]:
    ds = _load_dataset(NOBODY_REPO, revision=NOBODY_REVISION, split=split)
    samples = []
    for row in ds:
        if row["lang"] != lang:
            continue
        sample = convert_nobody_row(row)
        if sample is not None:
            sample["doctype"] = row["doctype"]
            samples.append(sample)
    return _take(samples, num, seed)


def load_nym(
    num: int,
    seed: int,
    split: str,
    is_language: Callable[[str], bool] = looks_german,
) -> list[dict]:
    ds = _load_dataset(NYM_REPO, data_files={split: f"{split}.jsonl"}, split=split)
    samples = [convert_nym_row(row) for row in ds if is_language(row["text"])]
    return _take([s for s in samples if s["entities"]], num, seed)


def unscored_ai4privacy_spans(row: dict) -> list[tuple[int, int]]:
    """Spans of ai4privacy IP addresses and coordinates.

    The label mapping drops them because Kiji has no such label, so a model
    that masks an IP address would otherwise be charged a false positive.
    """
    text = row.get("source_text", "")
    spans = []
    for entity in row.get("privacy_mask", []):
        if entity.get("label") not in UNSCORED_AI4PRIVACY_LABELS:
            continue
        start, end = entity.get("start"), entity.get("end")
        if (
            start is not None
            and end is not None
            and text[start:end] == entity.get("value")
        ):
            spans.append((start, end))
    return spans


def load_ai4privacy(num: int, seed: int, split: str, lang: str = "de") -> list[dict]:
    from model.dataset.huggingface.import_ai4privacy import convert_ai4privacy_sample

    ds = _load_dataset(
        AI4PRIVACY_REPO,
        data_files={split: AI4PRIVACY_FILES[lang][split]},
        split=split,
    )
    samples = []
    for row in ds:
        converted = convert_ai4privacy_sample(row)
        if converted is None:
            continue
        entities = [
            (e["start"], e["end"], e["label"], e["value"])
            for e in converted["privacy_mask"]
        ]
        sample = _sample(converted["text"], entities)
        sample["unscored"] = unscored_ai4privacy_spans(row)
        samples.append(sample)
    return _take(samples, num, seed)


def load_probes(num: int, seed: int, split: str) -> list[dict]:
    with PROBES_PATH.open(encoding="utf-8") as f:
        samples = [convert_probe_row(json.loads(line)) for line in f if line.strip()]
    return _take(samples, num, seed)


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class EvalSource:
    name: str
    description: str
    default_split: str
    splits: tuple[str, ...]
    loader: Callable[[int, int, str], list[dict]]
    # Labels the source annotates. Predictions with other labels are reported
    # separately instead of counted as false positives (None = score all).
    labels: frozenset[str] | None = None
    # "coarse": predictions go through merge_coarse_spans before scoring.
    scheme: str = "kiji"

    def project(
        self, text: str, predicted: list[Span]
    ) -> tuple[list[Span], list[Span]]:
        """Return (scored predictions, out-of-schema predictions)."""
        if self.scheme == "coarse":
            predicted = merge_coarse_spans(text, predicted)
        if self.labels is None:
            return predicted, []
        scored = [p for p in predicted if p[2] in self.labels]
        ignored = [p for p in predicted if p[2] not in self.labels]
        return scored, ignored


SOURCES: dict[str, EvalSource] = {
    source.name: source
    for source in (
        EvalSource(
            name="nobody-de",
            description=f"{NOBODY_REPO} (German rows, coarse PERSON/ADDRESS spans)",
            default_split="test",
            splits=("train", "validation", "test"),
            loader=load_nobody,
            labels=frozenset(NOBODY_TO_COARSE.values()),
            scheme="coarse",
        ),
        EvalSource(
            name="nobody-en",
            description=f"{NOBODY_REPO} (English rows, coarse PERSON/ADDRESS spans)",
            default_split="test",
            splits=("train", "validation", "test"),
            loader=partial(load_nobody, lang="en"),
            labels=frozenset(NOBODY_TO_COARSE.values()),
            scheme="coarse",
        ),
        EvalSource(
            name="nym-en",
            description=f"{NYM_REPO} (English rows by heuristic, Kiji labels)",
            default_split="test",
            splits=("train", "validation", "test"),
            loader=partial(load_nym, is_language=looks_english),
            labels=frozenset(v for v in NYM_TO_KIJI.values() if v is not None),
        ),
        EvalSource(
            name="nym-de",
            description=f"{NYM_REPO} (German rows by heuristic, Kiji labels)",
            default_split="test",
            splits=("train", "validation", "test"),
            loader=load_nym,
            labels=frozenset(v for v in NYM_TO_KIJI.values() if v is not None),
        ),
        EvalSource(
            name="ai4privacy-300k-de",
            description=f"{AI4PRIVACY_REPO} German files",
            default_split="validation",
            splits=("train", "validation"),
            loader=partial(load_ai4privacy, lang="de"),
        ),
        EvalSource(
            name="ai4privacy-300k-en",
            description=f"{AI4PRIVACY_REPO} English files",
            default_split="validation",
            splits=("train", "validation"),
            loader=partial(load_ai4privacy, lang="en"),
        ),
        EvalSource(
            name="probes-de",
            description="Hand-written German workshop probes (hard negatives, "
            "sign-off names, addresses)",
            default_split="test",
            splits=("test",),
            loader=load_probes,
        ),
    )
}


def load_german_samples(
    source: str,
    num: int,
    seed: int,
    split: str | None = None,
) -> list[dict]:
    """Load ``num`` samples (0 = all) from a German evaluation source."""
    spec = SOURCES[source]
    split = split or spec.default_split
    if split not in spec.splits:
        raise ValueError(f"{source} has no split {split!r}; choose from {spec.splits}")
    return spec.loader(num, seed, split)


# ---------------------------------------------------------------------------
# Sanity check / stats CLI
# ---------------------------------------------------------------------------


def check_offsets(samples: list[dict]) -> list[str]:
    """Return one error per entity whose offsets do not match its value.

    Also flags spans with leading/trailing whitespace, which can never match
    a prediction exactly because the model trims them.
    """
    errors = []
    for i, sample in enumerate(samples):
        text = sample["text"]
        for (start, end, label), value in zip(
            sample["entities"], sample["values"], strict=True
        ):
            if not (0 <= start < end <= len(text)) or text[start:end] != value:
                errors.append(f"sample {i} {label}: {text[start:end]!r} != {value!r}")
            elif value != value.strip():
                errors.append(f"sample {i} {label}: untrimmed span {value!r}")
    return errors


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--source", choices=sorted(SOURCES), action="append")
    ap.add_argument("--split", default=None)
    ap.add_argument("--num", type=int, default=5, help="0 = all rows.")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--check", action="store_true", help="Verify offsets.")
    ap.add_argument("--stats", action="store_true", help="Print label counts.")
    ap.add_argument("--show", type=int, default=2, help="Samples to print.")
    args = ap.parse_args()

    failed = False
    for name in args.source or sorted(SOURCES):
        spec = SOURCES[name]
        split = args.split or spec.default_split
        print(f"\n=== {name} [{split}] — {spec.description}")
        samples = load_german_samples(name, args.num, args.seed, split)
        n_ents = sum(len(s["entities"]) for s in samples)
        print(f"  {len(samples)} samples, {n_ents} entities")
        for sample in samples[: args.show]:
            print(f"  text: {sample['text'][:160]!r}")
            for (s, e, lbl), _ in zip(
                sample["entities"], sample["values"], strict=True
            ):
                print(f"    [{lbl:<16s}] {sample['text'][s:e]!r}")
        if args.stats:
            counts = Counter(lbl for s in samples for _, _, lbl in s["entities"])
            for lbl, c in counts.most_common():
                print(f"    {lbl:<18s} {c:>6d}")
        if args.check:
            errors = check_offsets(samples)
            print(f"  offset check: {'OK' if not errors else f'{len(errors)} errors'}")
            for err in errors[:10]:
                print(f"    {err}")
            failed |= bool(errors)
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
