from __future__ import annotations

import random
import re

import pytest

from model.dataset.id_templates import (
    ID_LABELS,
    REFERENCES,
    UK,
    US,
    _IdGen,
    generate,
)
from model.dataset.label_utils import LabelUtils

ROWS = generate(1500, seed=11)
COUNTRY_OF = {UK.country: UK, US.country: US}


def _cue_re(phrases: list[str]) -> re.Pattern:
    alternatives = "|".join(
        re.escape(p) for p in sorted(phrases, key=len, reverse=True)
    )
    return re.compile(rf"(?<!\w)(?:{alternatives})(?!\w)", re.IGNORECASE)


def test_rows_have_valid_offsets_and_a_third_pii_free():
    assert len(ROWS) == 1500 and len({r["text"] for r in ROWS}) == 1500
    assert {r["country"] for r in ROWS} == {UK.country, US.country}
    assert {r["language"] for r in ROWS} == {"English"}
    free = sum(r["pii_free"] for r in ROWS)
    assert 0.25 * len(ROWS) < free < 0.42 * len(ROWS)
    for r in ROWS:
        assert r["pii_free"] == (not r["privacy_mask"])
        for m in r["privacy_mask"]:
            assert r["text"][m["start"] : m["end"]] == m["value"]
            assert m["value"] == m["value"].strip() and m["value"]


def test_labels_are_kiji_labels_and_cover_every_id_type():
    labels = {m["label"] for r in ROWS for m in r["privacy_mask"]}
    assert labels <= set(LabelUtils.STANDARD_PII_LABELS)
    for code, country in (("uk", UK), ("us", US)):
        found = {
            m["label"]
            for r in ROWS
            if r["country"] == country.country
            for m in r["privacy_mask"]
        }
        assert set(ID_LABELS[code]) | {"COUNTRY"} <= found
    uk_labels = {
        m["label"]
        for r in ROWS
        if r["country"] == UK.country
        for m in r["privacy_mask"]
    }
    assert not uk_labels & {"SSN", "IDCARDNUM"}


def _csv_header_key(text: str, start: int) -> str | None:
    """The CSV column name above the value at ``start``, if the row is CSV."""
    lines = text.split("\n")
    if "," not in lines[0] or not lines[0].startswith("first_name"):
        return None
    line_start = text.rfind("\n", 0, start) + 1
    column = text[line_start:start].count(",")
    return lines[0].split(",")[column]


def test_every_id_follows_a_cue_for_its_own_label():
    """The nearest ID cue before a number names that number's type, so the
    rows teach "label from context", not "label from shape"."""
    for r in ROWS:
        c = COUNTRY_OF[r["country"]]
        cues = {
            label: _cue_re(c.cues[label] + c.keys[label]) for label in ID_LABELS[c.code]
        }
        for m in r["privacy_mask"]:
            if m["label"] not in cues:
                continue
            header = _csv_header_key(r["text"], m["start"])
            if header is not None:
                assert header in c.keys[m["label"]], (header, m)
                continue
            line_start = r["text"].rfind("\n", 0, m["start"]) + 1
            before = r["text"][line_start : m["start"]]
            nearest = max(
                (
                    (hit.end(), label)
                    for label, rx in cues.items()
                    for hit in rx.finditer(before)
                ),
                default=(None, None),
            )
            assert nearest[1] == m["label"], (before, m)


def test_reference_numbers_after_non_id_cues_stay_unlabelled():
    cues = {cue for kinds in REFERENCES.values() for cue, _ in kinds}
    # "Unique Taxpayer Reference" is a UK tax ID cue, not a reference cue.
    rx = re.compile(
        rf"(?<!\w)(?<!Taxpayer )(?:{'|'.join(map(re.escape, cues))}) (#?[A-Z0-9-]+)",
        re.IGNORECASE,
    )
    seen = 0
    for r in ROWS:
        labelled = [(m["start"], m["end"]) for m in r["privacy_mask"]]
        for hit in rx.finditer(r["text"]):
            seen += 1
            assert not any(s < hit.end(1) and hit.start(1) < e for s, e in labelled)
    assert seen > 500


def test_cue_words_without_a_number_appear_in_pii_free_rows():
    free = [r["text"] for r in ROWS if r["pii_free"]]
    assert sum("passport" in t for t in free) > 50
    assert (
        sum(re.search(r"driving licence|driver's license", t) is not None for t in free)
        > 30
    )


def test_countries_come_from_the_whole_world():
    countries = {
        m["value"] for r in ROWS for m in r["privacy_mask"] if m["label"] == "COUNTRY"
    }
    assert len(countries) > 100


def test_ni_numbers_have_valid_prefixes():
    rx = re.compile(r"^([A-Z]{2}) ?\d{2} ?\d{2} ?\d{2} ?[A-D]$")
    for r in ROWS:
        for m in r["privacy_mask"]:
            if m["label"] == "NATIONALID":
                match = rx.match(m["value"])
                assert match, m["value"]
                prefix = match.group(1)
                assert prefix[0] not in "DFIQUV" and prefix[1] not in "DFIQUVO"
                assert prefix not in {"BG", "GB", "KN", "NK", "NT", "TN", "ZZ"}


@pytest.mark.parametrize(
    ("last", "first", "female", "expected"),
    [
        ("Bennett-Jones", "Oliver", False, "BENNE807035O"),
        ("Wood", "Freya", True, "WOOD9857035F"),
        ("O'Brien", "Siân", False, "OBRIE807035S"),
    ],
)
def test_uk_driving_licence_follows_the_dvla_pattern(last, first, female, expected):
    g = _IdGen(random.Random(0), UK)
    person = {"last": last, "first": first, "female": female, "dob_parts": (3, 7, 1985)}
    number = g.uk_driving_licence(person).replace(" ", "")
    assert len(number) == 16
    assert number.startswith(expected)
    assert re.fullmatch(r"[A-Z9]{2}9[A-Z]{2}", number[11:])


def test_ssns_and_itins_are_told_apart_by_the_cue():
    """Both look like 123-45-6789; ITINs start with 9 and are TAXNUM."""
    us = [r for r in ROWS if r["country"] == US.country]
    ssns = [m["value"] for r in us for m in r["privacy_mask"] if m["label"] == "SSN"]
    itins = [
        m["value"]
        for r in us
        for m in r["privacy_mask"]
        if m["label"] == "TAXNUM" and re.fullmatch(r"9\d\d-\d\d-\d{4}", m["value"])
    ]
    assert ssns and itins
    assert not any(v.startswith("9") or v.startswith("666") for v in ssns)


def test_generation_is_deterministic():
    assert generate(60, seed=2) == generate(60, seed=2)
    assert generate(60, seed=2) != generate(60, seed=3)
