from __future__ import annotations

import random
import re

import pytest

from model.dataset.id_templates import (
    EIN_CUES,
    EIN_PREFIXES,
    FEMALE_NAMES,
    ID_LABELS,
    REFERENCES,
    STATE_FORMATS,
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
        phrases = {lab: c.cues[lab] + c.keys[lab] for lab in ID_LABELS[c.code]}
        if c is US:
            phrases["TAXNUM"] = phrases["TAXNUM"] + EIN_CUES  # business accounts
        cues = {label: _cue_re(p) for label, p in phrases.items()}
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
        ("MacLeod", "Hamish", False, "MCLEO807035H"),
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


def _spans(r: dict, label: str) -> list[str]:
    return [m["value"] for m in r["privacy_mask"] if m["label"] == label]


def test_uk_and_us_rows_name_their_country_as_kiji_does():
    """Kiji labels Scotland, Wales and Northern Ireland STATE, so UK rows use
    only "United Kingdom" / "UK"; "U.S." would be relabelled inside the cue
    "U.S. passport no." by the Label Studio converter."""
    uk_nations = {"England", "Scotland", "Wales", "Northern Ireland"}
    for r in ROWS:
        assert not set(_spans(r, "COUNTRY")) & uk_nations
        assert "U.S." not in _spans(r, "COUNTRY")


def test_person_birth_sex_drives_the_dvla_month():
    for country in (UK, US):
        g = _IdGen(random.Random(1), country)
        people = [g.person() for _ in range(300)]
        assert all(p["female"] == (p["first"] in FEMALE_NAMES) for p in people)
        assert 0.3 < sum(p["female"] for p in people) / len(people) < 0.7


def test_a_us_person_has_an_ssn_or_an_itin_not_both():
    for r in ROWS:
        if r["country"] != US.country:
            continue
        itins = [v for v in _spans(r, "TAXNUM") if re.fullmatch(r"9\d\d-\d\d-\d{4}", v)]
        assert not (itins and _spans(r, "SSN")), r["text"]


def test_eins_use_assigned_prefixes():
    eins = [
        v for r in ROWS for v in _spans(r, "TAXNUM") if re.fullmatch(r"\d\d-\d{7}", v)
    ]
    assert eins
    assert all(v[:2] in EIN_PREFIXES for v in eins)


def _format_re(pattern: str) -> re.Pattern:
    rx = re.escape(pattern)
    for kind, chars in (("d", r"\d"), ("C", "[A-Z]")):
        rx = re.sub(
            r"\\\{" + kind + r"(\d+)\\\}", lambda m, c=chars: f"{c}{{{m.group(1)}}}", rx
        )
    return re.compile(rx)


def test_us_licences_zips_and_states_agree():
    zip3 = {(state, city): z for state, city, z in US.cities}
    checked = 0
    for r in ROWS:
        if r["country"] != US.country:
            continue
        mask = sorted(r["privacy_mask"], key=lambda m: m["start"])
        for a, b, c in zip(mask, mask[1:], mask[2:], strict=False):
            if (a["label"], b["label"], c["label"]) == ("CITY", "STATE", "ZIP"):
                assert c["value"][:3] == zip3[(b["value"], a["value"])]
                checked += 1
        hit = re.search(r": (\S[^\n]*), issued in ([A-Z]{2})\n", r["text"])
        if hit:
            assert _format_re(STATE_FORMATS[hit.group(2)]).fullmatch(hit.group(1))
            checked += 1
    assert checked > 100


_BIRTH_WORD = re.compile(r"birth|born|\bdob\b", re.IGNORECASE)


def test_many_birth_dates_have_no_birth_word():
    """Cue-only birth dates teach "no birth word, no birth date"."""
    total = uncued = 0
    for r in ROWS:
        for m in r["privacy_mask"]:
            if m["label"] != "DATEOFBIRTH":
                continue
            header = _csv_header_key(r["text"], m["start"])
            line_start = r["text"].rfind("\n", 0, m["start"]) + 1
            before = (
                header if header is not None else r["text"][line_start : m["start"]]
            )
            total += 1
            uncued += not _BIRTH_WORD.search(before[-40:])
    assert 0.3 < uncued / total < 0.6


def test_pii_free_rows_are_not_a_few_fixed_frames():
    free = [r["text"] for r in ROWS if r["pii_free"]]
    skeletons = {re.sub(r"[A-Z0-9]{2,}|\d", "#", t) for t in free}
    assert len(skeletons) > 0.5 * len(free)


def test_sentences_are_not_broken():
    for r in ROWS:
        assert ".." not in r["text"]
        assert not re.search(r" is \S+ (matches|has been)", r["text"]), r["text"]
