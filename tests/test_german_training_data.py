from __future__ import annotations

import re

import pytest

from model.dataset.huggingface.german_sources import (
    convert_nym_row,
    convert_openpii_row,
)
from model.dataset.label_utils import LabelUtils
from model.dataset.synthetic_negatives import (
    LOCALES,
    PROBE_NAMES,
    generate,
)

LANGUAGES = sorted(LOCALES)


def _labels(sample: dict) -> list[tuple[str, str]]:
    return [(m["label"], m["value"]) for m in sample["privacy_mask"]]


def _openpii(text: str, ents: list[tuple[str, str]], region: str = "DE") -> dict:
    mask = []
    for label, value in ents:
        start = text.index(value)
        mask.append(
            {"label": label, "start": start, "end": start + len(value), "value": value}
        )
    return {
        "source_text": text,
        "privacy_mask": mask,
        "language": "de",
        "region": region,
    }


def test_openpii_birth_date_after_cue_is_dateofbirth():
    row = _openpii(
        "Name: Lena Graf, Geburtsdatum: 03/04/1981. Termin am 12/05/2025.",
        [
            ("GIVENNAME", "Lena"),
            ("SURNAME", "Graf"),
            ("DATE", "03/04/1981"),
            ("DATE", "12/05/2025"),
        ],
    )
    sample = convert_openpii_row(row)
    assert _labels(sample) == [
        ("FIRSTNAME", "Lena"),
        ("SURNAME", "Graf"),
        ("DATEOFBIRTH", "03/04/1981"),
    ]
    assert sample["country"] == "Germany"


def test_openpii_ambiguous_birth_date_row_is_skipped():
    row = _openpii(
        "Das Geburtsdatum des Auditors wird bis zum 12/05/2025 geprüft, Frau Graf.",
        [("DATE", "12/05/2025"), ("SURNAME", "Graf")],
    )
    assert convert_openpii_row(row) is None


@pytest.mark.parametrize(
    "text",
    [
        # No event word: may be a birth date in a record.
        "Teilnehmer: Lena Graf, 03/04/1981, aus Köln.",
        # Event word, but openpii's random year makes it a likely birth date.
        "Bitte senden Sie das Formular bis zum 28/12/1943 an Frau Graf.",
    ],
)
def test_openpii_row_with_uncertain_date_is_skipped(text):
    date = next(w.strip(",.") for w in text.split() if "/" in w)
    row = _openpii(text, [("DATE", date), ("SURNAME", "Graf")])
    assert convert_openpii_row(row) is None


def test_openpii_recent_event_date_stays_unlabelled():
    row = _openpii(
        "Bitte senden Sie das Formular bis zum 28/12/2025 an Frau Graf.",
        [("DATE", "28/12/2025"), ("SURNAME", "Graf")],
    )
    assert _labels(convert_openpii_row(row)) == [("SURNAME", "Graf")]


def test_nym_row_with_customer_id_is_skipped():
    text = "Bitte prüfen Sie die Kundennummer 88231 für Herrn Graf und die Rechnung."
    row = {
        "text": text,
        "entities": [
            {"start": 34, "end": 39, "label": "CUSTOMER_ID"},
            {"start": 50, "end": 54, "label": "SURNAME"},
        ],
    }
    assert convert_nym_row(row) is None


def test_nym_street_address_is_split():
    text = "Bitte senden Sie die Unterlagen an die Lindenstraße 5a und nicht an mich."
    start = text.index("Lindenstraße")
    row = {
        "text": text,
        "entities": [{"start": start, "end": start + 15, "label": "STREET_ADDRESS"}],
    }
    assert _labels(convert_nym_row(row)) == [
        ("STREET", "Lindenstraße"),
        ("BUILDINGNUM", "5a"),
    ]


def test_synthetic_generator_covers_the_six_kiji_languages():
    assert set(LANGUAGES) == set(LabelUtils.LANGUAGES)


@pytest.mark.parametrize("language", LANGUAGES)
def test_synthetic_rows_have_valid_offsets_and_pii_free_rows(language):
    rows = generate(500, seed=1, language=language)
    assert len(rows) == 500
    assert {r["language"] for r in rows} == {language}
    free = sum(r["pii_free"] for r in rows)
    assert 0.35 * len(rows) < free < 0.65 * len(rows)
    for r in rows:
        assert r["pii_free"] == (not r["privacy_mask"])
        for m in r["privacy_mask"]:
            assert r["text"][m["start"] : m["end"]] == m["value"]
            assert m["value"] == m["value"].strip() and m["value"]


@pytest.mark.parametrize("language", LANGUAGES)
def test_synthetic_labels_are_kiji_labels(language):
    labels = {
        m["label"]
        for r in generate(500, seed=3, language=language)
        for m in r["privacy_mask"]
    }
    assert labels <= set(LabelUtils.STANDARD_PII_LABELS)
    # Every language has the contrast labels the negatives are paired with.
    assert {"LICENSEPLATENUM", "DATEOFBIRTH", "FIRSTNAME", "SURNAME"} <= labels


_BIRTH_WORDS = re.compile(
    r"birth|born|geb|geburt|naiss|né|nac|geboor|født|fødsel", re.IGNORECASE
)


@pytest.mark.parametrize("language", LANGUAGES)
def test_synthetic_rows_have_birth_dates_without_a_birth_word(language):
    """Person records: birth dates next to a name with no birth word before
    them, so the model does not learn that birth dates always follow a cue."""
    uncued = 0
    for r in generate(1000, seed=4, language=language):
        for m in r["privacy_mask"]:
            if m["label"] == "DATEOFBIRTH":
                line_start = r["text"].rfind("\n", 0, m["start"]) + 1
                uncued += not _BIRTH_WORDS.search(r["text"][line_start : m["start"]])
    assert uncued >= 100


def test_synthetic_record_event_dates_stay_unlabelled():
    rows = generate(1000, seed=4, language="English")
    events = 0
    for r in rows:
        labelled = {(m["start"], m["end"]) for m in r["privacy_mask"]}
        for m in re.finditer(
            r"(Registered|Last updated|Joined|Created|Signed up): (\S+)", r["text"]
        ):
            assert (m.start(2), m.end(2)) not in labelled
            events += 1
    assert events > 0


def test_synthetic_rows_are_deterministic_and_differ_by_language():
    assert generate(50, seed=5, language="French") == generate(
        50, seed=5, language="French"
    )
    texts = {
        lang: {r["text"] for r in generate(50, seed=5, language=lang)}
        for lang in LANGUAGES
    }
    assert not texts["French"] & texts["Spanish"]


@pytest.mark.parametrize("language", LANGUAGES)
def test_synthetic_pools_avoid_probe_names(language):
    loc = LOCALES[language]
    pools = set(loc.first_names) | set(loc.surnames) | set(loc.streets)
    pools |= {city for _, city in loc.cities}
    assert not pools & PROBE_NAMES


@pytest.mark.parametrize("language", LANGUAGES)
def test_synthetic_rows_avoid_probe_codes(language):
    probe_codes = [
        "GA14DE",
        "P0420",
        "P0171",
        "P0016",
        "CJZA",
        "W204",
        "E46",
        "MX-5",
        "04E 115 561 H",
        "NC10-41-400",
        "A 271 050 47 47",
        "M-AB 1234",
        "HH-KL 482E",
        "B-MX 55",
    ]
    text = "\n".join(r["text"] for r in generate(3000, seed=2, language=language))
    assert not [
        c for c in probe_codes if re.search(rf"(?<!\w){re.escape(c)}(?!\w)", text)
    ]
