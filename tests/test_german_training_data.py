from __future__ import annotations

import re

from model.dataset.huggingface.german_sources import (
    convert_nym_row,
    convert_openpii_row,
)
from model.dataset.synthetic_german import generate


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


def test_synthetic_rows_have_valid_offsets_and_pii_free_rows():
    rows = generate(500, seed=1)
    assert len(rows) == 500
    assert any(r["pii_free"] for r in rows) and any(not r["pii_free"] for r in rows)
    for r in rows:
        assert r["pii_free"] == (not r["privacy_mask"])
        for m in r["privacy_mask"]:
            assert r["text"][m["start"] : m["end"]] == m["value"]


def test_synthetic_rows_avoid_probe_codes():
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
    ]
    text = "\n".join(r["text"] for r in generate(3000, seed=2))
    assert not [
        c for c in probe_codes if re.search(rf"(?<!\w){re.escape(c)}(?!\w)", text)
    ]
