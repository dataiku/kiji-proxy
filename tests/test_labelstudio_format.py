from __future__ import annotations

from model.dataset.labelstudio.labelstudio_format import convert_sample_to_labelstudio


def _spans(sample: dict) -> list[tuple[int, int, str, str]]:
    converted = convert_sample_to_labelstudio(sample)
    out = []
    for item in converted["predictions"][0]["result"]:
        value = item.get("value")
        if value and value.get("labels"):
            out.append(
                (value["start"], value["end"], value["labels"][0], value["text"])
            )
    return sorted(out)


def test_repeated_name_is_labelled_at_every_occurrence():
    text = "Anna Müller hat angerufen. Frau Müller bittet um Rückruf. Grüße, Anna"
    spans = _spans(
        {
            "text": text,
            "privacy_mask": [
                {"value": "Anna", "label": "FIRSTNAME"},
                {"value": "Müller", "label": "SURNAME"},
            ],
        }
    )
    assert [(text[s:e], lbl) for s, e, lbl, _ in spans] == [
        ("Anna", "FIRSTNAME"),
        ("Müller", "SURNAME"),
        ("Müller", "SURNAME"),
        ("Anna", "FIRSTNAME"),
    ]


def test_repeat_inside_longer_word_is_not_labelled():
    text = "Herr Berg wohnt in Bergheim. Berg"
    spans = _spans(
        {"text": text, "privacy_mask": [{"value": "Berg", "label": "SURNAME"}]}
    )
    assert [s for s, _, _, _ in spans] == [5, 29]


def test_short_numbers_are_not_repeated():
    text = "Hauptstraße 12, Termin am 12. März"
    spans = _spans(
        {
            "text": text,
            "privacy_mask": [
                {"value": "Hauptstraße", "label": "STREET"},
                {"value": "12", "label": "BUILDINGNUM"},
            ],
        }
    )
    assert [(text[s:e], lbl) for s, e, lbl, _ in spans] == [
        ("Hauptstraße", "STREET"),
        ("12", "BUILDINGNUM"),
    ]


def test_given_offsets_are_used():
    text = "Kopie an 4711, Kundennummer 4711"
    spans = _spans(
        {
            "text": text,
            "privacy_mask": [
                {"value": "4711", "label": "ZIP", "start": 28, "end": 32},
            ],
        }
    )
    assert spans == [(28, 32, "ZIP", "4711")]


def test_wrong_offsets_fall_back_to_search():
    text = "Ich heiße Jonas."
    spans = _spans(
        {
            "text": text,
            "privacy_mask": [
                {"value": "Jonas", "label": "FIRSTNAME", "start": 0, "end": 5}
            ],
        }
    )
    assert spans == [(10, 15, "FIRSTNAME", "Jonas")]


def test_repeat_overlapping_a_labelled_value_is_skipped():
    text = "Müller GmbH, Ansprechpartnerin Frau Müller"
    spans = _spans(
        {
            "text": text,
            "privacy_mask": [
                {"value": "Müller GmbH", "label": "COMPANYNAME"},
                {"value": "Müller", "label": "SURNAME", "start": 36, "end": 42},
            ],
        }
    )
    assert [(text[s:e], lbl) for s, e, lbl, _ in spans] == [
        ("Müller GmbH", "COMPANYNAME"),
        ("Müller", "SURNAME"),
    ]


def test_pii_free_rows_are_kept_only_when_marked():
    from model.src.preprocessing import DatasetProcessor

    convert = DatasetProcessor.convert_labelstudio_to_training_format
    empty = {"predictions": [{"result": []}]}
    marked = {"data": {"text": "Termin am 12.03.", "pii_free": True}, **empty}
    unmarked = {"data": {"text": "Termin am 12.03."}, **empty}

    assert convert(marked, "a.json") == {
        "text": "Termin am 12.03.",
        "privacy_mask": [],
        "language": None,
        "country": None,
    }
    assert convert(unmarked, "b.json") is None
