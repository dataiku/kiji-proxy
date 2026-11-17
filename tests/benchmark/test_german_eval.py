"""Offline checks for the German and English evaluation loaders (no Hub access, no model)."""

from tests.benchmark.german_eval import (
    SOURCES,
    check_offsets,
    convert_nobody_row,
    convert_nym_row,
    load_german_samples,
    looks_english,
    looks_german,
    merge_coarse_spans,
    split_street_address,
    unscored_ai4privacy_spans,
)


def test_probes_offsets_match_values():
    samples = load_german_samples("probes-de", num=0, seed=0)
    assert len(samples) >= 20
    assert check_offsets(samples) == []
    categories = {s["category"] for s in samples}
    assert {"hard_negative", "signoff_name", "address", "workshop_pii"} <= categories
    assert all(not s["entities"] for s in samples if s["category"] == "hard_negative")


def test_nobody_row_token_spans_become_char_offsets():
    row = {
        "text": "Viele Grüße\nJürgen Großmann\nHauptstraße 12, 80331 München",
        "tokenized_text": [
            "Viele",
            "Grüße",
            "Jürgen",
            "Großmann",
            "Hauptstraße",
            "12",
            ",",
            "80331",
            "München",
        ],
        "ner": [
            {"start": 2, "end": 3, "label": "person"},
            {"start": 4, "end": 8, "label": "address"},
        ],
    }
    sample = convert_nobody_row(row)
    text = sample["text"]
    assert [(text[s:e], lbl) for s, e, lbl in sample["entities"]] == [
        ("Jürgen Großmann", "PERSON"),
        ("Hauptstraße 12, 80331 München", "ADDRESS"),
    ]


def test_nym_street_address_is_split():
    text = "Lieferung an Valerie-Flantz-Straße 916 in Bad Freienwalde."
    row = {
        "text": text,
        "entities": [
            {"start": 13, "end": 38, "label": "STREET_ADDRESS"},
            {"start": 42, "end": 57, "label": "CITY"},
            {"start": 0, "end": 9, "label": "CUSTOMER_ID"},  # unmapped → dropped
        ],
    }
    sample = convert_nym_row(row)
    assert [(text[s:e], lbl) for s, e, lbl in sample["entities"]] == [
        ("Valerie-Flantz-Straße", "STREET"),
        ("916", "BUILDINGNUM"),
        ("Bad Freienwalde", "CITY"),
    ]


def test_split_street_address_variants():
    text = "Butteplatz 33/93"
    assert split_street_address(text, 0, len(text)) == [
        (0, 10, "STREET"),
        (11, 16, "BUILDINGNUM"),
    ]
    text = "Postfach"
    assert split_street_address(text, 0, len(text)) == [(0, 8, "STREET")]


def test_merge_coarse_spans_joins_name_and_address_fragments():
    text = "Gruß\nG. Jopich\nHauptstraße 12, 80331 München"
    predicted = [
        (5, 6, "FIRSTNAME"),
        (8, 14, "SURNAME"),
        (15, 26, "STREET"),
        (27, 29, "BUILDINGNUM"),
        (31, 36, "ZIP"),
        (37, 44, "CITY"),
    ]
    assert merge_coarse_spans(text, predicted) == [
        (5, 14, "PERSON"),
        (15, 44, "ADDRESS"),
    ]


def test_merge_coarse_spans_keeps_separate_people_apart():
    text = "Anna Schmidt, Max Weber"
    predicted = [
        (0, 4, "FIRSTNAME"),
        (5, 12, "SURNAME"),
        (14, 17, "FIRSTNAME"),
        (18, 23, "SURNAME"),
    ]
    assert merge_coarse_spans(text, predicted) == [
        (0, 12, "PERSON"),
        (14, 23, "PERSON"),
    ]


def test_coarse_source_reports_out_of_schema_predictions():
    text = "Kennzeichen M-AB 1234, Max Weber"
    scored, ignored = SOURCES["nobody-de"].project(
        text,
        [(12, 21, "LICENSEPLATENUM"), (23, 26, "FIRSTNAME"), (27, 32, "SURNAME")],
    )
    assert scored == [(23, 32, "PERSON")]
    assert ignored == [(12, 21, "LICENSEPLATENUM")]


def test_looks_german():
    assert looks_german("Sehr geehrte Frau Böhm, bitte senden Sie mir die Unterlagen.")
    assert not looks_german("Please send the documents to your manager by Monday.")
    assert not looks_german(
        "Vă rog să configurați plata, vom configura contul de plată vom trimite."
    )


def test_looks_english():
    assert looks_english(
        "Please send the documents to your manager by the end of the week."
    )
    assert not looks_english(
        "Sehr geehrte Frau Böhm, bitte senden Sie mir die Unterlagen und den Vertrag."
    )
    assert not looks_english(
        "Bonjour, nous avons reçu votre demande et la facture du mois."
    )
    assert not looks_english("Invoice 4711")


def test_english_sources_reuse_the_german_converters():
    nobody_en, nym_en = SOURCES["nobody-en"], SOURCES["nym-en"]
    assert nobody_en.loader.keywords == {"lang": "en"}
    assert nobody_en.scheme == SOURCES["nobody-de"].scheme == "coarse"
    assert nobody_en.labels == SOURCES["nobody-de"].labels
    assert nym_en.loader.keywords == {"is_language": looks_english}
    assert nym_en.labels == SOURCES["nym-de"].labels
    assert SOURCES["ai4privacy-300k-en"].loader.keywords == {"lang": "en"}
    assert SOURCES["ai4privacy-300k-en"].default_split == "validation"


def test_unscored_ai4privacy_spans_are_ip_addresses_and_coordinates():
    text = "Login from 10.0.0.1 by anna at 14:05"
    row = {
        "source_text": text,
        "privacy_mask": [
            {"label": "IP", "value": "10.0.0.1", "start": 11, "end": 19},
            {"label": "USERNAME", "value": "anna", "start": 23, "end": 27},
            {"label": "TIME", "value": "14:05", "start": 31, "end": 36},
            # Offsets that do not match the value are skipped, as in training.
            {"label": "IP", "value": "1.2.3.4", "start": 0, "end": 7},
        ],
    }
    # TIME stays scored: masking an ordinary time is over-masking.
    assert unscored_ai4privacy_spans(row) == [(11, 19)]


def test_nym_unscored_spans_are_unmapped_identifiers_only():
    text = "Account 12345678 opened on 2024-05-01 for Anna"
    row = {
        "text": text,
        "entities": [
            {"label": "ACCOUNT_NUMBER", "start": 8, "end": 16},
            {"label": "DATE", "start": 27, "end": 37},
            {"label": "GIVEN_NAME", "start": 42, "end": 46},
        ],
    }
    sample = convert_nym_row(row)
    assert sample["entities"] == [(42, 46, "FIRSTNAME")]
    assert sample["unscored"] == [(8, 16)]
