from __future__ import annotations

from model.src.span_decoder import group_bio_spans, trim_span


def test_inside_tag_with_different_label_starts_entity():
    text = "Dominik Kärcher"
    tokens = ["▁Dominik", "▁K", "är", "cher"]
    offsets = [(0, 7), (7, 9), (9, 11), (11, 15)]
    labels = ["B-FIRSTNAME", "I-STREET", "I-STREET", "I-STREET"]

    spans = group_bio_spans(text, tokens, offsets, labels)

    assert [(text[s:e], label) for s, e, label in spans] == [
        ("Dominik", "FIRSTNAME"),
        ("Kärcher", "STREET"),
    ]


def test_inside_tag_after_o_starts_entity():
    text = "Hallo Kärcher"
    tokens = ["▁Hallo", "▁K", "är", "cher"]
    offsets = [(0, 5), (5, 7), (7, 9), (9, 13)]
    labels = ["O", "I-SURNAME", "I-SURNAME", "I-SURNAME"]

    spans = group_bio_spans(text, tokens, offsets, labels)

    assert [(text[s:e], label) for s, e, label in spans] == [("Kärcher", "SURNAME")]


def _trimmed(text: str, start: int, end: int) -> str:
    s, e = trim_span(text, start, end)
    return text[s:e]


def test_trim_span_drops_unbalanced_closing_bracket():
    text = "Kennzeichen (B-MX 55) bitte"
    # The model tags "B-MX 55)" without the opening bracket.
    assert _trimmed(text, 13, 21) == "B-MX 55"


def test_trim_span_drops_closing_bracket_before_sentence_dot():
    text = "Kombi (BMW F31)."
    assert _trimmed(text, 10, 16) == "F31"


def test_trim_span_drops_unbalanced_opening_bracket():
    text = "Kennzeichen (B-MX 55 bitte"
    assert _trimmed(text, 11, 20) == "B-MX 55"


def test_trim_span_drops_pair_enclosing_whole_span():
    text = "Kennzeichen (B-MX 55) bitte"
    assert _trimmed(text, 11, 21) == "B-MX 55"


def test_trim_span_keeps_brackets_that_belong_to_value():
    text = "Telefon (030) 123456."
    assert _trimmed(text, 7, 21) == "(030) 123456"


def test_trim_span_without_brackets_is_unchanged():
    text = "Kennzeichen M-AB 1234"
    assert _trimmed(text, 11, 21) == "M-AB 1234"


def test_trim_span_handles_non_ascii_text():
    text = "Fahrzeug (KÜN-Ä 9E), danke"
    assert _trimmed(text, 10, 20) == "KÜN-Ä 9E"


def test_group_bio_spans_trims_bracket_from_plate():
    text = "Kombi (B-MX 55)"
    tokens = ["▁Kombi", "▁(", "B", "-", "MX", "▁55", ")"]
    offsets = [(0, 5), (5, 7), (7, 8), (8, 9), (9, 11), (11, 14), (14, 15)]
    labels = [
        "O",
        "O",
        "B-LICENSEPLATENUM",
        "I-LICENSEPLATENUM",
        "I-LICENSEPLATENUM",
        "I-LICENSEPLATENUM",
        "I-LICENSEPLATENUM",
    ]

    spans = group_bio_spans(text, tokens, offsets, labels)

    assert [(text[s:e], label) for s, e, label in spans] == [
        ("B-MX 55", "LICENSEPLATENUM")
    ]
