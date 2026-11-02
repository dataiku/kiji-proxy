from __future__ import annotations

from model.src.span_decoder import group_bio_spans


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
