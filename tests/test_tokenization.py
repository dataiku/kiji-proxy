from __future__ import annotations

from model.dataset.label_utils import LabelUtils
from model.dataset.tokenization import TokenizationProcessor


class FakeTokenizer:
    def __init__(self, tokens: list[str], offsets: list[tuple[int, int]]):
        self.tokens = tokens
        self.offsets = offsets
        self.token_to_id = {token: idx for idx, token in enumerate(tokens)}

    def __call__(self, _text: str, **_kwargs):
        return {
            "input_ids": list(range(len(self.tokens))),
            "attention_mask": [1] * len(self.tokens),
            "offset_mapping": self.offsets,
        }

    def convert_ids_to_tokens(self, token_ids: list[int]) -> list[str]:
        return [self.tokens[token_id] for token_id in token_ids]

    def convert_tokens_to_string(self, tokens: list[str]) -> str:
        return "".join(tokens).lstrip("▁")


def _processor(tokens: list[str], offsets: list[tuple[int, int]]):
    label2id, id2label = LabelUtils.create_standard_label2id()
    return TokenizationProcessor(FakeTokenizer(tokens, offsets), label2id, id2label)


def _label_names(sample: dict) -> list[str]:
    _label2id, id2label = LabelUtils.create_standard_label2id()
    return [id2label.get(label_id, "O") for label_id in sample["labels"]]


def test_embedded_email_in_markup_is_labeled():
    text = "<Email>jankulovska18@aol.com</Email>"
    value = "jankulovska18@aol.com"
    tokens = [
        "[CLS]",
        "▁<",
        "Email",
        ">",
        "jan",
        "kul",
        "ovska",
        "18",
        "@",
        "aol",
        ".",
        "com",
        "<",
        "/",
        "Email",
        ">",
        "[SEP]",
    ]
    offsets = [
        (0, 0),
        (0, 1),
        (1, 6),
        (6, 7),
        (7, 10),
        (10, 13),
        (13, 18),
        (18, 20),
        (20, 21),
        (21, 24),
        (24, 25),
        (25, 28),
        (28, 29),
        (29, 30),
        (30, 35),
        (35, 36),
        (0, 0),
    ]
    start = text.index(value)
    sample = _processor(tokens, offsets).create_pii_sample(
        text,
        [{"value": value, "label": "EMAIL", "start": start, "end": start + len(value)}],
    )

    assert _label_names(sample) == [
        "IGNORE",
        "O",
        "O",
        "O",
        "B-EMAIL",
        "I-EMAIL",
        "I-EMAIL",
        "I-EMAIL",
        "I-EMAIL",
        "I-EMAIL",
        "I-EMAIL",
        "I-EMAIL",
        "O",
        "O",
        "O",
        "O",
        "IGNORE",
    ]


def test_trailing_email_period_is_not_labeled():
    text = "Email jonathan.reyes@example.com."
    value = "jonathan.reyes@example.com"
    tokens = [
        "[CLS]",
        "▁Email",
        "▁jonathan",
        ".",
        "re",
        "yes",
        "@",
        "example",
        ".",
        "com",
        ".",
        "[SEP]",
    ]
    offsets = [
        (0, 0),
        (0, 5),
        (6, 14),
        (14, 15),
        (15, 17),
        (17, 20),
        (20, 21),
        (21, 28),
        (28, 29),
        (29, 32),
        (32, 33),
        (0, 0),
    ]
    start = text.index(value)
    sample = _processor(tokens, offsets).create_pii_sample(
        text,
        [{"value": value, "label": "EMAIL", "start": start, "end": start + len(value)}],
    )

    assert _label_names(sample) == [
        "IGNORE",
        "O",
        "B-EMAIL",
        "I-EMAIL",
        "I-EMAIL",
        "I-EMAIL",
        "I-EMAIL",
        "I-EMAIL",
        "I-EMAIL",
        "I-EMAIL",
        "O",
        "IGNORE",
    ]
