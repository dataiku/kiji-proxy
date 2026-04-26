"""Shared BIO span decoding helpers for PII inference."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

Span = tuple[int, int, str]


def split_bio_label(label: str) -> tuple[str, bool, bool]:
    """Split a BIO label into base label and B/I flags."""
    is_beginning = label.startswith("B-")
    is_inside = label.startswith("I-")
    if is_beginning or is_inside:
        return label[2:], is_beginning, is_inside
    return label, False, False


def token_text_at(text: str, start: int, end: int) -> str:
    """Return token text from an offset pair, with bounds checks."""
    if start < 0 or end > len(text) or end <= start:
        return ""
    return text[start:end]


def is_entity_joiner_token(token_text: str) -> bool:
    """Return true for punctuation tokens that can join compact entities."""
    trimmed = token_text.strip()
    return bool(trimmed) and all(c in ".,@_-+:/#%&=" for c in trimmed)


def trim_span(text: str, start: int, end: int) -> tuple[int, int]:
    """Trim whitespace and trailing sentence punctuation from a span."""
    while start < end and text[start] in " \t\n\r":
        start += 1
    while end > start and text[end - 1] in " \t\n\r":
        end -= 1
    while end > start and text[end - 1] in ",.;:!?":
        if end < len(text) and text[end] not in " \t\n\r":
            break
        end -= 1
    return start, end


def _as_int(value: Any) -> int:
    """Convert tensor/scalar-like offset values to plain ints."""
    if hasattr(value, "item"):
        return int(value.item())
    return int(value)


def _offset_pair(offset: Any) -> tuple[int, int]:
    """Convert an offset object to a plain ``(start, end)`` pair."""
    return _as_int(offset[0]), _as_int(offset[1])


def _token_starts_at_previous_end(
    token_index: int,
    current_tokens: list[int],
    offsets: Sequence[Any],
) -> bool:
    """Check whether the current token is contiguous with the active entity."""
    if not current_tokens or token_index >= len(offsets):
        return False
    previous_token_index = current_tokens[-1]
    if previous_token_index >= len(offsets):
        return False
    _, previous_end = _offset_pair(offsets[previous_token_index])
    current_start, _ = _offset_pair(offsets[token_index])
    return previous_end == current_start


def _bridge_joiner_token(
    index: int,
    text: str,
    offsets: Sequence[Any],
    label: str,
    confidence: float,
    labels: Sequence[str],
    confidences: Sequence[float],
    current_label: str | None,
    confidence_threshold: float,
) -> tuple[str, float]:
    """Bridge punctuation inside compact entities such as emails and phone numbers."""
    start, end = _offset_pair(offsets[index])
    token_text = token_text_at(text, start, end)
    if label != "O" or current_label is None or not is_entity_joiner_token(token_text):
        return label, confidence

    next_index = index + 1
    if next_index >= len(labels):
        return label, confidence

    next_label = labels[next_index]
    next_confidence = confidences[next_index]
    next_base_label, next_is_beginning, next_is_inside = split_bio_label(next_label)
    if (
        next_confidence >= confidence_threshold
        and (next_is_beginning or next_is_inside)
        and next_base_label == current_label
    ):
        return f"I-{current_label}", confidence

    return label, confidence


def group_bio_spans(
    text: str,
    tokens: Sequence[str],
    offsets: Sequence[Any],
    labels: Sequence[str],
    *,
    confidences: Sequence[float] | None = None,
    confidence_threshold: float = 0.0,
    special_tokens: set[str] | None = None,
) -> list[Span]:
    """Group token-level BIO predictions into character spans."""
    if len(tokens) != len(offsets) or len(tokens) != len(labels):
        raise ValueError("tokens, offsets, and labels must have the same length")

    if confidences is None:
        confidences = [1.0] * len(tokens)
    if len(confidences) != len(tokens):
        raise ValueError("confidences must have the same length as tokens")

    special_tokens = special_tokens or set()
    spans: list[Span] = []
    current_label: str | None = None
    current_start = 0
    current_end = 0
    current_tokens: list[int] = []

    def finish_current() -> None:
        nonlocal current_label, current_start, current_end, current_tokens
        if current_label is None:
            return
        start, end = trim_span(text, current_start, current_end)
        if start < end:
            spans.append((start, end, current_label))
        current_label = None
        current_tokens = []

    for index, (token, raw_label) in enumerate(zip(tokens, labels, strict=True)):
        start, end = _offset_pair(offsets[index])
        if token in special_tokens or end <= start:
            continue

        confidence = float(confidences[index])
        label = raw_label if confidence >= confidence_threshold else "O"
        label, confidence = _bridge_joiner_token(
            index,
            text,
            offsets,
            label,
            confidence,
            labels,
            confidences,
            current_label,
            confidence_threshold,
        )

        base_label, is_beginning, is_inside = split_bio_label(label)
        is_same_compact_entity = (
            label != "O"
            and current_label is not None
            and current_label == base_label
            and _token_starts_at_previous_end(index, current_tokens, offsets)
        )

        if (
            label != "O"
            and current_label is not None
            and current_label == base_label
            and (is_inside or is_same_compact_entity)
        ):
            current_end = end
            current_tokens.append(index)
        elif label != "O" and (is_beginning or current_label is None):
            finish_current()
            current_label = base_label
            current_start = start
            current_end = end
            current_tokens = [index]
        else:
            finish_current()

    finish_current()
    return spans
