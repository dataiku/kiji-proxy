"""Tokenization utilities for training samples."""

import logging
from typing import Any

from transformers import AutoTokenizer


class TokenizationProcessor:
    """Processes text tokenization and label alignment."""

    def __init__(
        self,
        tokenizer: AutoTokenizer,
        label2id: dict[str, int],
        id2label: dict[int, str],
    ):
        self.tokenizer = tokenizer
        self.label2id = label2id
        self.id2label = id2label

    def _drop_overlapping_positions(
        self, positions: list[dict[str, Any]]
    ) -> list[dict[str, Any]]:
        """Drop overlapping spans before string replacement.

        A single token can only have one BIO label. If the source annotations contain
        nested or overlapping spans, keep the longest span and drop the shorter one
        so label replacement cannot corrupt neighboring text.
        """
        kept: list[dict[str, Any]] = []
        dropped = 0

        for item in sorted(
            positions,
            key=lambda x: (-(x["end"] - x["start"]), x["start"], x["label"]),
        ):
            overlaps_kept = any(
                item["start"] < kept_item["end"] and kept_item["start"] < item["end"]
                for kept_item in kept
            )
            if overlaps_kept:
                dropped += 1
                continue
            kept.append(item)

        if dropped:
            logging.getLogger(__name__).debug(
                "Dropped %d overlapping privacy-mask span(s)", dropped
            )

        return sorted(kept, key=lambda x: x["start"], reverse=True)

    def _find_privacy_mask_positions(
        self, text: str, privacy_mask: list[dict[str, str]]
    ) -> list[dict[str, Any]]:
        """Find start and end positions for each privacy mask item.

        Uses character offsets from annotations when available (preferred).
        Falls back to word-boundary-aware regex search for data sources
        that don't provide offsets.
        """
        privacy_mask_with_positions = []
        for item in privacy_mask:
            if "start" in item and "end" in item:
                # Use annotation offsets directly — no search needed
                entry = {
                    "value": item["value"],
                    "label": item["label"],
                    "start": item["start"],
                    "end": item["end"],
                }
                # Validate that the offset matches the expected value
                actual = text[entry["start"] : entry["end"]]
                if actual != entry["value"]:
                    logging.getLogger(__name__).debug(
                        "Offset mismatch: expected '%s' but found '%s' at [%d:%d]",
                        entry["value"],
                        actual,
                        entry["start"],
                        entry["end"],
                    )
                else:
                    privacy_mask_with_positions.append(entry)
            else:
                raise ValueError(
                    f"Privacy mask item missing 'start'/'end' offsets: {item}"
                )

        # Sort by start position (reverse order for replacement)
        return self._drop_overlapping_positions(privacy_mask_with_positions)

    def _is_punctuation_only(self, token_text: str) -> bool:
        """Check if a token contains only punctuation characters."""
        stripped = token_text.strip()
        if not stripped:
            return False
        punctuation_chars = set(",.;:!?)]}['\"-–—()[]{}")
        return all(c in punctuation_chars for c in stripped)

    def _token_overlaps_entity(
        self,
        token_start: int,
        token_end: int,
        entity_label: str,
        privacy_mask_with_positions: list[dict[str, Any]] | None,
    ) -> bool:
        """Check whether a token span overlaps an entity span with the same label."""
        if (
            token_start < 0
            or token_end <= token_start
            or not privacy_mask_with_positions
        ):
            return False

        for item in privacy_mask_with_positions:
            if item.get("label") != entity_label:
                continue
            entity_start = item.get("start", 0)
            entity_end = item.get("end", 0)
            if token_start < entity_end and entity_start < token_end:
                return True
        return False

    def _best_entity_for_token(
        self,
        token_start: int,
        token_end: int,
        privacy_mask_with_positions: list[dict[str, Any]],
    ) -> dict[str, Any] | None:
        """Return the entity span with the largest overlap for a token."""
        best_entity = None
        best_overlap = 0

        for item in privacy_mask_with_positions:
            entity_start = item.get("start", 0)
            entity_end = item.get("end", 0)
            overlap = max(
                0,
                min(token_end, entity_end) - max(token_start, entity_start),
            )
            if overlap > best_overlap:
                best_overlap = overlap
                best_entity = item

        return best_entity

    def _get_label_id(self, bio_label: str) -> int:
        """Get the label ID for a BIO-prefixed label (e.g. ``B-EMAIL``, ``I-SSN``, ``O``)."""
        if bio_label == "O":
            return 0
        return self.label2id.get(bio_label, 0)

    def _align_labels_with_offsets(
        self,
        token_offsets: list[tuple[int, int]] | None,
        token_texts: list[str] | None,
        privacy_mask_with_positions: list[dict[str, Any]],
    ) -> list[int]:
        """Align BIO labels directly from character spans to token offsets."""
        if token_offsets is None:
            return []

        sorted_positions = sorted(
            privacy_mask_with_positions,
            key=lambda item: (item["start"], item["end"], item["label"]),
        )
        label_ids = []
        prev_entity_key = None

        for idx, (token_start, token_end) in enumerate(token_offsets):
            if token_end <= token_start:
                label_ids.append(-100)
                prev_entity_key = None
                continue

            token_text = (
                token_texts[idx] if token_texts and idx < len(token_texts) else ""
            )
            entity = self._best_entity_for_token(
                token_start,
                token_end,
                sorted_positions,
            )
            if entity is None:
                label_ids.append(0)
                prev_entity_key = None
                continue

            label = entity["label"]
            entity_key = (entity["start"], entity["end"], label)
            prefix = "I" if entity_key == prev_entity_key else "B"

            # Keep punctuation that is inside the entity span (email dots, phone
            # separators), but leave standalone punctuation outside spans as O.
            if self._is_punctuation_only(
                token_text
            ) and not self._token_overlaps_entity(
                token_start,
                token_end,
                label,
                sorted_positions,
            ):
                label_ids.append(0)
                prev_entity_key = None
                continue

            label_ids.append(self._get_label_id(f"{prefix}-{label}"))
            prev_entity_key = entity_key

        return label_ids

    def create_pii_sample(
        self, text: str, privacy_mask: list[dict[str, str]]
    ) -> dict[str, Any]:
        """Create a PII training sample with tokenized input and labels."""
        # Find positions for privacy mask items
        privacy_mask_with_positions = self._find_privacy_mask_positions(
            text, privacy_mask
        )

        tokenized = self.tokenizer(
            text,
            truncation=True,
            max_length=512,
            return_offsets_mapping=True,
        )
        token_offsets = tokenized.get("offset_mapping")

        # Get token texts to check for punctuation-only tokens
        # Use raw token strings for better punctuation detection
        token_texts = None
        try:
            # Handle both 1D and 2D input_ids (depends on tokenizer behavior)
            input_ids = tokenized["input_ids"]
            if isinstance(input_ids, list) and len(input_ids) > 0:
                # Check if it's 2D (list of lists) or 1D (list of ints)
                if isinstance(input_ids[0], list):
                    token_ids = input_ids[0]
                else:
                    token_ids = input_ids
            else:
                token_ids = list(input_ids)

            # Convert token IDs to clean token strings for punctuation detection.
            # The tokenizer.decode / convert_tokens_to_string path handles
            # subword prefixes for any tokenizer family:
            #   - WordPiece  (DistilBERT):  "##ing"  → "ing"
            #   - SentencePiece (DeBERTa):  "▁hello" → "hello"
            token_texts = []
            for tid in token_ids:
                try:
                    token_str = self.tokenizer.convert_ids_to_tokens([tid])[0]
                    decoded_text = self.tokenizer.convert_tokens_to_string([token_str])
                    if decoded_text and decoded_text.strip():
                        token_texts.append(decoded_text)
                    else:
                        # Fallback: strip known subword prefixes manually
                        cleaned = token_str
                        if cleaned.startswith("##"):
                            cleaned = cleaned[2:]
                        elif cleaned.startswith("\u2581"):
                            cleaned = cleaned[1:]
                        token_texts.append(cleaned)
                except (IndexError, TypeError, AttributeError):
                    token_texts.append("")
        except (TypeError, KeyError, IndexError, AttributeError):
            token_texts = None

        # Align labels directly from annotation character spans. This handles
        # compact formats such as XML/JSON/email addresses where entities are
        # embedded inside a whitespace-delimited "word".
        label_ids = self._align_labels_with_offsets(
            token_offsets,
            token_texts,
            privacy_mask_with_positions,
        )

        return {
            "input_ids": tokenized["input_ids"],
            "attention_mask": tokenized["attention_mask"],
            "labels": label_ids,
            "text": text,
            "label2id": self.label2id,
            "id2label": self.id2label,
        }
