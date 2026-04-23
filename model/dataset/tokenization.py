"""Tokenization utilities for training samples."""

import re
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
                    import logging

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
        return sorted(
            privacy_mask_with_positions, key=lambda x: x["start"], reverse=True
        )

    def _create_word_labels(
        self, text: str, privacy_mask_with_positions: list[dict[str, Any]]
    ) -> list[str]:
        """Create word-level BIO labels from privacy mask positions.

        Each word gets a BIO-prefixed label: ``B-LABEL`` for the first word
        of an entity span and ``I-LABEL`` for subsequent words.  Words outside
        any entity span receive ``O``.
        """
        # Replace sensitive text with BIO-prefixed label placeholders.
        # privacy_mask_with_positions is sorted by start descending, so
        # replacing from the end preserves earlier offsets.
        text_with_labels = text
        for item in privacy_mask_with_positions:
            label = item["label"]
            start = item["start"]
            end = item["end"]
            value = item["value"]

            word_count = len(value.split())
            bio_labels = [f"B-{label}"] + [f"I-{label}"] * (word_count - 1)
            replacement = " ".join(bio_labels)
            text_with_labels = (
                text_with_labels[:start] + replacement + text_with_labels[end:]
            )

        # Split into words and parse BIO labels
        words = text_with_labels.split()
        word_labels = []
        for word in words:
            match = re.search(r"(\w[\w-]*)", word)
            if match:
                token = match.group(1)
                # Check for BIO-prefixed PII labels (e.g. B-EMAIL, I-FIRSTNAME)
                if token.startswith(("B-", "I-")):
                    word_labels.append(token)
                else:
                    word_labels.append("O")
            else:
                word_labels.append("O")

        return word_labels

    def _is_punctuation_only(self, token_text: str) -> bool:
        """Check if a token contains only punctuation characters."""
        stripped = token_text.strip()
        if not stripped:
            return False
        punctuation_chars = set(",.;:!?)]}['\"-–—()[]{}")
        return all(c in punctuation_chars for c in stripped)

    def _is_punctuation_in_entity(
        self,
        punct_text: str,
        word_idx: int,
        words_original: list[str] | None,
        privacy_mask_with_positions: list[dict[str, Any]] | None,
    ) -> bool:
        """Check if punctuation is part of an entity value (e.g., comma in 'Google, Inc.')."""
        if (
            not words_original
            or not privacy_mask_with_positions
            or word_idx >= len(words_original)
        ):
            return False

        original_word = words_original[word_idx]
        word_without_punct = original_word.rstrip(",.;:!?)]}")

        for item in privacy_mask_with_positions:
            entity_value = item.get("value", "")
            # Punctuation is part of entity if both:
            # 1. Punctuation char is in the entity value
            # 2. The word (without trailing punct) is part of the entity
            if punct_text in entity_value and word_without_punct in entity_value:
                return True
        return False

    def _get_label_id(self, bio_label: str) -> int:
        """Get the label ID for a BIO-prefixed label (e.g. ``B-EMAIL``, ``I-SSN``, ``O``)."""
        if bio_label == "O":
            return 0
        return self.label2id.get(bio_label, 0)

    def _align_labels_with_tokens(
        self,
        word_labels: list[str],
        word_ids: list[int | None],
        token_texts: list[str] | None = None,
        words_original: list[str] | None = None,
        privacy_mask_with_positions: list[dict[str, Any]] | None = None,
    ) -> list[int]:
        """Align word-level BIO labels with token IDs.

        ``word_labels`` already carry the correct BIO prefix (``B-LABEL``,
        ``I-LABEL``, or ``O``).  Sub-word tokens that continue the same
        word receive the ``I-`` variant of their word's label.
        """
        label_ids = []
        prev_word_idx = None

        for idx, word_idx in enumerate(word_ids):
            # Handle special tokens and out-of-bounds
            if word_idx is None or word_idx >= len(word_labels):
                label_ids.append(-100)
                continue

            bio_label = word_labels[word_idx]
            token_text = (
                token_texts[idx] if token_texts and idx < len(token_texts) else ""
            )
            is_punct = self._is_punctuation_only(token_text)

            # Determine effective label for this token
            if is_punct:
                base = (
                    bio_label[2:] if bio_label.startswith(("B-", "I-")) else bio_label
                )
                if base != "O" and not self._is_punctuation_in_entity(
                    token_text.strip(),
                    word_idx,
                    words_original,
                    privacy_mask_with_positions,
                ):
                    bio_label = "O"

            # Sub-word continuation: second+ token of the same word gets I-
            if word_idx == prev_word_idx and bio_label.startswith("B-"):
                bio_label = "I-" + bio_label[2:]

            label_ids.append(self._get_label_id(bio_label))
            prev_word_idx = word_idx

        # Truncate to max_length if needed
        if len(label_ids) > 512:
            label_ids = label_ids[:511] + [-100]

        return label_ids

    def create_pii_sample(
        self, text: str, privacy_mask: list[dict[str, str]]
    ) -> dict[str, Any]:
        """Create a PII training sample with tokenized input and labels."""
        # Find positions for privacy mask items
        privacy_mask_with_positions = self._find_privacy_mask_positions(
            text, privacy_mask
        )

        # Create word-level labels
        word_labels = self._create_word_labels(text, privacy_mask_with_positions)

        # Tokenize the original text
        words_original = text.split()
        tokenized = self.tokenizer(
            words_original,
            truncation=True,
            is_split_into_words=True,
            max_length=512,
            return_offsets_mapping=True,
        )

        # Get word IDs for alignment
        try:
            word_ids = tokenized.word_ids(batch_index=0)
        except (TypeError, AttributeError):
            word_ids = tokenized.word_ids()

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

        # Align labels with tokens
        label_ids = self._align_labels_with_tokens(
            word_labels,
            word_ids,
            token_texts,
            words_original,
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
