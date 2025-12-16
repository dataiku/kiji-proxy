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
        """Find start and end positions for each privacy mask item."""
        privacy_mask_with_positions = []
        for item in privacy_mask:
            value = item["value"]
            label = item["label"]

            # Find all occurrences of the value in the text
            start = 0
            while True:
                pos = text.find(value, start)
                if pos == -1:
                    break
                privacy_mask_with_positions.append(
                    {
                        "value": value,
                        "label": label,
                        "start": pos,
                        "end": pos + len(value),
                    }
                )
                start = pos + 1

        # Sort by start position (reverse order for replacement)
        return sorted(
            privacy_mask_with_positions, key=lambda x: x["start"], reverse=True
        )

    def _create_word_labels(
        self, text: str, privacy_mask_with_positions: list[dict[str, Any]]
    ) -> list[str]:
        """Create word-level labels from privacy mask positions."""
        # Replace sensitive text with label placeholders
        text_with_labels = text
        for item in privacy_mask_with_positions:
            label = item["label"]
            start = item["start"]
            end = item["end"]
            value = item["value"]

            # Count words in the sensitive value
            word_count = len(value.split())

            # Replace with appropriate number of label placeholders
            replacement = " ".join([label] * word_count)
            text_with_labels = (
                text_with_labels[:start] + replacement + text_with_labels[end:]
            )

        # Split into words and assign labels
        words = text_with_labels.split()
        word_labels = []
        for word in words:
            match = re.search(r"(\w+)", word)
            if match:
                label = match.group(1)
                # Check if it's a valid PII label (all uppercase, not "O")
                if label.isupper() and label != "O":
                    word_labels.append(label)
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

    def _get_label_id(self, word_label: str, is_beginning: bool) -> int:
        """Get the label ID for a word label with B-/I- prefix."""
        if word_label == "O":
            return 0
        prefix = "B-" if is_beginning else "I-"
        return self.label2id.get(f"{prefix}{word_label}", 0)

    def _align_labels_with_tokens(
        self,
        word_labels: list[str],
        word_ids: list[int | None],
        token_texts: list[str] | None = None,
        words_original: list[str] | None = None,
        privacy_mask_with_positions: list[dict[str, Any]] | None = None,
    ) -> list[int]:
        """Align word-level labels with token IDs."""
        label_ids = []
        prev_word_idx = None
        prev_word_label = None

        for idx, word_idx in enumerate(word_ids):
            # Handle special tokens and out-of-bounds
            if word_idx is None or word_idx >= len(word_labels):
                label_ids.append(-100)
                continue

            word_label = word_labels[word_idx]
            token_text = (
                token_texts[idx] if token_texts and idx < len(token_texts) else ""
            )
            is_punct = self._is_punctuation_only(token_text)

            # Determine effective label for this token
            if is_punct:
                # Punctuation: only label as entity if it's actually part of entity value
                if word_label != "O" and not self._is_punctuation_in_entity(
                    token_text.strip(),
                    word_idx,
                    words_original,
                    privacy_mask_with_positions,
                ):
                    # Punctuation after entity (e.g., comma after "Smith") -> "O"
                    word_label = "O"

            # Determine if this is beginning of entity or inside
            is_beginning = (prev_word_idx != word_idx) or (
                prev_word_label != word_label
            )
            label_ids.append(self._get_label_id(word_label, is_beginning))

            prev_word_idx = word_idx
            prev_word_label = word_label

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

            # Convert token IDs to token strings
            token_texts = []
            for tid in token_ids:
                try:
                    # Convert ID to raw token string (before decoding)
                    # This preserves punctuation marks better
                    token_str = self.tokenizer.convert_ids_to_tokens([tid])[0]
                    # For punctuation detection, use the raw token string
                    # Remove special prefixes like ## for subword tokens, but keep punctuation
                    if token_str.startswith("##"):
                        token_text = token_str[2:]
                    else:
                        token_text = token_str
                    # Also try decoded version as fallback for better accuracy
                    decoded_text = self.tokenizer.convert_tokens_to_string([token_str])
                    # Use decoded text if it's more reliable (non-empty and matches token)
                    if decoded_text and len(decoded_text.strip()) > 0:
                        token_texts.append(decoded_text)
                    else:
                        token_texts.append(token_text)
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

    def create_coreference_sample(
        self, text: str, coreferences: list[dict[str, Any]]
    ) -> dict[str, Any]:
        """Create a coreference detection training sample."""
        # Tokenize the text
        words_original = text.split()
        tokenized = self.tokenizer(
            words_original,
            truncation=True,
            is_split_into_words=True,
            max_length=512,
            return_offsets_mapping=True,
        )

        # Get word IDs for token alignment
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

            token_texts = []
            for tid in token_ids:
                try:
                    # Convert ID to raw token string (before decoding)
                    token_str = self.tokenizer.convert_ids_to_tokens([tid])[0]
                    # For punctuation detection, use the raw token string
                    if token_str.startswith("##"):
                        token_text = token_str[2:]
                    else:
                        token_text = token_str
                    # Also try decoded version as fallback for better accuracy
                    decoded_text = self.tokenizer.convert_tokens_to_string([token_str])
                    # Use decoded text if it's more reliable (non-empty and matches token)
                    if decoded_text and len(decoded_text.strip()) > 0:
                        token_texts.append(decoded_text)
                    else:
                        token_texts.append(token_text)
                except (IndexError, TypeError, AttributeError):
                    token_texts.append("")
        except (TypeError, KeyError, IndexError, AttributeError):
            token_texts = None

        # Create a mapping from word index to cluster ID
        word_to_cluster = [-1] * len(words_original)

        # Process each coreference cluster
        for coref in coreferences:
            cluster_id = coref["cluster_id"]
            mentions = coref["mentions"]

            # For each mention in the cluster, find its position in the text
            for mention in mentions:
                # Strip punctuation from mention for matching
                mention_clean = mention.strip().rstrip(",.;:!?)]}")
                start = 0
                while True:
                    pos = text.find(mention_clean, start)
                    if pos == -1:
                        break

                    text_before_mention = text[:pos]
                    words_before = text_before_mention.split()
                    start_word_idx = len(words_before)

                    mention_words = mention_clean.split()
                    end_word_idx = start_word_idx + len(mention_words)

                    # Verify the match by checking if words align correctly
                    if start_word_idx < len(words_original):
                        mention_text_at_pos = " ".join(
                            words_original[start_word_idx:end_word_idx]
                        )
                        if (
                            mention_clean.lower() in mention_text_at_pos.lower()
                            or mention_text_at_pos.lower() in mention_clean.lower()
                        ):
                            # Assign cluster ID to all words in this mention (skip punctuation-only words)
                            for word_idx in range(
                                start_word_idx, min(end_word_idx, len(words_original))
                            ):
                                # Check if this word is punctuation-only
                                word_text = words_original[word_idx]
                                is_punctuation_only = word_text.strip() and all(
                                    c in ",.;:!?)]} " for c in word_text.strip()
                                )
                                if (
                                    not is_punctuation_only
                                    and word_to_cluster[word_idx] == -1
                                ):
                                    word_to_cluster[word_idx] = cluster_id

                    start = pos + 1

        # Align cluster IDs with tokens
        cluster_labels = []
        for idx, word_idx in enumerate(word_ids):
            if word_idx is None:
                cluster_labels.append(-100)
            elif word_idx >= len(word_to_cluster):
                cluster_labels.append(-100)
            else:
                # Check if this token is punctuation-only
                is_punctuation_only = False
                if token_texts is not None and idx < len(token_texts):
                    token_text = token_texts[idx]
                    stripped = token_text.strip()
                    is_punctuation_only = stripped and all(
                        c in ",.;:!?)]} " for c in stripped
                    )

                # If punctuation-only, always label as NO_COREF (0)
                if is_punctuation_only:
                    cluster_labels.append(0)
                else:
                    cluster_id = word_to_cluster[word_idx]
                    if cluster_id == -1:
                        cluster_labels.append(0)  # No coreference
                    else:
                        cluster_labels.append(cluster_id + 1)  # Add 1 to avoid 0

        # Truncate to max_length if needed
        if len(cluster_labels) > 512:
            cluster_labels = cluster_labels[:511] + [-100]

        # Create cluster_id to label mapping
        cluster_id2label = {0: "NO_COREF"}
        for coref in coreferences:
            cluster_id = coref["cluster_id"]
            entity_type = coref.get("entity_type", "unknown")
            cluster_id2label[cluster_id + 1] = f"CLUSTER_{cluster_id}_{entity_type}"
        cluster_id2label[-100] = "IGNORE"

        return {
            "input_ids": tokenized["input_ids"],
            "attention_mask": tokenized["attention_mask"],
            "coreference_labels": cluster_labels,
            "text": text,
            "cluster_id2label": cluster_id2label,
        }
