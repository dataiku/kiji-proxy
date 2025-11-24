"""Tokenization utilities for training samples."""
import re
from typing import Any

from transformers import AutoTokenizer


class TokenizationProcessor:
    """Processes text tokenization and label alignment."""

    def __init__(self, tokenizer: AutoTokenizer, label2id: dict[str, int], id2label: dict[int, str]):
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
                privacy_mask_with_positions.append({
                    "value": value,
                    "label": label,
                    "start": pos,
                    "end": pos + len(value),
                })
                start = pos + 1

        # Sort by start position (reverse order for replacement)
        return sorted(privacy_mask_with_positions, key=lambda x: x["start"], reverse=True)

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
            text_with_labels = text_with_labels[:start] + replacement + text_with_labels[end:]

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

    def _align_labels_with_tokens(
        self, word_labels: list[str], word_ids: list[int | None]
    ) -> list[int]:
        """Align word-level labels with token IDs."""
        label_ids = []
        previous_word_idx = None
        previous_label = None

        for word_idx in word_ids:
            if word_idx is None:
                # Special tokens (CLS, SEP, etc.) - ignore with -100
                label_ids.append(-100)
                continue

            if word_idx >= len(word_labels):
                # Out of bounds - mark as ignored
                label_ids.append(-100)
                continue

            current_word_label = word_labels[word_idx]

            # Determine if this is beginning or inside
            is_beginning = (previous_word_idx != word_idx) or (previous_label != current_word_label)

            if current_word_label == "O":
                label_ids.append(0)
            else:
                # Use standard mapping - get the full label with prefix
                prefix = "B-" if is_beginning else "I-"
                full_label = f"{prefix}{current_word_label}"
                # Get ID from standard mapping, or use 0 if label not in standard set
                label_ids.append(self.label2id.get(full_label, 0))

            previous_word_idx = word_idx
            previous_label = current_word_label

        # Truncate to max_length if needed
        if len(label_ids) > 512:
            label_ids = label_ids[:511] + [-100]

        return label_ids

    def create_pii_sample(
        self, text: str, privacy_mask: list[dict[str, str]]
    ) -> dict[str, Any]:
        """Create a PII training sample with tokenized input and labels."""
        # Find positions for privacy mask items
        privacy_mask_with_positions = self._find_privacy_mask_positions(text, privacy_mask)

        # Create word-level labels
        word_labels = self._create_word_labels(text, privacy_mask_with_positions)

        # Tokenize the original text
        words_original = text.split()
        tokenized = self.tokenizer(
            words_original,
            truncation=True,
            is_split_into_words=True,
            max_length=512,
            return_offsets_mapping=False,
        )

        # Get word IDs for alignment
        try:
            word_ids = tokenized.word_ids(batch_index=0)
        except (TypeError, AttributeError):
            word_ids = tokenized.word_ids()

        # Align labels with tokens
        label_ids = self._align_labels_with_tokens(word_labels, word_ids)

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
            return_offsets_mapping=False,
        )

        # Get word IDs for token alignment
        try:
            word_ids = tokenized.word_ids(batch_index=0)
        except (TypeError, AttributeError):
            word_ids = tokenized.word_ids()

        # Create a mapping from word index to cluster ID
        word_to_cluster = [-1] * len(words_original)

        # Process each coreference cluster
        for coref in coreferences:
            cluster_id = coref["cluster_id"]
            mentions = coref["mentions"]

            # For each mention in the cluster, find its position in the text
            for mention in mentions:
                start = 0
                while True:
                    pos = text.find(mention, start)
                    if pos == -1:
                        break

                    text_before_mention = text[:pos]
                    words_before = text_before_mention.split()
                    start_word_idx = len(words_before)

                    mention_words = mention.split()
                    end_word_idx = start_word_idx + len(mention_words)

                    # Verify the match by checking if words align correctly
                    if start_word_idx < len(words_original):
                        mention_text_at_pos = " ".join(words_original[start_word_idx:end_word_idx])
                        if (
                            mention.lower() in mention_text_at_pos.lower()
                            or mention_text_at_pos.lower() in mention.lower()
                        ):
                            # Assign cluster ID to all words in this mention
                            for word_idx in range(start_word_idx, min(end_word_idx, len(words_original))):
                                if word_to_cluster[word_idx] == -1:
                                    word_to_cluster[word_idx] = cluster_id

                    start = pos + 1

        # Align cluster IDs with tokens
        cluster_labels = []
        for word_idx in word_ids:
            if word_idx is None:
                cluster_labels.append(-100)
            elif word_idx >= len(word_to_cluster):
                cluster_labels.append(-100)
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

