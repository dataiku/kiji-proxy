"""Tokenization utilities for training samples.

This module handles tokenization and label alignment for PII detection training.
It uses character-offset-based alignment to ensure consistency between training
and inference tokenization.
"""

from typing import Any

from transformers import PreTrainedTokenizerBase


class TokenizationProcessor:
    """Processes text tokenization and label alignment using character offsets."""

    def __init__(
        self,
        tokenizer: PreTrainedTokenizerBase,
        label2id: dict[str, int],
        id2label: dict[int, str],
        max_length: int = 4096,
    ):
        self.tokenizer = tokenizer
        self.label2id = label2id
        self.id2label = id2label
        self.max_length = max_length

    def _find_all_occurrences(
        self, text: str, privacy_mask: list[dict[str, str]]
    ) -> list[dict[str, Any]]:
        """
        Find all character-level positions for each privacy mask item.

        Returns list of {value, label, start, end} sorted by start position.
        """
        occurrences = []
        for item in privacy_mask:
            value = item["value"]
            label = item["label"]

            # Find all occurrences of the value in the text
            start_pos = 0
            while True:
                pos = text.find(value, start_pos)
                if pos == -1:
                    break
                occurrences.append(
                    {
                        "value": value,
                        "label": label,
                        "start": pos,
                        "end": pos + len(value),
                    }
                )
                start_pos = pos + 1

        # Sort by start position
        return sorted(occurrences, key=lambda x: x["start"])

    def _get_label_for_position(
        self,
        char_start: int,
        char_end: int,
        entities: list[dict[str, Any]],
        prev_label: str | None,
        prev_entity_idx: int | None,
    ) -> tuple[str, int | None]:
        """
        Determine the label for a token based on its character position.

        Returns (label, entity_index) where:
        - label is "O", "B-LABEL", or "I-LABEL"
        - entity_index is the index of the entity this token belongs to (or None)
        """
        # Find which entity (if any) this token overlaps with
        for idx, entity in enumerate(entities):
            entity_start = entity["start"]
            entity_end = entity["end"]
            entity_label = entity["label"]

            # Check if token overlaps with entity
            # Token overlaps if: token_start < entity_end AND token_end > entity_start
            if char_start < entity_end and char_end > entity_start:
                # Determine if this is the beginning or inside of the entity
                # It's B- if:
                # 1. Previous token was not part of this same entity, OR
                # 2. Previous token was a different entity type
                if prev_entity_idx != idx:
                    return f"B-{entity_label}", idx
                else:
                    return f"I-{entity_label}", idx

        return "O", None

    def _align_labels_to_tokens(
        self,
        text: str,
        offsets: list[tuple[int, int]],
        privacy_mask: list[dict[str, str]],
    ) -> list[int]:
        """
        Align PII labels to tokens using character offsets.

        Args:
            text: Original text
            offsets: List of (start, end) character offsets for each token
            privacy_mask: List of {value, label} items

        Returns:
            List of label IDs for each token
        """
        # Find all entity occurrences with positions
        entities = self._find_all_occurrences(text, privacy_mask)

        label_ids = []
        prev_entity_idx = None
        prev_label = "O"

        for start, end in offsets:
            # Special tokens have offset (0, 0) - mark as ignore
            if start == 0 and end == 0:
                label_ids.append(-100)
                prev_entity_idx = None
                prev_label = "O"
                continue

            # Get label for this token position
            label, entity_idx = self._get_label_for_position(
                start, end, entities, prev_label, prev_entity_idx
            )

            # Convert label to ID
            label_id = self.label2id.get(label, 0)
            label_ids.append(label_id)

            prev_entity_idx = entity_idx
            prev_label = label

        return label_ids

    def _align_coref_labels_to_tokens(
        self,
        text: str,
        offsets: list[tuple[int, int]],
        coreferences: list[dict[str, Any]],
    ) -> list[int]:
        """
        Align coreference labels to tokens using character offsets.

        Args:
            text: Original text
            offsets: List of (start, end) character offsets for each token
            coreferences: List of coreference clusters with mentions

        Returns:
            List of coreference label IDs for each token
        """
        # Build a mapping of character positions to cluster IDs
        # Each position can belong to at most one cluster
        char_to_cluster = {}

        for coref in coreferences:
            cluster_id = coref["cluster_id"]
            mentions = coref["mentions"]

            for mention in mentions:
                # Find all occurrences of this mention in the text
                mention_clean = mention.strip()
                start_pos = 0
                while True:
                    pos = text.find(mention_clean, start_pos)
                    if pos == -1:
                        break

                    # Mark all characters in this mention as belonging to this cluster
                    for char_idx in range(pos, pos + len(mention_clean)):
                        if char_idx not in char_to_cluster:
                            char_to_cluster[char_idx] = cluster_id

                    start_pos = pos + 1

        # Align to tokens
        coref_labels = []

        for start, end in offsets:
            # Special tokens have offset (0, 0)
            if start == 0 and end == 0:
                coref_labels.append(-100)
                continue

            # Check if any character in this token belongs to a cluster
            cluster_id = None
            for char_idx in range(start, end):
                if char_idx in char_to_cluster:
                    cluster_id = char_to_cluster[char_idx]
                    break

            if cluster_id is not None:
                coref_labels.append(cluster_id + 1)  # +1 because 0 = NO_COREF
            else:
                coref_labels.append(0)  # NO_COREF

        return coref_labels

    def create_pii_sample(
        self, text: str, privacy_mask: list[dict[str, str]]
    ) -> dict[str, Any]:
        """
        Create a PII training sample with tokenized input and labels.

        Uses raw text tokenization (NOT is_split_into_words) to match inference.
        """
        # Tokenize the raw text - this matches inference tokenization
        tokenized = self.tokenizer(
            text,
            truncation=True,
            max_length=self.max_length,
            return_offsets_mapping=True,
        )

        # Get offsets for label alignment
        offsets = tokenized["offset_mapping"]

        # Align labels to tokens using character offsets
        label_ids = self._align_labels_to_tokens(text, offsets, privacy_mask)

        # Truncate labels if needed
        if len(label_ids) > self.max_length:
            label_ids = label_ids[: self.max_length]

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
        """
        Create a coreference detection training sample.

        Uses raw text tokenization (NOT is_split_into_words) to match inference.
        """
        # Tokenize the raw text - this matches inference tokenization
        tokenized = self.tokenizer(
            text,
            truncation=True,
            max_length=self.max_length,
            return_offsets_mapping=True,
        )

        # Get offsets for label alignment
        offsets = tokenized["offset_mapping"]

        # Align coreference labels to tokens
        coref_labels = self._align_coref_labels_to_tokens(text, offsets, coreferences)

        # Truncate labels if needed
        if len(coref_labels) > self.max_length:
            coref_labels = coref_labels[: self.max_length]

        # Create cluster_id to label mapping
        cluster_id2label = {0: "NO_COREF", -100: "IGNORE"}
        for coref in coreferences:
            cluster_id = coref["cluster_id"]
            entity_type = coref.get("entity_type", "unknown")
            cluster_id2label[cluster_id + 1] = f"CLUSTER_{cluster_id}_{entity_type}"

        return {
            "input_ids": tokenized["input_ids"],
            "attention_mask": tokenized["attention_mask"],
            "coreference_labels": coref_labels,
            "text": text,
            "cluster_id2label": cluster_id2label,
        }
