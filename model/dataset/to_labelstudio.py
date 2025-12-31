"""
Convert the reviewed samples to Label Studio format.
"""

import json
import re
from pathlib import Path
from typing import Any

from absl import app, flags
from absl.flags import DuplicateFlagError

# Export the main conversion functions
__all__ = [
    "convert_to_labelstudio",
    "convert_all_samples_to_labelstudio",
    "convert_sample_to_labelstudio",
]

FLAGS = flags.FLAGS

# Guard flag definitions to prevent duplicates when module is imported
try:
    flags.DEFINE_string(
        "samples_dir",
        "model/dataset/reviewed_samples",
        "Directory containing reviewed sample JSON files",
    )

    flags.DEFINE_string(
        "output_dir",
        "model/dataset/annotation_samples",
        "Directory to output Label Studio JSON files",
    )
except DuplicateFlagError:
    # Flags already defined (module imported multiple times)
    pass


def find_all_occurrences(text: str, value: str, use_word_boundaries: bool = False) -> list[int]:
    """
    Find all start positions of value in text.

    Args:
        text: The text to search in
        value: The value to search for
        use_word_boundaries: If True, only match complete words (prevents "her" matching "here")

    Returns:
        List of start positions where value appears
    """
    if use_word_boundaries:
        # Use regex to find word boundaries - escape special regex characters in value
        escaped_value = re.escape(value)
        pattern = r'\b' + escaped_value + r'\b'
        positions = []
        for match in re.finditer(pattern, text):
            positions.append(match.start())
        return positions
    else:
        # Original simple substring matching (for privacy_mask values that might be part of words)
        positions = []
        start = 0
        while True:
            pos = text.find(value, start)
            if pos == -1:
                break
            positions.append(pos)
            start = pos + 1
        return positions


def convert_sample_to_labelstudio(sample: dict[str, Any]) -> dict[str, Any]:
    """
    Convert a single reviewed sample to Label Studio prediction format.

    Args:
        sample: Dictionary with 'text', 'privacy_mask', and 'coreferences' keys

    Returns:
        Label Studio format dictionary with 'data' and 'predictions'
    """
    text = sample["text"]
    privacy_mask = sample["privacy_mask"]
    coreferences = sample.get("coreferences", [])

    # Track which positions we've already used to avoid duplicates
    used_positions = set()

    # Create entity annotations
    entities = []
    entity_id_counter = 1

    # Map from (value, position) to entity_id for coreference resolution
    value_to_entity_id = {}

    # First, add all privacy mask entities
    for mask_item in privacy_mask:
        value = mask_item["value"]
        label = mask_item["label"]

        # Find all occurrences of this value in the text
        positions = find_all_occurrences(text, value)

        for start_pos in positions:
            # Skip if we've already used this exact position
            if start_pos in used_positions:
                continue

            end_pos = start_pos + len(value)
            entity_id = f"ent-{entity_id_counter}"

            entities.append(
                {
                    "id": entity_id,
                    "from_name": "entities",
                    "to_name": "text",
                    "type": "labels",
                    "value": {
                        "start": start_pos,
                        "end": end_pos,
                        "text": value,
                        "labels": [label],
                    },
                }
            )

            # Store mapping for coreference resolution
            value_to_entity_id[(value, start_pos)] = entity_id
            used_positions.add(start_pos)
            entity_id_counter += 1

            # Only use the first occurrence of each value to avoid duplicates
            break

    # Now add entities for coreference mentions that aren't in privacy_mask
    # Also track which mentions map to which entity IDs for relation creation
    mention_to_entity_ids = {}  # Maps mention text to list of (entity_id, position) tuples

    for coref_cluster in coreferences:
        mentions_raw = coref_cluster["mentions"]
        entity_type = coref_cluster.get("entity_type", "mention")

        # Normalize mentions: handle both old format (list of strings) and new format (list of objects)
        mentions = []
        for mention_item in mentions_raw:
            if isinstance(mention_item, str):
                # Old format: just a string
                mentions.append({
                    "text": mention_item,
                    "type": "pronoun" if mention_item.lower() in ["i", "me", "my", "he", "she", "him", "her", "they", "them", "their", "it", "its", "we", "us", "our", "you", "your"] else "reference"
                })
            else:
                # New format: object with text, type, and optionally privacy_mask_labels
                mentions.append(mention_item)

        for mention_obj in mentions:
            mention = mention_obj["text"]
            mention_type = mention_obj.get("type", "reference")
            privacy_mask_labels = mention_obj.get("privacy_mask_labels", [])
            # Find all occurrences of this mention in the text using word boundaries
            # This prevents false matches (e.g., "her" matching "here")
            # First try exact match with word boundaries
            positions = find_all_occurrences(text, mention, use_word_boundaries=True)

            # If no exact match, try case-insensitive search with word boundaries
            if not positions:
                mention_lower = mention.lower()
                # Use regex for case-insensitive word boundary matching
                escaped_mention = re.escape(mention_lower)
                pattern = r'\b' + escaped_mention + r'\b'
                positions = []
                for match in re.finditer(pattern, text, re.IGNORECASE):
                    positions.append(match.start())

            # Track entity IDs for this mention (may be multiple if split)
            mention_entity_ids_for_mention = []

            # If privacy_mask_labels are provided, use them to find the corresponding entities
            # This helps when names are split (e.g., "Aroha Patel" → ["FIRSTNAME", "SURNAME"])
            if privacy_mask_labels and positions:
                for pos in positions:
                    key = (mention, pos)
                    if key not in value_to_entity_id:
                        # Find the first entity in privacy_mask that matches one of the labels at this position
                        matching_entity_id = None
                        for mask_item in privacy_mask:
                            if mask_item["label"] in privacy_mask_labels:
                                # Check if this value appears at or near the mention position
                                mask_value = mask_item["value"]
                                mask_positions = find_all_occurrences(text, mask_value)
                                for mask_pos in mask_positions:
                                    # Allow some flexibility for split names (e.g., "Aroha" at pos 4, "Patel" at pos 10)
                                    if abs(mask_pos - pos) < 20:
                                        mask_key = (mask_value, mask_pos)
                                        if mask_key in value_to_entity_id:
                                            matching_entity_id = value_to_entity_id[mask_key]
                                            break
                                if matching_entity_id:
                                    break

                        if matching_entity_id:
                            # Create a new entity for the full mention to link everything together
                            end_pos = pos + len(mention)
                            entity_id = f"ent-{entity_id_counter}"

                            entities.append(
                                {
                                    "id": entity_id,
                                    "from_name": "entities",
                                    "to_name": "text",
                                    "type": "labels",
                                    "value": {
                                        "start": pos,
                                        "end": end_pos,
                                        "text": mention,
                                        "labels": [],  # No label - just for coreference relationships
                                    },
                                }
                            )

                            value_to_entity_id[key] = entity_id
                            mention_entity_ids_for_mention.append((entity_id, pos))
                            entity_id_counter += 1
                            continue

            for pos in positions:
                # Check if this mention is already in value_to_entity_id
                key = (mention, pos)
                if key not in value_to_entity_id:
                    # Check if this is a multi-word mention that might be split into separate entities
                    # For example, "Aroha Patel" might be split into "Aroha" (FIRSTNAME) and "Patel" (SURNAME)
                    words = mention.split()
                    if len(words) > 1:
                        # Check if the first word exists as an entity at this position
                        first_word = words[0]
                        first_word_key = (first_word, pos)
                        if first_word_key in value_to_entity_id:
                            # Create a new entity for the full mention (without a label) for coreference relationships
                            # This allows pronouns/references to point to the full name entity
                            # Note: We allow this even if pos is in used_positions because it's a different span
                            end_pos = pos + len(mention)
                            entity_id = f"ent-{entity_id_counter}"

                            entities.append(
                                {
                                    "id": entity_id,
                                    "from_name": "entities",
                                    "to_name": "text",
                                    "type": "labels",
                                    "value": {
                                        "start": pos,
                                        "end": end_pos,
                                        "text": mention,
                                        "labels": [],  # No label - just for coreference relationships
                                    },
                                }
                            )

                            value_to_entity_id[key] = entity_id
                            mention_entity_ids_for_mention.append((entity_id, pos))
                            # Don't add to used_positions since this is for coreference, not privacy masking
                            entity_id_counter += 1
                            continue

                    # Skip if already used (for single-word mentions that aren't split entities)
                    if pos in used_positions:
                        continue

                    # This is a new mention (likely a pronoun or reference)
                    end_pos = pos + len(mention)
                    entity_id = f"ent-{entity_id_counter}"

                    # Determine label based on entity type
                    if entity_type.lower() == "person":
                        label = "PRONOUN"  # Assume mentions not in privacy_mask are pronouns
                    elif entity_type.lower() == "organization":
                        label = "REFERENCE"  # References to organizations
                    else:
                        label = "MENTION"  # Generic mention

                    entities.append(
                        {
                            "id": entity_id,
                            "from_name": "entities",
                            "to_name": "text",
                            "type": "labels",
                            "value": {
                                "start": pos,
                                "end": end_pos,
                                "text": mention,
                                "labels": [label],
                            },
                        }
                    )

                    value_to_entity_id[key] = entity_id
                    mention_entity_ids_for_mention.append((entity_id, pos))
                    used_positions.add(pos)
                    entity_id_counter += 1
                else:
                    # Already exists, add to tracking
                    mention_entity_ids_for_mention.append((value_to_entity_id[key], pos))

            # Store mapping for relation creation
            if mention_entity_ids_for_mention:
                if mention not in mention_to_entity_ids:
                    mention_to_entity_ids[mention] = []
                mention_to_entity_ids[mention].extend(mention_entity_ids_for_mention)

    # Create relation annotations for coreferences
    relations = []

    for coref_cluster in coreferences:
        mentions_raw = coref_cluster["mentions"]

        # Normalize mentions: handle both old format (list of strings) and new format (list of objects)
        mentions = []
        for mention_item in mentions_raw:
            if isinstance(mention_item, str):
                # Old format: just a string
                mentions.append({"text": mention_item})
            else:
                # New format: object with text, type, etc.
                mentions.append(mention_item)

        # Find entity IDs for all mentions in this cluster
        mention_entity_ids = []

        for mention_obj in mentions:
            # Extract text from mention object (handles both old string format and new object format)
            if isinstance(mention_obj, dict):
                mention = mention_obj["text"]
            else:
                mention = mention_obj
            # First check if we already tracked this mention
            if mention in mention_to_entity_ids:
                # Use ALL entity IDs for this mention (handles multiple occurrences)
                for entity_id, _ in mention_to_entity_ids[mention]:
                    mention_entity_ids.append((entity_id, mention))
            else:
                # Fallback: try to find in value_to_entity_id using word boundaries
                # Try exact match first with word boundaries
                positions = find_all_occurrences(text, mention, use_word_boundaries=True)

                # If no exact match, try case-insensitive with word boundaries
                if not positions:
                    mention_lower = mention.lower()
                    # Use regex for case-insensitive word boundary matching
                    escaped_mention = re.escape(mention_lower)
                    pattern = r'\b' + escaped_mention + r'\b'
                    positions = []
                    for match in re.finditer(pattern, text, re.IGNORECASE):
                        positions.append(match.start())

                for pos in positions:
                    key = (mention, pos)
                    if key in value_to_entity_id:
                        mention_entity_ids.append((value_to_entity_id[key], mention))
                        # Continue to find all occurrences, not just the first one

        # Create relations from pronouns/references to the main entity (first mention)
        # The main entity is typically the first mention in the cluster (usually the full name)
        if len(mention_entity_ids) > 1:
            main_entity_id = mention_entity_ids[0][0]

            for ref_entity_id, _ in mention_entity_ids[1:]:
                relations.append(
                    {
                        "from_id": ref_entity_id,
                        "to_id": main_entity_id,
                        "type": "relation",
                        "direction": "right",
                    }
                )

    # Combine entities and relations in the result
    result = entities + relations

    # Create the Label Studio format
    labelstudio_format = {
        "data": {"text": text},
        "predictions": [
            {"model_version": "reviewed-v1", "score": 1.0, "result": result}
        ],
    }

    return labelstudio_format


def convert_to_labelstudio(
    sample: dict[str, Any], language: str | None = None, country: str | None = None
) -> dict[str, Any]:
    """
    Convert a single reviewed sample to Label Studio format with metadata.

    Args:
        sample: Dictionary with 'text', 'privacy_mask', and 'coreferences' keys
        language: Optional language code to add to data section
        country: Optional country to add to data section

    Returns:
        Label Studio format dictionary with 'data' and 'predictions',
        including language and country metadata if provided
    """
    # Convert sample using the base conversion function
    converted = convert_sample_to_labelstudio(sample)

    # Add metadata from the sample if not explicitly provided
    if language is None and "language" in sample:
        language = sample["language"]
    if country is None and "country" in sample:
        country = sample["country"]

    # Add metadata to the data section
    if language is not None:
        converted["data"]["language"] = language
    if country is not None:
        converted["data"]["country"] = country

    # Add file_name if present in sample
    if "file_name" in sample:
        converted["data"]["file_name"] = sample["file_name"]

    return converted


def convert_all_samples_to_labelstudio(samples_dir: str, output_dir: str) -> None:
    """
    Convert all reviewed samples in a directory to Label Studio format.

    Args:
        samples_dir: Directory containing reviewed sample JSON files
        output_dir: Path to output directory for Label Studio JSON files
    """
    samples_path = Path(samples_dir)

    if not samples_path.exists():
        raise ValueError(f"Samples directory does not exist: {samples_dir}")

    # Read all JSON files from the directory
    json_files = sorted(samples_path.glob("*.json"))

    if not json_files:
        raise ValueError(f"No JSON files found in {samples_dir}")

    print(f"Found {len(json_files)} sample files")

    # Ensure output directory exists
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    counter = 0
    for json_file in json_files:
        print(f"Converting {json_file.name}...")

        try:
            # Read the sample
            with open(json_file, encoding="utf-8") as f:
                sample = json.load(f)

            # Convert to Label Studio format
            converted = convert_to_labelstudio(sample)

            # Write to output directory
            output_file = output_path / json_file.name
            with open(output_file, "w", encoding="utf-8") as f:
                json.dump(converted, f, indent=2, ensure_ascii=False)

            counter += 1

        except Exception as e:
            print(f"Error converting {json_file.name}: {e}")
            continue

    print(f"\nSuccessfully converted {counter} samples")
    print(f"Output written to: {output_dir}")


def main(argv):
    """Main function to convert reviewed samples to Label Studio format."""
    del argv  # Unused

    convert_all_samples_to_labelstudio(FLAGS.samples_dir, FLAGS.output_dir)


if __name__ == "__main__":
    app.run(main)
