"""
Convert the reviewed samples to Label Studio format.
"""

import json
from pathlib import Path
from typing import Any

from absl import app, flags

FLAGS = flags.FLAGS

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

def find_all_occurrences(text: str, value: str) -> list[int]:
    """Find all start positions of value in text."""
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

            entities.append({
                "id": entity_id,
                "from_name": "entities",
                "to_name": "text",
                "type": "labels",
                "value": {
                    "start": start_pos,
                    "end": end_pos,
                    "text": value,
                    "labels": [label]
                }
            })

            # Store mapping for coreference resolution
            value_to_entity_id[(value, start_pos)] = entity_id
            used_positions.add(start_pos)
            entity_id_counter += 1

            # Only use the first occurrence of each value to avoid duplicates
            break

    # Now add entities for coreference mentions that aren't in privacy_mask
    for coref_cluster in coreferences:
        mentions = coref_cluster["mentions"]
        entity_type = coref_cluster.get("entity_type", "mention")

        for mention in mentions:
            # Find all occurrences of this mention in the text
            positions = find_all_occurrences(text, mention)

            for pos in positions:
                # Skip if already used
                if pos in used_positions:
                    continue

                # Check if this mention is already in value_to_entity_id
                key = (mention, pos)
                if key not in value_to_entity_id:
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

                    entities.append({
                        "id": entity_id,
                        "from_name": "entities",
                        "to_name": "text",
                        "type": "labels",
                        "value": {
                            "start": pos,
                            "end": end_pos,
                            "text": mention,
                            "labels": [label]
                        }
                    })

                    value_to_entity_id[key] = entity_id
                    used_positions.add(pos)
                    entity_id_counter += 1

                # Only use first occurrence
                break

    # Create relation annotations for coreferences
    relations = []

    for coref_cluster in coreferences:
        mentions = coref_cluster["mentions"]

        # Find entity IDs for all mentions in this cluster
        mention_entity_ids = []

        for mention in mentions:
            # Find this mention in the text
            positions = find_all_occurrences(text, mention)

            for pos in positions:
                key = (mention, pos)
                if key in value_to_entity_id:
                    mention_entity_ids.append((value_to_entity_id[key], mention))
                    break

        # Create relations from pronouns/references to the main entity (first mention)
        if len(mention_entity_ids) > 1:
            main_entity_id = mention_entity_ids[0][0]

            for ref_entity_id, _ in mention_entity_ids[1:]:
                relations.append({
                    "from_id": ref_entity_id,
                    "to_id": main_entity_id,
                    "type": "relation",
                    "direction": "right"
                })

    # Combine entities and relations in the result
    result = entities + relations

    # Create the Label Studio format
    labelstudio_format = {
        "data": {
            "text": text
        },
        "predictions": [
            {
                "model_version": "reviewed-v1",
                "score": 1.0,
                "result": result
            }
        ]
    }

    return labelstudio_format


def convert_to_labelstudio(samples_dir: str, output_dir: str) -> None:
    """
    Convert all reviewed samples to Label Studio format.

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
    output_path.parent.mkdir(parents=True, exist_ok=True)

    counter = 0
    for json_file in json_files:
        print(f"Converting {json_file.name}...")

        with open(json_file, encoding='utf-8') as f:
            sample = json.load(f)

        try:
            converted = convert_sample_to_labelstudio(sample)
            # carry forward the language, country, and text to the output file
            converted["data"]["language"] = sample["language"]
            converted["data"]["country"] = sample["country"]
            # converted["data"]["text"] = sample["text"]
            with open(output_path / f"{json_file.name}", 'w', encoding='utf-8') as f:
                json.dump(converted, f, indent=2)
            counter += 1
        except Exception as e:
            print(f"Error converting {json_file.name}: {e}")
            continue
        break  # TODO: Remove this

    print(f"\nSuccessfully converted {counter} samples")
    print(f"Output written to: {output_dir}")


def main(argv):
    """Main function to convert reviewed samples to Label Studio format."""
    del argv  # Unused

    convert_to_labelstudio(FLAGS.samples_dir, FLAGS.output_dir)


if __name__ == "__main__":
    app.run(main)
