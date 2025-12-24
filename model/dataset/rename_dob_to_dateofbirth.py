#!/usr/bin/env python3
"""
Rename DOB labels to DATEOFBIRTH in dataset files.

This script searches through all JSON files in the samples/ and reviewed_samples/
directories and renames any "DOB" labels to "DATEOFBIRTH" to maintain consistency
with the label naming scheme.

Usage:
    python rename_dob_to_dateofbirth.py [--dry-run]
"""

import argparse
import json
from pathlib import Path
from typing import Any


def rename_dob_in_privacy_mask(data: dict[str, Any]) -> tuple[dict[str, Any], int]:
    """
    Rename DOB labels to DATEOFBIRTH in the privacy_mask array.

    Args:
        data: The JSON data dictionary

    Returns:
        Tuple of (modified_data, count_of_changes)
    """
    changes = 0

    if "privacy_mask" in data and isinstance(data["privacy_mask"], list):
        for item in data["privacy_mask"]:
            if isinstance(item, dict) and item.get("label") == "DOB":
                item["label"] = "DATEOFBIRTH"
                changes += 1

    return data, changes


def process_file(file_path: Path, dry_run: bool = False) -> tuple[bool, int]:
    """
    Process a single JSON file to rename DOB labels.

    Args:
        file_path: Path to the JSON file
        dry_run: If True, only report changes without writing

    Returns:
        Tuple of (was_modified, count_of_changes)
    """
    try:
        # Read the file
        with file_path.open("r", encoding="utf-8") as f:
            data = json.load(f)

        # Rename DOB labels
        modified_data, changes = rename_dob_in_privacy_mask(data)

        if changes > 0:
            if not dry_run:
                # Write back the modified data
                with file_path.open("w", encoding="utf-8") as f:
                    json.dump(modified_data, f, indent=2, ensure_ascii=False)
                    f.write("\n")  # Add trailing newline

            return True, changes

        return False, 0

    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return False, 0


def main():
    """Main function to process all dataset files."""
    parser = argparse.ArgumentParser(
        description="Rename DOB labels to DATEOFBIRTH in dataset files"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be changed without modifying files",
    )
    args = parser.parse_args()

    # Get the script directory and dataset paths
    script_dir = Path(__file__).parent
    samples_dir = script_dir / "samples"
    reviewed_dir = script_dir / "reviewed_samples"

    total_files = 0
    total_changes = 0

    print("=" * 80)
    print("Rename DOB labels to DATEOFBIRTH")
    print("=" * 80)

    if args.dry_run:
        print("🔍 DRY RUN MODE - No files will be modified\n")

    # Process both directories
    for directory in [samples_dir, reviewed_dir]:
        if not directory.exists():
            print(f"⚠️  Directory not found: {directory}")
            continue

        print(f"\n📁 Processing: {directory.name}/")
        print("-" * 80)

        # Get all JSON files
        json_files = sorted(directory.glob("*.json"))
        dir_modified = 0
        dir_changes = 0

        for json_file in json_files:
            was_modified, changes = process_file(json_file, dry_run=args.dry_run)

            if was_modified:
                total_files += 1
                total_changes += changes
                dir_modified += 1
                dir_changes += changes

                status = "Would rename" if args.dry_run else "Renamed"
                print(f"  ✓ {status} {changes} DOB label(s) in: {json_file.name}")

        print(
            f"\n  Summary for {directory.name}/: "
            f"{dir_modified} files, {dir_changes} labels"
        )

    # Print final summary
    print("\n" + "=" * 80)
    print("📊 Final Summary")
    print("=" * 80)
    print(f"  Files modified: {total_files}")
    print(f"  Total DOB → DATEOFBIRTH changes: {total_changes}")

    if args.dry_run:
        print("\n💡 Run without --dry-run to apply changes")
    else:
        print("\n✅ All changes applied successfully!")


if __name__ == "__main__":
    main()
