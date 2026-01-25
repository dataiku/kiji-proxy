"""
Export annotations from Label Studio in original format.

This script:
1. Checks if Label Studio is running
2. Exports all tasks from Label Studio in their original format
3. Preserves data, annotations, and predictions as-is
4. Saves to model/dataset/data_samples/training_samples/
"""

import json
import os
import sys
from datetime import datetime
from pathlib import Path

import requests
from label_studio_sdk.client import LabelStudio
from tqdm import tqdm


def serialize_datetime(obj):
    """Convert datetime objects to ISO format strings."""
    if isinstance(obj, datetime):
        return obj.isoformat()
    return obj


def check_label_studio_running(base_url: str) -> bool:
    """
    Check if Label Studio is running and accessible.

    Args:
        base_url: The base URL of Label Studio

    Returns:
        True if Label Studio is accessible, False otherwise
    """
    try:
        response = requests.get(f"{base_url}/api/health", timeout=5)
        return response.status_code == 200
    except (requests.ConnectionError, requests.Timeout):
        return False


def show_startup_instructions():
    """Display instructions for starting Label Studio."""
    print("\n" + "=" * 80)
    print("LABEL STUDIO IS NOT RUNNING")
    print("=" * 80)
    print("\nTo start Label Studio locally, follow these steps:\n")
    print("1. Install Label Studio (if not already installed):")
    print("   uv pip install --extra labelstudio label-studio\n")
    print("2. Start Label Studio:")
    print("   uv run label-studio start\n")
    print("3. Open your browser and go to:")
    print("   http://localhost:8080\n")
    print("4. Set the following environment variables:")
    print("   export LABEL_STUDIO_API_KEY='your-api-key'")
    print("   export LABEL_STUDIO_PROJECT_ID='your-project-id'\n")
    print("=" * 80 + "\n")


def export_annotations(
    base_url: str,
    api_key: str,
    project_id: str,
    output_dir: str | Path,
) -> None:
    """
    Export annotations from Label Studio to local files.

    Args:
        base_url: Label Studio base URL
        api_key: Label Studio API key
        project_id: Label Studio project ID
        output_dir: Directory to save exported tasks
    """
    # Check if Label Studio is running
    print(f"Checking if Label Studio is running at {base_url}...")
    if not check_label_studio_running(base_url):
        show_startup_instructions()
        sys.exit(1)

    print("✓ Label Studio is running\n")

    # Validate required parameters
    if not api_key:
        print("✗ Error: LABEL_STUDIO_API_KEY environment variable is not set")
        print("  Set it with: export LABEL_STUDIO_API_KEY='your-api-key'")
        sys.exit(1)

    if not project_id:
        print("✗ Error: LABEL_STUDIO_PROJECT_ID environment variable is not set")
        print("  Set it with: export LABEL_STUDIO_PROJECT_ID='your-project-id'")
        sys.exit(1)

    # Initialize Label Studio client
    try:
        ls = LabelStudio(base_url=base_url, api_key=api_key)
        # Test connection by trying to get project info
        project = ls.projects.get(id=project_id)
        print(f"✓ Connected to project: {project.title} (ID: {project_id})\n")
    except Exception as e:
        print(f"✗ Failed to connect to Label Studio project: {e}")
        print("\nPlease check:")
        print("  - LABEL_STUDIO_API_KEY is correct")
        print("  - LABEL_STUDIO_PROJECT_ID is correct")
        print("  - The project exists in Label Studio")
        sys.exit(1)

    # Get all tasks from the project
    print("Fetching tasks from Label Studio...")
    tasks = list(ls.tasks.list(project=project_id))

    # Setup output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    print(f"Exporting {len(tasks)} samples to {output_path}\n")

    # Process all tasks (with and without annotations)
    exported_count = 0
    error_count = 0
    annotated_count = 0
    predicted_count = 0
    both_count = 0
    neither_count = 0

    for t in tqdm(tasks, desc="Exporting tasks"):
        try:
            # Build task in Label Studio format
            task_export = {
                "id": t.id,
                "data": dict(t.data) if hasattr(t.data, "__dict__") else t.data,
            }

            has_annotations = False
            has_predictions = False

            # Include annotations if they exist
            if t.annotations and len(t.annotations) > 0:
                task_export["annotations"] = []
                for ann in t.annotations:
                    # Convert annotation object to dict
                    if hasattr(ann, "__dict__"):
                        ann_dict = {
                            "id": ann.id if hasattr(ann, "id") else None,
                            "completed_by": ann.completed_by
                            if hasattr(ann, "completed_by")
                            else None,
                            "result": ann.result
                            if hasattr(ann, "result")
                            else ann.get("result", []),
                            "was_cancelled": ann.was_cancelled
                            if hasattr(ann, "was_cancelled")
                            else False,
                            "created_at": serialize_datetime(ann.created_at)
                            if hasattr(ann, "created_at")
                            else None,
                            "updated_at": serialize_datetime(ann.updated_at)
                            if hasattr(ann, "updated_at")
                            else None,
                        }
                    else:
                        ann_dict = ann
                    task_export["annotations"].append(ann_dict)
                has_annotations = True

            # Include predictions if they exist
            if t.predictions and len(t.predictions) > 0:
                task_export["predictions"] = []
                for pred in t.predictions:
                    # Convert prediction object to dict
                    if hasattr(pred, "__dict__"):
                        pred_dict = {
                            "id": pred.id if hasattr(pred, "id") else None,
                            "model_version": pred.model_version
                            if hasattr(pred, "model_version")
                            else None,
                            "score": pred.score if hasattr(pred, "score") else None,
                            "result": pred.result
                            if hasattr(pred, "result")
                            else pred.get("result", []),
                            "created_at": serialize_datetime(pred.created_at)
                            if hasattr(pred, "created_at")
                            else None,
                            "updated_at": serialize_datetime(pred.updated_at)
                            if hasattr(pred, "updated_at")
                            else None,
                        }
                    else:
                        pred_dict = pred
                    task_export["predictions"].append(pred_dict)
                has_predictions = True

            # Count the combinations
            if has_annotations and has_predictions:
                both_count += 1
                annotated_count += 1
                predicted_count += 1
            elif has_annotations:
                annotated_count += 1
            elif has_predictions:
                predicted_count += 1
            else:
                neither_count += 1

            # Determine filename
            filename = t.data.get("file_name", None)
            if filename is None:
                filename = f"task_{t.id:04d}.json"
            output_file = output_path / filename

            # Save to file
            with open(output_file, "w", encoding="utf-8") as f:
                json.dump(task_export, f, indent=2, ensure_ascii=False)

            exported_count += 1

        except Exception as e:
            print(f"\n✗ Error exporting task {t.id}: {e}")
            error_count += 1
            continue

    # Print summary
    print("\n" + "=" * 80)
    print("EXPORT SUMMARY")
    print("=" * 80)
    print(f"Total tasks found:     {len(tasks)}")
    print(f"Successfully exported: {exported_count}")
    print(f"Errors:               {error_count}")
    print(f"  - With annotations:  {annotated_count}")
    print(f"  - With predictions:  {predicted_count}")
    print(f"  - With both:         {both_count}")
    print(f"  - With neither:      {neither_count}")
    print("=" * 80 + "\n")

    if exported_count > 0:
        print(f"✓ Successfully exported {exported_count} tasks to {output_path}")
        print("  Tasks are in Label Studio format (data, annotations, predictions)")


def main():
    """Main function to export annotations from Label Studio."""
    # Get configuration from environment variables
    base_url = os.environ.get("LABEL_STUDIO_URL", "http://localhost:8080")
    api_key = os.environ.get("LABEL_STUDIO_API_KEY")
    project_id = os.environ.get("LABEL_STUDIO_PROJECT_ID")

    # Get output directory
    output_dir = Path(__file__).parent.parent / "data_samples" / "training_samples"

    # Export annotations
    export_annotations(
        base_url=base_url,
        api_key=api_key,
        project_id=project_id,
        output_dir=output_dir,
    )


if __name__ == "__main__":
    main()
