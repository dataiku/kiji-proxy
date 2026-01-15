"""
Import annotation samples to Label Studio with predictions.

This script:
1. Checks if Label Studio is running
2. Imports all JSON files from model/dataset/annotation_samples/
3. Creates tasks with predictions in Label Studio
"""

import json
import os
import sys
from pathlib import Path

import requests
from label_studio_sdk.client import LabelStudio
from tqdm import tqdm


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
    print("4. Create an account or log in\n")
    print("5. Create a new project or use an existing one\n")
    print("6. Get your API key from:")
    print("   Account & Settings > Access Token\n")
    print("7. Set the following environment variables:")
    print("   export LABEL_STUDIO_API_KEY='your-api-key'")
    print("   export LABEL_STUDIO_PROJECT_ID='your-project-id'\n")
    print("8. (Optional) Set custom URL if not using default:")
    print("   export LABEL_STUDIO_URL='http://localhost:8080'\n")
    print("=" * 80 + "\n")


def import_predictions(
    annotation_dir: str | Path,
    base_url: str,
    api_key: str,
    project_id: str,
) -> None:
    """
    Import annotation files to Label Studio as tasks with predictions.

    Args:
        annotation_dir: Directory containing annotation JSON files
        base_url: Label Studio base URL
        api_key: Label Studio API key
        project_id: Label Studio project ID
    """
    # Check if Label Studio is running
    print(f"Checking if Label Studio is running at {base_url}...")
    if not check_label_studio_running(base_url):
        show_startup_instructions()
        sys.exit(1)

    print("✓ Label Studio is running\n")

    # Validate required environment variables
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

    # Get all JSON files from annotation directory
    annotation_path = Path(annotation_dir)
    if not annotation_path.exists():
        print(f"✗ Annotation directory does not exist: {annotation_dir}")
        sys.exit(1)

    json_files = sorted(annotation_path.glob("*.json"))
    if not json_files:
        print(f"✗ No JSON files found in {annotation_dir}")
        sys.exit(1)

    print(f"Found {len(json_files)} annotation files to import\n")

    # Import each file
    imported_count = 0
    skipped_count = 0
    error_count = 0

    for json_file in tqdm(json_files):
        try:
            # Read the annotation file
            with open(json_file, encoding="utf-8") as f:
                task_data = json.load(f)

            # Validate the structure
            if "data" not in task_data:
                print(f"⚠ Skipping {json_file.name}: Missing 'data' field")
                skipped_count += 1
                continue

            # Add file_name to data for tracking
            task_data["data"]["file_name"] = json_file.name

            # Import the task using the Label Studio SDK
            # The SDK expects a list of tasks for import
            tasks_to_import = [task_data]

            # Import tasks (this will create tasks with predictions)
            _ = ls.projects.import_tasks(
                id=project_id,
                request=tasks_to_import,
            )

            imported_count += 1

        except Exception as e:
            print(f"✗ Error importing {json_file.name}: {e}")
            error_count += 1
            continue

    # Print summary
    print("\n" + "=" * 80)
    print("IMPORT SUMMARY")
    print("=" * 80)
    print(f"Total files found:     {len(json_files)}")
    print(f"Successfully imported: {imported_count}")
    print(f"Skipped:              {skipped_count}")
    print(f"Errors:               {error_count}")
    print("=" * 80 + "\n")

    if imported_count > 0:
        print(f"✓ Successfully imported {imported_count} tasks to Label Studio")
        print(f"  View them at: {base_url}/projects/{project_id}/data")


def main():
    """Main function to import predictions to Label Studio."""
    # Get configuration from environment variables
    base_url = os.environ.get("LABEL_STUDIO_URL", "http://localhost:8080")
    api_key = os.environ.get("LABEL_STUDIO_API_KEY")
    project_id = os.environ.get("LABEL_STUDIO_PROJECT_ID")

    # Get annotation directory
    annotation_dir = Path(__file__).parent.parent / "annotation_samples"

    # Import predictions
    import_predictions(
        annotation_dir=annotation_dir,
        base_url=base_url,
        api_key=api_key,
        project_id=project_id,
    )


if __name__ == "__main__":
    main()
