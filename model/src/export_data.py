"""Data export from Label Studio to local files."""

import os
import sys
from pathlib import Path
from typing import Optional

from absl import logging

# Import export function from labelstudio module
try:
    project_root = Path(__file__).parent.parent.parent
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))
    from dataset.labelstudio.export_annotations import export_annotations
except ImportError:
    from model.dataset.labelstudio.export_annotations import export_annotations


class ExportDataProcessor:
    """Handles data export from Label Studio to local files."""

    def __init__(self, config, raw_config: Optional[dict] = None):
        """
        Initialize export data processor.

        Args:
            config: Training configuration object (TrainingConfig)
            raw_config: Optional raw config dict from TOML file (for accessing labelstudio settings)
        """
        self.config = config
        self.raw_config = raw_config or {}

        # Get Label Studio settings from config file, with env var fallback
        # Base URL: config file -> env var -> default
        self.base_url = (
            self.raw_config.get("labelstudio", {}).get("base_url")
            or os.environ.get("LABEL_STUDIO_URL", "http://localhost:8080")
        )

        # API Key: config file -> env var
        self.api_key = (
            self.raw_config.get("labelstudio", {}).get("api_key")
            or os.environ.get("LABEL_STUDIO_API_KEY")
        )

        # Project ID: config file ([labelstudio].project_id or [data].labelstudio_project) -> env var
        # Convert to string if it's an int from config
        project_id_from_config = (
            self.raw_config.get("labelstudio", {}).get("project_id")
            or self.raw_config.get("data", {}).get("labelstudio_project")  # Backward compatibility
        )
        if project_id_from_config is not None:
            self.project_id = str(project_id_from_config)
        else:
            self.project_id = os.environ.get("LABEL_STUDIO_PROJECT_ID")

    def export_data(
        self,
        output_dir: Optional[str] = None,
        base_url: Optional[str] = None,
        api_key: Optional[str] = None,
        project_id: Optional[str] = None,
    ) -> dict:
        """
        Export annotations from Label Studio to local files.

        Args:
            output_dir: Directory to save exported tasks (defaults to config.training_samples_dir)
            base_url: Label Studio base URL (defaults to LABEL_STUDIO_URL env var or http://localhost:8080)
            api_key: Label Studio API key (defaults to LABEL_STUDIO_API_KEY env var)
            project_id: Label Studio project ID (defaults to LABEL_STUDIO_PROJECT_ID env var)

        Returns:
            Dictionary with export statistics and results
        """
        # Use provided values or fall back to instance attributes
        base_url = base_url or self.base_url
        api_key = api_key or self.api_key
        project_id = project_id or self.project_id
        output_dir = output_dir or getattr(
            self.config, "training_samples_dir", "model/dataset/training_samples"
        )

        # Validate required parameters
        if not api_key:
            raise ValueError(
                "Label Studio API key is required. "
                "Set it in training_config.toml under [labelstudio].api_key "
                "or LABEL_STUDIO_API_KEY environment variable, or pass api_key parameter."
            )

        if not project_id:
            raise ValueError(
                "Label Studio project ID is required. "
                "Set it in training_config.toml under [labelstudio].project_id "
                "or LABEL_STUDIO_PROJECT_ID environment variable, or pass project_id parameter."
            )

        # Ensure output directory exists
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        logging.info(f"\n📤 Exporting data from Label Studio...")
        logging.info(f"  Base URL: {base_url}")
        logging.info(f"  Project ID: {project_id}")
        logging.info(f"  Output directory: {output_path}")

        # Export annotations using the labelstudio module
        try:
            export_annotations(
                base_url=base_url,
                api_key=api_key,
                project_id=project_id,
                output_dir=output_path,
            )

            # Count exported files
            json_files = list(output_path.glob("*.json"))
            exported_count = len(json_files)

            logging.info(f"✅ Successfully exported {exported_count} samples to {output_path}")

            return {
                "exported_count": exported_count,
                "output_dir": str(output_path),
                "base_url": base_url,
                "project_id": project_id,
                "success": True,
            }

        except Exception as e:
            logging.error(f"❌ Failed to export data from Label Studio: {e}")
            raise
