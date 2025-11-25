"""File operations for saving training samples."""

import hashlib
import json
import os
from datetime import datetime
from pathlib import Path
from typing import Any


class FileManager:
    """Manages file operations for training samples."""

    def __init__(self, base_output_dir: str = "dataset"):
        self.base_output_dir = base_output_dir

    def save_sample(
        self, result: dict[str, Any], subfolder: str, file_name: str | None = None
    ) -> str:
        """
        Save a sample to a JSON file.

        Args:
            result: The sample data to save
            subfolder: Subfolder within base_output_dir (e.g., "samples", "reviewed_samples")
            file_name: Optional filename. If None, generates one from timestamp and text hash.

        Returns:
            The filename that was used
        """
        output_dir = os.path.join(self.base_output_dir, subfolder)
        os.makedirs(output_dir, exist_ok=True)

        if file_name is None:
            timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
            text_hash = hashlib.sha256(result["text"].encode()).hexdigest()
            file_name = f"{timestamp}_{text_hash}.json"

        output_path = os.path.join(output_dir, file_name)
        with open(output_path, "w") as f:
            json.dump(result, f, indent=2)

        return file_name

    def ensure_directory(self, subfolder: str) -> Path:
        """Ensure a directory exists and return its Path."""
        output_dir = Path(self.base_output_dir) / subfolder
        output_dir.mkdir(parents=True, exist_ok=True)
        return output_dir
