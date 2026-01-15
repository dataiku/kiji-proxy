"""State management for dataset generation pipeline."""

import json
from datetime import datetime
from pathlib import Path
from typing import Any

from absl import logging


class PipelineState:
    """Manages pipeline state persistence for resumability."""

    def __init__(self, output_dir: str, pipeline_id: str | None = None):
        """
        Initialize pipeline state manager.

        Args:
            output_dir: Base output directory
            pipeline_id: Optional pipeline ID (generates one if not provided)
        """
        self.output_dir = Path(output_dir)
        self.pipeline_id = pipeline_id or self._generate_pipeline_id()
        self.state_file = self.output_dir / f".pipeline_state_{self.pipeline_id}.json"
        self.state = self._load_state()

    def _generate_pipeline_id(self) -> str:
        """Generate a unique pipeline ID."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        return f"dataset_{timestamp}"

    def _load_state(self) -> dict:
        """Load state from file if it exists."""
        if self.state_file.exists():
            with open(self.state_file) as f:
                state = json.load(f)
                logging.info(f"Loaded pipeline state from {self.state_file}")
                return state
        else:
            return {
                "pipeline_id": self.pipeline_id,
                "created_at": datetime.now().isoformat(),
                "stages": {},
            }

    def _save_state(self):
        """Save current state to file."""
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.state["updated_at"] = datetime.now().isoformat()

        with open(self.state_file, "w") as f:
            json.dump(self.state, f, indent=2)

        logging.debug(f"Saved pipeline state to {self.state_file}")

    def save_stage(self, stage_name: str, data: dict[str, Any]):
        """
        Save stage completion data.

        Args:
            stage_name: Name of the stage (e.g., "ner_submission", "ner_completion")
            data: Dictionary of stage data to save
        """
        self.state["stages"][stage_name] = {
            **data,
            "status": "completed",
            "completed_at": datetime.now().isoformat(),
        }
        self._save_state()
        logging.info(f"Stage '{stage_name}' marked as completed")

    def get_stage(self, stage_name: str) -> dict[str, Any] | None:
        """
        Get stage data if it exists.

        Args:
            stage_name: Name of the stage

        Returns:
            Stage data dictionary or None if stage doesn't exist
        """
        return self.state["stages"].get(stage_name)

    def is_stage_complete(self, stage_name: str) -> bool:
        """
        Check if a stage is complete.

        Args:
            stage_name: Name of the stage

        Returns:
            True if stage is complete, False otherwise
        """
        stage = self.get_stage(stage_name)
        return stage is not None and stage.get("status") == "completed"

    def get_current_stage(self) -> str:
        """
        Determine the current stage to execute.

        Returns:
            Name of the next stage to execute
        """
        # Define stage order
        stages = [
            "ner_submission",
            "ner_completion",
            "coref_submission",
            "coref_completion",
            "review_submission",  # Optional stage
            "review_completion",  # Optional stage
            "final_processing",
        ]

        for stage in stages:
            if not self.is_stage_complete(stage):
                return stage

        return "complete"

    def get_summary(self) -> str:
        """
        Get a human-readable summary of pipeline state.

        Returns:
            Formatted string summarizing the pipeline state
        """
        lines = []
        lines.append(f"Pipeline ID: {self.pipeline_id}")
        lines.append(f"Created: {self.state.get('created_at', 'Unknown')}")

        current_stage = self.get_current_stage()
        if current_stage == "complete":
            lines.append("Status: ✅ Complete")
        else:
            lines.append(f"Status: ⏸ Waiting at '{current_stage}'")

        lines.append("\nStages:")

        # Check each stage
        stage_info = [
            ("ner_submission", "NER Batch Submission"),
            ("ner_completion", "NER Batch Completion"),
            ("coref_submission", "Coreference Batch Submission"),
            ("coref_completion", "Coreference Batch Completion"),
            ("review_submission", "Review Batch Submission (Optional)"),
            ("review_completion", "Review Batch Completion (Optional)"),
            ("final_processing", "Final Processing"),
        ]

        for stage_key, stage_name in stage_info:
            stage = self.get_stage(stage_key)
            if stage:
                status_icon = "✓"
                details = []
                if "batch_id" in stage:
                    details.append(f"Batch: {stage['batch_id']}")
                if "file_id" in stage:
                    details.append(f"File: {stage['file_id']}")
                if "results_path" in stage:
                    details.append(f"Results: {stage['results_path']}")

                detail_str = ", ".join(details) if details else ""
                lines.append(f"  {status_icon} {stage_name} {detail_str}")
            else:
                lines.append(f"  ⏸ {stage_name}")

        return "\n".join(lines)

    def reset(self):
        """Reset pipeline state (clear all stages)."""
        self.state = {
            "pipeline_id": self.pipeline_id,
            "created_at": datetime.now().isoformat(),
            "stages": {},
        }
        self._save_state()
        logging.info(f"Reset pipeline state for {self.pipeline_id}")

    def delete(self):
        """Delete the state file."""
        if self.state_file.exists():
            self.state_file.unlink()
            logging.info(f"Deleted pipeline state file: {self.state_file}")

    @staticmethod
    def list_pipelines(output_dir: str) -> list[str]:
        """
        List all pipeline state files in the output directory.

        Args:
            output_dir: Directory to search for state files

        Returns:
            List of pipeline IDs
        """
        output_path = Path(output_dir)
        state_files = output_path.glob(".pipeline_state_*.json")
        return [f.stem.replace(".pipeline_state_", "") for f in state_files]
