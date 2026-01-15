"""Doubleword API client for batch operations."""

import os
from pathlib import Path

import requests
from absl import logging
from dotenv import load_dotenv
from openai import OpenAI

# Load .env file from root directory
root_dir = Path(__file__).parent.parent.parent
env_path = root_dir / ".env"
load_dotenv(env_path)


class DoublewordClient:
    """Client for interacting with Doubleword batch API."""

    BASE_URL = "https://api.doubleword.ai/v1"

    def __init__(self, api_key: str | None = None):
        """
        Initialize Doubleword client.

        Args:
            api_key: Doubleword API key. If None, reads from DOUBLEWORD_API_KEY env var.
        """
        self.api_key = api_key or os.getenv("DOUBLEWORD_API_KEY")
        if not self.api_key:
            raise ValueError(
                "Doubleword API key not provided. Set DOUBLEWORD_API_KEY "
                "environment variable or pass api_key parameter."
            )
        self.client = OpenAI(api_key=self.api_key, base_url=self.BASE_URL)

    def upload_batch_file(self, file_path: str) -> str:
        """
        Upload batch request file to Doubleword.

        Args:
            file_path: Path to JSONL file with batch requests

        Returns:
            File ID from Doubleword
        """
        logging.info(f"Uploading batch file: {file_path}")
        with open(file_path, "rb") as f:
            batch_file = self.client.files.create(file=f, purpose="batch")

        logging.info(f"File uploaded successfully. File ID: {batch_file.id}")
        return batch_file.id

    def create_batch_job(self, file_id: str, completion_window: str = "24h") -> str:
        """
        Create a batch job from uploaded file.

        Args:
            file_id: Uploaded file ID
            completion_window: Time window for completion (default: "24h")

        Returns:
            Batch ID
        """
        logging.info(f"Creating batch job for file {file_id}...")
        batch = self.client.batches.create(
            input_file_id=file_id,
            endpoint="/v1/chat/completions",
            completion_window=completion_window,
        )

        logging.info(f"Batch job created successfully. Batch ID: {batch.id}")
        return batch.id

    def get_batch_status(self, batch_id: str) -> dict:
        """
        Get status information for a batch job.

        Args:
            batch_id: Batch job ID

        Returns:
            Dictionary with status information:
            - status: Current status (validating, in_progress, completed, failed, etc.)
            - completed: Number of completed requests
            - failed: Number of failed requests
            - total: Total number of requests
            - output_file_id: Output file ID (if completed)
        """
        batch = self.client.batches.retrieve(batch_id)
        return {
            "status": batch.status,
            "completed": getattr(batch.request_counts, "completed", 0),
            "failed": getattr(batch.request_counts, "failed", 0),
            "total": getattr(batch.request_counts, "total", 0),
            "output_file_id": batch.output_file_id
            if batch.status == "completed"
            else None,
        }

    def download_results(self, file_id: str) -> str:
        """
        Download batch results file.

        Args:
            file_id: Output file ID from completed batch

        Returns:
            Content of results file as string
        """
        url = f"{self.BASE_URL}/files/{file_id}/content"
        headers = {"Authorization": f"Bearer {self.api_key}"}

        logging.info(f"Downloading results file: {file_id}")
        response = requests.get(url, headers=headers)
        response.raise_for_status()

        # Check for incomplete file
        is_incomplete = response.headers.get("X-Incomplete") == "true"
        if is_incomplete:
            last_line = response.headers.get("X-Last-Line")
            logging.warning(
                f"Partial file downloaded (up to line {last_line}). "
                "Batch may still be running."
            )

        return response.text

    def submit_batch(self, batch_file_path: str) -> tuple[str, str]:
        """
        Upload and submit a batch in one operation.

        Args:
            batch_file_path: Path to JSONL batch file

        Returns:
            Tuple of (file_id, batch_id)
        """
        file_id = self.upload_batch_file(batch_file_path)
        batch_id = self.create_batch_job(file_id)
        return file_id, batch_id
