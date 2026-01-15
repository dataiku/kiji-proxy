"""
Process doubleword batch processing results and convert to Label Studio format.

This module downloads JSONL files from the doubleword API, extracts the
annotated samples, and converts them to the same Label Studio format as training_set.py.
"""

import json
import os
import re
import sys
from pathlib import Path
from typing import Any

import requests
from absl import app, flags, logging
from absl.flags import DuplicateFlagError
from dotenv import load_dotenv
from tqdm import tqdm

# Add project root to sys.path for imports when running as a script
project_root = Path(__file__).parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

# Load .env file from root directory
root_dir = Path(__file__).parent.parent.parent
env_path = root_dir / ".env"
load_dotenv(env_path)

try:
    from .file_operations import FileManager
    from .to_labelstudio import convert_to_labelstudio
except ImportError:
    from file_operations import FileManager
    from to_labelstudio import convert_to_labelstudio

FLAGS = flags.FLAGS

try:
    flags.DEFINE_string(
        "batch_output_file_ids",
        None,
        "Comma-separated list of batch output file IDs to download and process",
    )
except DuplicateFlagError:
    pass

try:
    flags.DEFINE_string(
        "doubleword_api_key",
        None,
        "Doubleword API key (can also be set via DOUBLEWORD_API_KEY env var)",
    )
except DuplicateFlagError:
    pass

try:
    flags.DEFINE_string(
        "dw_output_dir",
        "model/dataset",
        "Output directory for processed samples",
    )
except DuplicateFlagError:
    pass

DOUBLEWORD_API_BASE = "https://api.doubleword.ai/v1"


def get_api_key() -> str:
    """Get the doubleword API key from flags or environment."""
    api_key = FLAGS.doubleword_api_key or os.getenv("DOUBLEWORD_API_KEY")
    if not api_key:
        raise ValueError(
            "Doubleword API key not provided. Set --doubleword_api_key flag "
            "or DOUBLEWORD_API_KEY environment variable."
        )
    return api_key


def download_batch_file(file_id: str, api_key: str) -> str:
    """
    Download a batch output file from the doubleword API.

    Args:
        file_id: The batch output file ID
        api_key: Doubleword API key

    Returns:
        The content of the JSONL file as a string
    """
    url = f"{DOUBLEWORD_API_BASE}/files/{file_id}/content"
    headers = {"Authorization": f"Bearer {api_key}"}

    response = requests.get(url, headers=headers)
    response.raise_for_status()

    is_incomplete = response.headers.get("X-Incomplete") == "true"
    last_line = response.headers.get("X-Last-Line")

    if is_incomplete:
        logging.warning(
            f"Partial file downloaded for {file_id} (up to line {last_line}). "
            f"Batch may still be running."
        )

    return response.text


def extract_json_from_content(content: str) -> list[dict[str, Any]]:
    """
    Extract JSON from the content string, which may be wrapped in markdown code blocks.

    The content typically looks like:
    ```json
    [{"text": "...", "privacy_mask": [...], "coreferences": [...]}]
    ```

    Args:
        content: The raw content string from the LLM response

    Returns:
        List of sample dictionaries
    """
    # Try to extract JSON from markdown code block
    json_match = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", content)
    if json_match:
        json_str = json_match.group(1).strip()
    else:
        # Assume the entire content is JSON
        json_str = content.strip()

    try:
        parsed = json.loads(json_str)
        if isinstance(parsed, list):
            return parsed
        elif isinstance(parsed, dict):
            return [parsed]
        else:
            raise ValueError(f"Unexpected JSON type: {type(parsed)}")
    except json.JSONDecodeError as e:
        logging.error(f"Failed to parse JSON: {e}")
        logging.debug(f"Content was: {content[:500]}...")
        raise


def process_batch_line(line_data: dict[str, Any], file_id: str) -> list[dict[str, Any]]:
    """
    Process a single line from the batch JSONL file.

    Args:
        line_data: Parsed JSON from a single line of the batch file
        file_id: The batch file ID to use as prefix for output filenames

    Returns:
        List of processed samples with file_name added
    """
    custom_id = line_data.get("custom_id", line_data.get("customID", "unknown"))

    # Handle different response structures
    choices = None

    if "response" in line_data:
        response = line_data["response"]

        # Check response status
        status_code = response.get("status_code")
        if status_code and status_code != 200:
            logging.warning(f"Non-200 status code ({status_code}) for {custom_id}")
            return []

        # The response body might contain the request, and choices might be elsewhere
        if "body" in response:
            body = response["body"]
            choices = body.get("choices", [])

    elif "body" in line_data:
        # Standard batch API format
        body = line_data["body"]
        choices = body.get("choices", [])

    # if not choices:
    #     logging.warning(f"Unexpected format for {custom_id}: {list(line_data.keys())}")
    #     return []

    if not choices:
        # Provide detailed debugging information
        logging.warning(
            f"No choices found for {custom_id}. "
            f"Full structure: {json.dumps(line_data, indent=2)[:1000]}"
        )
        assert 1 == 2
        return []

    content = choices[0].get("message", {}).get("content", "")
    if not content:
        logging.warning(f"No content found for {custom_id}")
        return []

    try:
        samples = extract_json_from_content(content)
        # Add file_name to each sample with file_id prefix
        for i, sample in enumerate(samples):
            if len(samples) > 1:
                sample["file_name"] = f"{file_id}-{custom_id}_{i}.json"
            else:
                sample["file_name"] = f"{file_id}-{custom_id}.json"
        return samples
    except (json.JSONDecodeError, ValueError) as e:
        logging.error(f"Failed to process {custom_id}: {e}")
        return []


def process_batch_content(
    content: str,
    file_id: str,
    file_manager: FileManager,
) -> list[tuple[str, str]]:
    """
    Process the content of a batch JSONL file.

    Args:
        content: The JSONL content as a string
        file_id: The file ID (for logging/naming)
        file_manager: FileManager instance for saving samples

    Returns:
        List of (custom_id, file_name) tuples for successfully processed samples
    """
    results = []
    lines = content.strip().split("\n")

    for line_num, line in enumerate(tqdm(lines, desc=f"Processing {file_id}")):
        if not line.strip():
            continue

        try:
            line_data = json.loads(line)
        except json.JSONDecodeError as e:
            logging.error(f"Failed to parse line {line_num + 1}: {e}")
            continue

        samples = process_batch_line(line_data, file_id)

        for sample in samples:
            file_name = sample.pop("file_name", None)
            if file_name is None:
                file_name = f"{file_id}_{line_num}.json"

            try:
                # Convert to Label Studio format
                training_sample = convert_to_labelstudio(sample)

                # Save the sample
                saved_file_name = file_manager.save_sample(
                    training_sample, "annotation_samples", file_name
                )

                custom_id = line_data.get(
                    "custom_id", line_data.get("customID", f"line_{line_num}")
                )
                results.append((custom_id, saved_file_name))
                logging.info(f"Saved sample: {saved_file_name}")

            except Exception as e:
                logging.error(
                    f"Failed to convert/save sample from line {line_num + 1}: {e}"
                )
                continue

    return results


def process_batch_files(
    file_ids: list[str],
    api_key: str,
    output_dir: str = "model/dataset",
) -> list[tuple[str, str]]:
    """
    Download and process multiple batch output files from the doubleword API.

    Args:
        file_ids: List of batch output file IDs
        api_key: Doubleword API key
        output_dir: Base output directory for samples

    Returns:
        List of (custom_id, file_name) tuples for all successfully processed samples
    """
    file_manager = FileManager(base_output_dir=output_dir)
    all_results = []

    for file_id in file_ids:
        logging.info(f"Downloading batch file: {file_id}")
        try:
            content = download_batch_file(file_id, api_key)
            results = process_batch_content(content, file_id, file_manager)
            all_results.extend(results)
            logging.info(f"Processed {len(results)} samples from {file_id}")
        except requests.RequestException as e:
            logging.error(f"Failed to download {file_id}: {e}")
            continue

    return all_results


def main(argv):
    """Main function to process doubleword batch results."""
    del argv  # Unused

    # Get API key
    try:
        api_key = get_api_key()
    except ValueError as e:
        logging.error(str(e))
        return

    # Get batch file IDs
    if not FLAGS.batch_output_file_ids:
        logging.error(
            "No batch output file IDs provided. Use --batch_output_file_ids flag."
        )
        return

    file_ids = [f.strip() for f in FLAGS.batch_output_file_ids.split(",")]

    logging.info(f"Processing {len(file_ids)} batch files")

    results = process_batch_files(file_ids, api_key, FLAGS.dw_output_dir)

    print(f"\nSuccessfully processed {len(results)} samples")
    print(f"Output written to: {FLAGS.dw_output_dir}/annotation_samples/")


if __name__ == "__main__":
    app.run(main)
