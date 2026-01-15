"""
Download samples from Doubleword, generate review prompts, and submit back for batch processing.

This module:
1. Downloads generated samples from Doubleword batch API
2. Generates review prompts for each sample using PromptBuilder
3. Submits the review prompts back to Doubleword for batch processing
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
from openai import OpenAI
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
    from .label_utils import LabelUtils
    from .prompts import PromptBuilder
except ImportError:
    from label_utils import LabelUtils
    from prompts import PromptBuilder

FLAGS = flags.FLAGS

try:
    flags.DEFINE_string(
        "input_batch_file_id",
        None,
        "Batch output file ID from the generation step to download and review",
    )
except DuplicateFlagError:
    pass

try:
    flags.DEFINE_string(
        "review_api_key",
        None,
        "Doubleword API key (can also be set via DOUBLEWORD_API_KEY env var)",
    )
except DuplicateFlagError:
    pass

try:
    flags.DEFINE_string(
        "review_output_dir",
        "model/dataset",
        "Output directory for review batch file",
    )
except DuplicateFlagError:
    pass

try:
    flags.DEFINE_string(
        "review_batch_file",
        "batch_review_requests.jsonl",
        "Output JSONL file for review batch requests",
    )
except DuplicateFlagError:
    pass

try:
    flags.DEFINE_string(
        "review_api_model",
        "Qwen/Qwen3-VL-235B-A22B-Instruct-FP8",
        "Model name to use for review",
    )
except DuplicateFlagError:
    pass

try:
    flags.DEFINE_boolean(
        "submit_review_batch",
        True,
        "Whether to submit the review batch to Doubleword after generating",
    )
except DuplicateFlagError:
    pass

DOUBLEWORD_API_BASE = "https://api.doubleword.ai/v1"


def get_api_key() -> str:
    """Get the doubleword API key from flags or environment."""
    api_key = FLAGS.review_api_key or os.getenv("DOUBLEWORD_API_KEY")
    if not api_key:
        raise ValueError(
            "Doubleword API key not provided. Set --review_api_key flag "
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


def extract_sample_from_line(
    line_data: dict[str, Any],
) -> tuple[str, dict[str, Any] | None]:
    """
    Extract the sample from a batch response line.

    Args:
        line_data: Parsed JSON from a single line of the batch file

    Returns:
        Tuple of (custom_id, sample_dict or None if failed)
    """
    custom_id = line_data.get("custom_id", line_data.get("customID", "unknown"))

    # Handle different response structures
    if "response" in line_data:
        response = line_data["response"]
        if "body" in response:
            body = response["body"]
            choices = body.get("choices", [])
        else:
            choices = response.get("choices", [])
    elif "choices" in line_data:
        choices = line_data["choices"]
    else:
        logging.warning(f"Unexpected format for {custom_id}: {list(line_data.keys())}")
        return custom_id, None

    if not choices:
        logging.warning(f"No choices found for {custom_id}")
        return custom_id, None

    content = choices[0].get("message", {}).get("content", "")
    if not content:
        logging.warning(f"No content found for {custom_id}")
        return custom_id, None

    try:
        samples = extract_json_from_content(content)
        if samples:
            return custom_id, samples[0]
        return custom_id, None
    except (json.JSONDecodeError, ValueError) as e:
        logging.error(f"Failed to process {custom_id}: {e}")
        return custom_id, None


def generate_review_request(
    custom_id: str,
    sample: dict[str, Any],
    model: str,
) -> dict[str, Any]:
    """
    Generate a review batch request for a sample.

    Args:
        custom_id: The custom ID from the original request
        sample: The sample to review
        model: The model to use for review

    Returns:
        Batch request dictionary for the review
    """
    # Get all PII labels for the review prompt
    all_labels = LabelUtils.LABEL_DESCRIPTIONS.copy()
    expected_labels = ", ".join(all_labels.keys())

    # Get language and country from sample, with defaults
    language = sample.get("language", "English")
    country = sample.get("country", "United States")

    # Build the review prompt
    prompt = PromptBuilder.build_review_prompt(
        sample, expected_labels, language=language, country=country
    )

    # Create batch request in Doubleword format
    batch_request = {
        "custom_id": f"review-{custom_id}",
        "method": "POST",
        "url": "/v1/chat/completions",
        "body": {
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
        },
    }

    return batch_request


def process_and_generate_reviews(
    content: str,
    file_id: str,
    model: str,
) -> list[dict[str, Any]]:
    """
    Process batch content and generate review requests.

    Args:
        content: The JSONL content as a string
        file_id: The file ID (for logging)
        model: The model to use for reviews

    Returns:
        List of review batch request dictionaries
    """
    review_requests = []
    lines = content.strip().split("\n")

    for line_num, line in enumerate(tqdm(lines, desc=f"Processing {file_id}")):
        if not line.strip():
            continue

        try:
            line_data = json.loads(line)
        except json.JSONDecodeError as e:
            logging.error(f"Failed to parse line {line_num + 1}: {e}")
            continue

        custom_id, sample = extract_sample_from_line(line_data)

        if sample is None:
            logging.warning(f"Skipping {custom_id}: failed to extract sample")
            continue

        try:
            review_request = generate_review_request(custom_id, sample, model)
            review_requests.append(review_request)
        except Exception as e:
            logging.error(f"Failed to generate review request for {custom_id}: {e}")
            continue

    return review_requests


def save_review_requests(
    review_requests: list[dict[str, Any]],
    output_dir: str,
    batch_file: str,
) -> str:
    """
    Save review requests to a JSONL file.

    Args:
        review_requests: List of review batch request dictionaries
        output_dir: Output directory
        batch_file: Output file name

    Returns:
        Path to the saved file
    """
    output_path = Path(output_dir) / batch_file
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w") as f:
        for request in review_requests:
            f.write(json.dumps(request) + "\n")

    return str(output_path)


def submit_to_doubleword(batch_file_path: str, api_key: str) -> tuple[str, str]:
    """
    Submit batch requests to Doubleword API.

    Args:
        batch_file_path: Path to the JSONL file with batch requests
        api_key: Doubleword API key

    Returns:
        Tuple of (file_id, batch_id)
    """
    client = OpenAI(api_key=api_key, base_url="https://api.doubleword.ai/v1")

    # Step 1: Upload the batch input file
    logging.info(f"Uploading batch file: {batch_file_path}")
    with open(batch_file_path, "rb") as file:
        batch_file = client.files.create(file=file, purpose="batch")

    logging.info(f"File uploaded successfully. File ID: {batch_file.id}")

    # Step 2: Create a batch job
    logging.info("Creating batch job...")
    batch = client.batches.create(
        input_file_id=batch_file.id,
        endpoint="/v1/chat/completions",
        completion_window="24h",
    )

    logging.info(f"Batch job created successfully. Batch ID: {batch.id}")

    # Step 3: Check batch status
    batch_status = client.batches.retrieve(batch.id)
    logging.info(f"Initial batch status: {batch_status.status}")

    return batch_file.id, batch.id


def main(argv):
    """Main function to download samples, generate reviews, and submit to Doubleword."""
    del argv  # Unused

    # Get API key
    try:
        api_key = get_api_key()
    except ValueError as e:
        logging.error(str(e))
        return

    # Get input batch file ID
    if not FLAGS.input_batch_file_id:
        logging.error(
            "No input batch file ID provided. Use --input_batch_file_id flag."
        )
        return

    file_id = FLAGS.input_batch_file_id

    # Download the batch output
    print(f"Downloading batch output: {file_id}")
    try:
        content = download_batch_file(file_id, api_key)
    except requests.RequestException as e:
        logging.error(f"Failed to download batch file: {e}")
        return

    # Process and generate review requests
    print("Generating review prompts...")
    review_requests = process_and_generate_reviews(
        content, file_id, FLAGS.review_api_model
    )

    if not review_requests:
        logging.error("No review requests generated. Check the input batch file.")
        return

    # Save review requests to file
    batch_file_path = save_review_requests(
        review_requests, FLAGS.review_output_dir, FLAGS.review_batch_file
    )
    print(f"Saved {len(review_requests)} review requests to: {batch_file_path}")

    # Submit to Doubleword if requested
    if FLAGS.submit_review_batch:
        print("\nSubmitting review batch to Doubleword...")
        try:
            uploaded_file_id, batch_id = submit_to_doubleword(batch_file_path, api_key)

            print(f"\n{'=' * 60}")
            print("Review batch submission successful!")
            print(f"{'=' * 60}")
            print(f"Uploaded File ID: {uploaded_file_id}")
            print(f"Batch ID: {batch_id}")
            print(f"Review batch file: {batch_file_path}")
            print(f"{'=' * 60}")
            print("\nTo process the reviewed samples, use:")
            print(
                "  python model/dataset/training_set_doubleword_result_processing.py \\"
            )
            print("    --batch_output_file_ids=<output_file_id_from_batch>")
            print(f"{'=' * 60}\n")
        except Exception as e:
            logging.error(f"Failed to submit review batch: {e}")
            return
    else:
        print(f"\n{'=' * 60}")
        print("Review batch file generated (not submitted)")
        print(f"{'=' * 60}")
        print(f"Review batch file: {batch_file_path}")
        print(f"Number of review requests: {len(review_requests)}")
        print(f"{'=' * 60}")
        print("\nTo submit manually, upload the file to Doubleword API.")
        print(f"{'=' * 60}\n")


if __name__ == "__main__":
    app.run(main)
