import json
import os
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from absl import flags, logging
from absl.flags import DuplicateFlagError
from dotenv import load_dotenv
from openai import OpenAI
from tqdm import tqdm

# Add project root to sys.path for imports when running as a script
project_root = Path(__file__).parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

try:
    from .prompts import PromptBuilder
    from .schemas import get_coreference_sample_schema
except ImportError:
    # Allow running as a script
    from prompts import PromptBuilder
    from schemas import get_coreference_sample_schema

# Load .env file from root directory
root_dir = Path(__file__).parent.parent.parent
env_path = root_dir / ".env"
load_dotenv(env_path)

# Define absl flags
FLAGS = flags.FLAGS

try:
    flags.DEFINE_string(
        "ner_results_file",
        None,
        "Path to the NER results file (JSONL format from Doubleword batch API)",
        required=True,
    )
    flags.DEFINE_string(
        "api_model",
        "Qwen/Qwen3-VL-235B-A22B-Instruct-FP8",
        "Model name to use for generation (e.g., 'Qwen/Qwen3-VL-235B-A22B-Instruct-FP8')",
    )
    flags.DEFINE_string(
        "output_dir", "model/dataset", "Output directory for generated samples"
    )
    flags.DEFINE_string(
        "log_level", "WARNING", "Logging level (DEBUG, INFO, WARNING, ERROR)"
    )
    flags.DEFINE_string(
        "doubleword_api_key",
        None,
        "Doubleword API key (if not set, will look for DOUBLEWORD_API_KEY env var)",
    )
    flags.DEFINE_string(
        "coref_batch_file",
        "batch_requests_coref.jsonl",
        "Output JSONL file for coreference batch requests",
    )
    flags.DEFINE_integer(
        "max_workers",
        None,
        "Maximum number of parallel workers (default: min(32, num_samples + 4))",
    )
except DuplicateFlagError:
    # Flags already defined (module imported multiple times)
    pass


def parse_ner_results(ner_results_file: str) -> list[dict[str, Any]]:
    """
    Parse NER results from Doubleword batch API response.

    Args:
        ner_results_file: Path to the JSONL file with NER results

    Returns:
        List of NER samples with metadata
    """
    samples = []

    with open(ner_results_file, "r") as f:
        for line in f:
            if not line.strip():
                continue

            result = json.loads(line)

            # Extract the response content
            if "response" in result and "body" in result["response"]:
                response_body = result["response"]["body"]

                if "choices" in response_body and len(response_body["choices"]) > 0:
                    content = response_body["choices"][0]["message"]["content"]

                    # Parse the JSON content (should be an array with one NER sample)
                    try:
                        # Extract JSON from markdown code blocks if present
                        if "```json" in content:
                            content = (
                                content.split("```json")[1].split("```")[0].strip()
                            )
                        elif "```" in content:
                            content = content.split("```")[1].split("```")[0].strip()

                        ner_data = json.loads(content)

                        # If it's an array, take the first element
                        if isinstance(ner_data, list) and len(ner_data) > 0:
                            ner_sample = ner_data[0]
                        else:
                            ner_sample = ner_data

                        # Extract metadata from the original request
                        metadata = {}
                        if (
                            "body" in result.get("response", {})
                            and "metadata" in result["response"]["body"]
                        ):
                            metadata = result["response"]["body"]["metadata"]

                        # Store sample with metadata
                        samples.append(
                            {
                                "custom_id": result.get("custom_id", ""),
                                "ner_sample": ner_sample,
                                "language": metadata.get("language", "English"),
                                "country": metadata.get("country", "United States"),
                                "sample_index": metadata.get("sample_index", 0),
                            }
                        )
                    except json.JSONDecodeError as e:
                        logging.error(f"Failed to parse NER sample content: {e}")
                        logging.error(f"Content: {content}")
                        continue

    logging.info(f"Parsed {len(samples)} NER samples from {ner_results_file}")
    return samples


def generate_single_coref_request(
    sample_data: dict[str, Any],
    api_model: str,
) -> dict:
    """
    Generate a single coreference batch request from NER data.

    Args:
        sample_data: Dictionary containing NER sample and metadata
        api_model: Model name to use

    Returns:
        Batch request dictionary
    """
    ner_sample = sample_data["ner_sample"]
    language = sample_data["language"]
    country = sample_data["country"]
    sample_index = sample_data["sample_index"]
    custom_id = sample_data["custom_id"]

    # Build the coreference prompt
    prompt = PromptBuilder.build_coreference_generation_prompt(
        ner_sample, language, country
    )

    # Create batch request in Doubleword format
    batch_request = {
        "custom_id": f"coref-{custom_id}",
        "method": "POST",
        "url": "/v1/chat/completions",
        "body": {
            "model": api_model,
            "messages": [{"role": "user", "content": prompt}],
            "metadata": {
                "language": language,
                "country": country,
                "sample_index": sample_index,
                "source_ner_id": custom_id,
            },
        },
    }

    return batch_request


def generate_coref_batch_requests(
    ner_samples: list[dict[str, Any]],
    output_dir: str,
    batch_file: str,
    api_model: str,
    max_workers: int | None = None,
) -> str:
    """
    Generate coreference batch requests from NER samples.

    Args:
        ner_samples: List of NER samples with metadata
        output_dir: Output directory
        batch_file: Output JSONL file path
        api_model: Model name to use
        max_workers: Maximum number of parallel workers

    Returns:
        Path to the generated JSONL file
    """
    num_samples = len(ner_samples)

    # Determine number of workers
    if max_workers is None:
        max_workers = min(32, num_samples + 4)

    batch_requests = []

    # Generate requests in parallel
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit all tasks
        future_to_index = {
            executor.submit(generate_single_coref_request, sample, api_model): i
            for i, sample in enumerate(ner_samples)
        }

        # Collect results with progress bar
        with tqdm(total=num_samples, desc="Generating coreference prompts") as pbar:
            for future in as_completed(future_to_index):
                sample_index = future_to_index[future]
                try:
                    batch_request = future.result()
                    batch_requests.append((sample_index, batch_request))
                    pbar.update(1)
                except Exception as exc:
                    logging.error(
                        f"Sample {sample_index} generated an exception: {exc}"
                    )
                    raise

    # Sort by sample_index to maintain order
    batch_requests.sort(key=lambda x: x[0])
    batch_requests = [req for _, req in batch_requests]

    # Write to JSONL file
    output_path = Path(output_dir) / batch_file
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w") as f:
        for request in batch_requests:
            f.write(json.dumps(request) + "\n")

    logging.info(f"Generated {num_samples} coreference batch requests in {output_path}")
    return str(output_path)


def submit_to_doubleword(
    batch_file_path: str, api_key: str, batch_type: str = "coreference"
) -> tuple[str, str]:
    """
    Submit batch requests to Doubleword API.

    Args:
        batch_file_path: Path to the JSONL file with batch requests
        api_key: Doubleword API key
        batch_type: Type of batch (for logging purposes)

    Returns:
        Tuple of (file_id, batch_id)
    """
    client = OpenAI(api_key=api_key, base_url="https://api.doubleword.ai/v1")

    # Step 1: Upload the batch input file
    logging.info(f"Uploading {batch_type} batch file: {batch_file_path}")
    with open(batch_file_path, "rb") as file:
        batch_file = client.files.create(file=file, purpose="batch")

    logging.info(f"File uploaded successfully. File ID: {batch_file.id}")

    # Step 2: Create a batch job
    logging.info(f"Creating {batch_type} batch job...")
    batch = client.batches.create(
        input_file_id=batch_file.id,
        endpoint="/v1/chat/completions",
        completion_window="24h",
    )

    logging.info(f"Batch job created successfully. Batch ID: {batch.id}")

    # Step 3: Check batch status
    batch_status = client.batches.retrieve(batch.id)
    logging.info(f"Initial {batch_type} batch status: {batch_status.status}")

    return batch_file.id, batch.id


def main():
    """Main function to generate and submit coreference batch requests to Doubleword."""
    # Parse absl flags
    FLAGS(sys.argv)

    # Set log level
    logging.set_verbosity(getattr(logging, FLAGS.log_level.upper(), logging.INFO))

    # Validate input file
    if not FLAGS.ner_results_file or not Path(FLAGS.ner_results_file).exists():
        logging.error(
            f"NER results file not found: {FLAGS.ner_results_file}\n"
            "Please provide a valid --ner_results_file path."
        )
        sys.exit(1)

    # Parse NER results
    logging.info("Step 1: Parsing NER results...")
    ner_samples = parse_ner_results(FLAGS.ner_results_file)

    if not ner_samples:
        logging.error("No valid NER samples found in the results file.")
        sys.exit(1)

    # Generate coreference batch requests
    logging.info("Step 2: Generating coreference batch requests...")
    coref_batch_file_path = generate_coref_batch_requests(
        ner_samples,
        FLAGS.output_dir,
        FLAGS.coref_batch_file,
        FLAGS.api_model,
        max_workers=FLAGS.max_workers,
    )

    # Get Doubleword API key from flags or environment
    api_key = FLAGS.doubleword_api_key or os.getenv("DOUBLEWORD_API_KEY")

    if not api_key:
        logging.error(
            "Doubleword API key not found. Please set --doubleword_api_key flag "
            "or DOUBLEWORD_API_KEY environment variable."
        )
        sys.exit(1)

    # Submit coreference batch to Doubleword
    logging.info("Step 3: Submitting coreference batch to Doubleword...")
    coref_file_id, coref_batch_id = submit_to_doubleword(
        coref_batch_file_path, api_key, batch_type="Coreference"
    )

    print(f"\n{'=' * 80}")
    print(f"Coreference Batch submission successful!")
    print(f"{'=' * 80}")
    print(f"Coreference File ID: {coref_file_id}")
    print(f"Coreference Batch ID: {coref_batch_id}")
    print(f"Coreference Batch file: {coref_batch_file_path}")
    print(f"{'=' * 80}")
    print(f"\nProcessed {len(ner_samples)} NER samples")
    print(f"NER results source: {FLAGS.ner_results_file}")
    print(f"{'=' * 80}")
    print(f"\nTo check coreference batch status later, use:")
    print(f"  Batch ID: {coref_batch_id}")
    print(f"{'=' * 80}\n")


if __name__ == "__main__":
    main()
