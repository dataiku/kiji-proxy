import json
import os
import random
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
    from .label_utils import LabelUtils
    from .prompts import PromptBuilder
    from .schemas import get_coreference_sample_schema, get_ner_sample_schema
except ImportError:
    # Allow running as a script
    from label_utils import LabelUtils
    from prompts import PromptBuilder
    from schemas import get_coreference_sample_schema, get_ner_sample_schema

# Load .env file from root directory
root_dir = Path(__file__).parent.parent.parent
env_path = root_dir / ".env"
load_dotenv(env_path)

# Define absl flags
FLAGS = flags.FLAGS

try:
    flags.DEFINE_integer("num_samples", 5, "Number of samples to generate")
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
        "ner_batch_file",
        "batch_requests_ner.jsonl",
        "Output JSONL file for NER batch requests",
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


@dataclass
class TrainingSetConfig:
    """Configuration for training set."""

    num_samples: int = 5
    output_dir: str = "model/dataset"
    api_model: str = "openai/gpt-oss-120b"

    def get_languages_countries(
        self, language_count: int = 10, is_testing: bool = False
    ) -> list[tuple[str, str]]:
        """Get a list of languages and their countries.

        TODO: Expand list of countries to generate more diverse dataset.
        Suggested Countries:
            "Italian", "Portuguese", "Swedish", "Norwegian", "Finnish", "Estonian", "Latvian",
            "Lithuanian", "Polish", "Romanian", "Russian", "Turkish", "Ukrainian", "Chinese", "Swahili",
            "Arabic", "Hausa", "Yoruba", "Zulu", "Amharic", "Afrikaans", "Thai", "Vietnamese", "Indonesian",
            "Malay", "Tagalog", "Burmese", "Malayalam", "Lao", "Khmer",
        """
        LANGUAGES_COUNTRIES = {
            "English": [
                "United States",
                "United Kingdom",
                "Canada",
                "Australia",
                "Ireland",
                "New Zealand",
            ],
            "German": ["Germany", "Austria", "Switzerland"],
            "French": ["France", "Belgium", "Canada", "Switzerland", "Luxembourg"],
            "Spanish": ["Spain", "Mexico", "Argentina", "Colombia", "Peru", "Chile"],
            "Dutch": [
                "Netherlands",
                "Belgium",
            ],  # excl. "Suriname", "Aruba", "Curaçao", "Sint Maarten" for now
            "Danish": ["Denmark"],  # excl. "Greenland", "Faroe Islands" for now
        }

        if is_testing:
            return [("English", "United States")]
        else:
            rs = []
            for _ in range(language_count):
                # pick a random key, value
                key = random.choice(list(LANGUAGES_COUNTRIES.keys()))
                value = random.choice(LANGUAGES_COUNTRIES[key])
                rs.append((key, value))
            return rs

    def get_pii_labels(
        self, all_labels: bool = False, return_count: int = 10, seed: int | None = None
    ) -> dict[str, str]:
        """
        Get PII labels with their human-readable descriptions.

        Args:
            all_labels: Whether to return all labels
            return_count: Number of labels to return if not all_labels
            seed: Random seed for label selection (for variation)
        """
        # Use the centralized label descriptions from LabelUtils
        labels = LabelUtils.LABEL_DESCRIPTIONS.copy()

        if not all_labels:
            labels = LabelUtils.select_label_subset(labels, return_count, seed=seed)
        return labels


def generate_single_ner_request(
    sample_index: int,
    config: TrainingSetConfig,
    is_testing: bool,
    language_count: int,
) -> dict:
    """
    Generate a single NER batch request.

    Args:
        sample_index: Index of the sample to generate
        config: TrainingSetConfig instance
        is_testing: Whether running in testing mode
        language_count: Number of languages to include

    Returns:
        Batch request dictionary with metadata
    """
    # Use sample_index as seed for variation
    sample_seed = sample_index if not is_testing else 42

    # Get languages and labels
    languages_countries = config.get_languages_countries(
        is_testing=is_testing, language_count=language_count
    )
    labels = config.get_pii_labels(return_count=4, seed=sample_seed)

    # Build the NER prompt
    prompt, language, country = PromptBuilder.build_ner_generation_prompt(
        labels, languages_countries, sample_index=sample_index
    )

    # Create batch request in Doubleword format
    batch_request = {
        "custom_id": f"ner-request-{sample_index}",
        "method": "POST",
        "url": "/v1/chat/completions",
        "body": {
            "model": config.api_model,
            "messages": [{"role": "user", "content": prompt}],
            "metadata": {
                "language": language,
                "country": country,
                "sample_index": sample_index,
            },
        },
    }

    return batch_request


def generate_ner_batch_requests(
    config: TrainingSetConfig,
    num_samples: int,
    batch_file: str,
    is_testing: bool = False,
    language_count: int = 5,
    max_workers: int | None = None,
) -> str:
    """
    Generate NER batch requests for Doubleword API and save to JSONL file.

    Args:
        config: TrainingSetConfig instance
        num_samples: Number of prompts to generate
        batch_file: Output JSONL file path
        is_testing: Whether running in testing mode
        language_count: Number of languages to include
        max_workers: Maximum number of parallel workers

    Returns:
        Path to the generated JSONL file
    """
    # Determine number of workers
    if max_workers is None:
        max_workers = min(32, num_samples + 4)

    batch_requests = []

    # Generate requests in parallel
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit all tasks
        future_to_index = {
            executor.submit(
                generate_single_ner_request, i, config, is_testing, language_count
            ): i
            for i in range(num_samples)
        }

        # Collect results with progress bar
        with tqdm(total=num_samples, desc="Generating NER prompts") as pbar:
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
    output_path = Path(config.output_dir) / batch_file
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w") as f:
        for request in batch_requests:
            f.write(json.dumps(request) + "\n")

    logging.info(f"Generated {num_samples} NER batch requests in {output_path}")
    return str(output_path)


def submit_to_doubleword(
    batch_file_path: str, api_key: str, batch_type: str = "batch"
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
    """Main function to generate and submit NER batch requests to Doubleword.

    Note: This script now only handles NER generation. Coreference generation
    will be handled separately after NER results are available.
    """
    # Parse absl flags
    FLAGS(sys.argv)

    # Set log level
    logging.set_verbosity(getattr(logging, FLAGS.log_level.upper(), logging.INFO))

    # Create config from flags
    config = TrainingSetConfig(
        num_samples=FLAGS.num_samples,
        output_dir=FLAGS.output_dir,
        api_model=FLAGS.api_model,
    )

    # Use testing mode only if generating very few samples (for quick testing)
    is_testing = config.num_samples <= 3

    # Generate NER batch requests and store in a jsonl file
    logging.info("Step 1: Generating NER batch requests...")
    ner_batch_file_path = generate_ner_batch_requests(
        config,
        config.num_samples,
        FLAGS.ner_batch_file,
        is_testing=is_testing,
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

    # Submit NER batch to Doubleword
    logging.info("Step 2: Submitting NER batch to Doubleword...")
    ner_file_id, ner_batch_id = submit_to_doubleword(
        ner_batch_file_path, api_key, batch_type="NER"
    )

    print(f"\n{'=' * 80}")
    print(f"NER Batch submission successful!")
    print(f"{'=' * 80}")
    print(f"NER File ID: {ner_file_id}")
    print(f"NER Batch ID: {ner_batch_id}")
    print(f"NER Batch file: {ner_batch_file_path}")
    print(f"{'=' * 80}")

    print(f"\n{'=' * 80}")
    print(f"NEXT STEPS:")
    print(f"{'=' * 80}")
    print(f"1. Wait for the NER batch to complete")
    print(f"2. Download the NER results")
    print(f"3. Run the coreference generation script with the NER results as input")
    print(f"   (This will be a separate step once NER data is available)")
    print(f"{'=' * 80}")
    print(f"\nTo check NER batch status later, use:")
    print(f"  Batch ID: {ner_batch_id}")
    print(f"{'=' * 80}\n")


if __name__ == "__main__":
    main()
