"""Generate batch requests for NER and coreference tasks."""

import json
import random
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

from absl import logging
from tqdm import tqdm

# Add project root to sys.path for imports when running as a script
project_root = Path(__file__).parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

try:
    from .label_utils import LabelUtils
    from .prompts import PromptBuilder
except ImportError:
    from label_utils import LabelUtils
    from prompts import PromptBuilder


class BatchRequestGenerator:
    """Generates batch requests for NER and coreference tasks."""

    def __init__(
        self,
        api_model: str,
        num_samples: int,
        output_dir: str,
        max_workers: int | None = None,
    ):
        """
        Initialize batch request generator.

        Args:
            api_model: Model name to use for generation
            num_samples: Number of samples to generate
            output_dir: Output directory for batch files
            max_workers: Maximum number of parallel workers
        """
        self.api_model = api_model
        self.num_samples = num_samples
        self.output_dir = Path(output_dir)
        self.max_workers = max_workers or min(32, num_samples + 4)

    def get_languages_countries(
        self, language_count: int = 10, is_testing: bool = False
    ) -> list[tuple[str, str]]:
        """
        Get a list of languages and their countries.

        Args:
            language_count: Number of languages to include
            is_testing: Whether running in testing mode

        Returns:
            List of (language, country) tuples
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
            "Dutch": ["Netherlands", "Belgium"],
            "Danish": ["Denmark"],
        }

        if is_testing:
            return [("English", "United States")]
        else:
            rs = []
            for _ in range(language_count):
                key = random.choice(list(LANGUAGES_COUNTRIES.keys()))
                value = random.choice(LANGUAGES_COUNTRIES[key])
                rs.append((key, value))
            return rs

    def get_pii_labels(
        self, all_labels: bool = False, return_count: int = 10, seed: int | None = None
    ) -> dict[str, dict]:
        """
        Get PII labels with their human-readable descriptions.

        Args:
            all_labels: Whether to return all labels
            return_count: Number of labels to return if not all_labels
            seed: Random seed for label selection (for variation)

        Returns:
            Dictionary of label names to descriptions
        """
        labels = LabelUtils.LABEL_DESCRIPTIONS.copy()

        if not all_labels:
            labels = LabelUtils.select_label_subset(labels, return_count, seed=seed)
        return labels

    def _generate_single_ner_request(
        self,
        sample_index: int,
        is_testing: bool,
        language_count: int,
    ) -> dict:
        """
        Generate a single NER batch request.

        Args:
            sample_index: Index of the sample to generate
            is_testing: Whether running in testing mode
            language_count: Number of languages to include

        Returns:
            Batch request dictionary with metadata
        """
        # Use sample_index as seed for variation
        sample_seed = sample_index if not is_testing else 42

        # Get languages and labels
        languages_countries = self.get_languages_countries(
            is_testing=is_testing, language_count=language_count
        )
        labels = self.get_pii_labels(return_count=4, seed=sample_seed)

        # Build the NER prompt
        prompt, language, country = PromptBuilder.build_ner_generation_prompt(
            labels, languages_countries, sample_index=sample_index
        )

        # generate random temperature between 0.01 and 1
        temperature = random.uniform(0.01, 1)
        # Create batch request in Doubleword format
        batch_request = {
            "custom_id": f"ner-request-{sample_index}",
            "method": "POST",
            "url": "/v1/chat/completions",
            "body": {
                "model": self.api_model,
                "messages": [{"role": "user", "content": prompt}],
                "temperature": temperature,
                "metadata": {
                    "language": language,
                    "country": country,
                    "sample_index": sample_index,
                },
            },
        }

        return batch_request

    def generate_ner_batch_requests(
        self,
        output_file: str = "batch_requests_ner.jsonl",
        is_testing: bool = False,
        language_count: int = 5,
    ) -> Path:
        """
        Generate NER batch requests and save to JSONL file.

        Args:
            output_file: Output JSONL filename
            is_testing: Whether running in testing mode
            language_count: Number of languages to include

        Returns:
            Path to the generated JSONL file
        """
        batch_requests = []

        # Generate requests in parallel
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            # Submit all tasks
            future_to_index = {
                executor.submit(
                    self._generate_single_ner_request, i, is_testing, language_count
                ): i
                for i in range(self.num_samples)
            }

            # Collect results with progress bar
            with tqdm(total=self.num_samples, desc="Generating NER requests") as pbar:
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
        output_path = self.output_dir / output_file
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, "w") as f:
            for request in batch_requests:
                f.write(json.dumps(request) + "\n")

        logging.info(
            f"Generated {self.num_samples} NER batch requests in {output_path}"
        )
        return output_path

    def _generate_single_coref_request(
        self,
        sample_data: dict[str, Any],
    ) -> dict:
        """
        Generate a single coreference batch request from NER data.

        Args:
            sample_data: Dictionary containing NER sample and metadata

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
                "model": self.api_model,
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
        self,
        ner_samples: list[dict[str, Any]],
        output_file: str = "batch_requests_coref.jsonl",
    ) -> Path:
        """
        Generate coreference batch requests from NER samples.

        Args:
            ner_samples: List of NER samples with metadata
            output_file: Output JSONL filename

        Returns:
            Path to the generated JSONL file
        """
        num_samples = len(ner_samples)
        batch_requests = []

        # Generate requests in parallel
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            # Submit all tasks
            future_to_index = {
                executor.submit(self._generate_single_coref_request, sample): i
                for i, sample in enumerate(ner_samples)
            }

            # Collect results with progress bar
            with tqdm(total=num_samples, desc="Generating coref requests") as pbar:
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
        output_path = self.output_dir / output_file
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, "w") as f:
            for request in batch_requests:
                f.write(json.dumps(request) + "\n")

        logging.info(
            f"Generated {num_samples} coreference batch requests in {output_path}"
        )
        return output_path

    def _generate_single_review_request(
        self,
        sample_data: dict[str, Any],
    ) -> dict:
        """
        Generate a single review batch request from a completed sample.

        Args:
            sample_data: Dictionary containing sample and metadata

        Returns:
            Batch request dictionary
        """
        sample = sample_data["sample"]
        custom_id = sample_data["custom_id"]
        language = sample_data.get("language", "English")
        country = sample_data.get("country", "United States")

        # Get all PII labels for the review prompt
        all_labels = LabelUtils.LABEL_DESCRIPTIONS.copy()
        expected_labels = ", ".join(all_labels.keys())

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
                "model": self.api_model,
                "messages": [{"role": "user", "content": prompt}],
                "metadata": {
                    "language": language,
                    "country": country,
                    "source_sample_id": custom_id,
                },
            },
        }

        return batch_request

    def generate_review_batch_requests(
        self,
        samples: list[dict[str, Any]],
        output_file: str = "batch_requests_review.jsonl",
    ) -> Path:
        """
        Generate review batch requests from completed samples.

        Args:
            samples: List of samples with metadata (from coref completion)
            output_file: Output JSONL filename

        Returns:
            Path to the generated JSONL file
        """
        num_samples = len(samples)
        batch_requests = []

        # Generate requests in parallel
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            # Submit all tasks
            future_to_index = {
                executor.submit(self._generate_single_review_request, sample): i
                for i, sample in enumerate(samples)
            }

            # Collect results with progress bar
            with tqdm(total=num_samples, desc="Generating review requests") as pbar:
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
        output_path = self.output_dir / output_file
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, "w") as f:
            for request in batch_requests:
                f.write(json.dumps(request) + "\n")

        logging.info(f"Generated {num_samples} review batch requests in {output_path}")
        return output_path
