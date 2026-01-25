"""Process batch results and convert to training format."""

import json
import re
import sys
from pathlib import Path
from typing import Any

from absl import logging
from tqdm import tqdm

# Add project root to sys.path for imports when running as a script
project_root = Path(__file__).parent.parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

try:
    from ..file_operations import FileManager
    from ..labelstudio import convert_to_labelstudio
except ImportError:
    from model.dataset.file_operations import FileManager
    from model.dataset.labelstudio import convert_to_labelstudio


class ResultProcessor:
    """Processes batch results and converts to Label Studio format."""

    def __init__(self, file_manager: FileManager):
        """
        Initialize result processor.

        Args:
            file_manager: FileManager instance for saving samples
        """
        self.file_manager = file_manager

    def extract_json_from_content(self, content: str) -> list[dict[str, Any]]:
        """
        Extract JSON from content string, which may be wrapped in markdown code blocks.

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

    def parse_ner_results(self, content: str) -> list[dict[str, Any]]:
        """
        Parse NER results from Doubleword batch API response.

        Args:
            content: JSONL content as string

        Returns:
            List of NER samples with metadata
        """
        samples = []

        for line in content.strip().split("\n"):
            if not line.strip():
                continue

            result = json.loads(line)

            # Extract the response content
            if "response" in result and "body" in result["response"]:
                response_body = result["response"]["body"]

                if "choices" in response_body and len(response_body["choices"]) > 0:
                    content_str = response_body["choices"][0]["message"]["content"]

                    # Parse the JSON content
                    try:
                        ner_data = self.extract_json_from_content(content_str)

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
                    except (json.JSONDecodeError, ValueError) as e:
                        logging.error(f"Failed to parse NER sample content: {e}")
                        continue

        logging.info(f"Parsed {len(samples)} NER samples from results")
        return samples

    def process_batch_line(
        self, line_data: dict[str, Any], file_id: str
    ) -> list[dict[str, Any]]:
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

        if not choices:
            logging.warning(
                f"No choices found for {custom_id}. Keys: {list(line_data.keys())}"
            )
            return []

        content = choices[0].get("message", {}).get("content", "")
        if not content:
            logging.warning(f"No content found for {custom_id}")
            return []

        try:
            samples = self.extract_json_from_content(content)
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
        self,
        content: str,
        file_id: str,
    ) -> list[tuple[str, str]]:
        """
        Process the content of a batch JSONL file and save as Label Studio samples.

        Args:
            content: The JSONL content as a string
            file_id: The file ID (for logging/naming)

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

            samples = self.process_batch_line(line_data, file_id)

            for sample in samples:
                file_name = sample.pop("file_name", None)
                if file_name is None:
                    file_name = f"{file_id}_{line_num}.json"

                try:
                    # Convert to Label Studio format
                    training_sample = convert_to_labelstudio(sample)

                    # Save the sample
                    saved_file_name = self.file_manager.save_sample(
                        training_sample, "data_samples/annotation_samples", file_name
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

    def parse_samples_for_review(self, content: str) -> list[dict[str, Any]]:
        """
        Parse batch results into samples suitable for review generation.

        Args:
            content: The JSONL content as a string

        Returns:
            List of samples with metadata for review
        """
        samples_for_review = []
        lines = content.strip().split("\n")

        for line_num, line in enumerate(lines):
            if not line.strip():
                continue

            try:
                line_data = json.loads(line)
            except json.JSONDecodeError as e:
                logging.error(f"Failed to parse line {line_num + 1}: {e}")
                continue

            custom_id = line_data.get(
                "custom_id", line_data.get("customID", f"line_{line_num}")
            )

            # Extract samples from the line
            samples = self.process_batch_line(line_data, "review_prep")

            for sample in samples:
                # Remove file_name if present
                sample.pop("file_name", None)

                # Get language and country from sample or use defaults
                language = sample.get("language", "English")
                country = sample.get("country", "United States")

                samples_for_review.append(
                    {
                        "custom_id": custom_id,
                        "sample": sample,
                        "language": language,
                        "country": country,
                    }
                )

        logging.info(f"Parsed {len(samples_for_review)} samples for review")
        return samples_for_review
