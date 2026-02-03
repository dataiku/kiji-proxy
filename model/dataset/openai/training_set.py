"""Training set generation using OpenAI API."""

import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from absl import app, flags, logging
from dotenv import load_dotenv
from tqdm import tqdm

# Add project root to sys.path for imports when running as a script
project_root = Path(__file__).parent.parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

try:
    from ..file_operations import FileManager
    from ..label_utils import LabelUtils
    from ..labelstudio import convert_to_labelstudio
    from .api_clients import LLMClient, OpenAIClient
    from .prompts import PromptBuilder
    from .schemas import get_pii_sample_schema, get_review_sample_schema
except ImportError:
    from api_clients import LLMClient, OpenAIClient
    from prompts import PromptBuilder
    from schemas import get_pii_sample_schema, get_review_sample_schema

    from model.dataset.file_operations import FileManager
    from model.dataset.label_utils import LabelUtils
    from model.dataset.labelstudio import convert_to_labelstudio

# Load .env file from root directory
root_dir = Path(__file__).parent.parent.parent.parent
env_path = root_dir / ".env"
load_dotenv(env_path)

# Define absl flags
FLAGS = flags.FLAGS

_NUM_SAMPLES = flags.DEFINE_integer("num_samples", 5, "Number of samples to generate")
_API_URL = flags.DEFINE_string(
    "api_url",
    None,
    "API URL for the LLM client. Defaults to OpenAI API. Use for OpenAI-compatible APIs.",
)
_API_MODEL = flags.DEFINE_string(
    "api_model",
    "gpt-4.1-mini",
    "Model name to use for generation (default: gpt-4.1-mini for cost efficiency)",
)
_TRAINING_OUTPUT_DIR = flags.DEFINE_string(
    "training_output_dir",
    "model/dataset",
    "Output directory for generated samples",
)
_LOG_LEVEL = flags.DEFINE_string(
    "log_level",
    "WARNING",
    "Logging level (DEBUG, INFO, WARNING, ERROR)",
)
_MAX_WORKERS = flags.DEFINE_integer(
    "max_workers",
    None,
    "Maximum number of parallel workers (default: min(12, num_samples + 4))",
)


@dataclass
class TrainingSetConfig:
    """Configuration for training set generation."""

    num_samples: int = 5
    output_dir: str = "model/dataset"
    api_url: str | None = None
    api_model: str = "gpt-4.1-mini"


class TrainingSetGenerator:
    """Generator for training sets using OpenAI API."""

    def __init__(
        self,
        config: TrainingSetConfig,
        llm_client: LLMClient | None = None,
        file_manager: FileManager | None = None,
        is_testing: bool = False,
        language_count: int = 5,
    ):
        """
        Initialize the training set generator.

        Args:
            config: Configuration for training set generation
            llm_client: LLM client to use (if None, creates OpenAIClient)
            file_manager: File manager for saving samples (if None, creates default)
            is_testing: Whether running in testing mode
            language_count: Number of languages to include in the training set
        """
        self.config = config
        self.is_testing = is_testing
        self.language_count = language_count

        # Initialize LLM client (OpenAI only)
        if llm_client is None:
            if config.api_url:
                self.llm_client = OpenAIClient(
                    api_url=config.api_url,
                    model=config.api_model,
                )
            else:
                self.llm_client = OpenAIClient(model=config.api_model)
        else:
            self.llm_client = llm_client

        # Initialize file manager
        self.file_manager = file_manager or FileManager(
            base_output_dir=config.output_dir
        )

        # Create standard label mappings
        self.label2id, self.id2label = LabelUtils.create_standard_label2id()

    def generate_pii_samples(self, sample_index: int = 0) -> dict[str, Any]:
        """Generate PII samples for a given language."""
        sample_seed = sample_index if not self.is_testing else 42

        languages_countries = LabelUtils.get_languages_countries(
            is_testing=self.is_testing, language_count=self.language_count
        )

        labels = LabelUtils.get_pii_labels(return_count=4, seed=sample_seed)

        prompt, language, country = PromptBuilder.build_generation_prompt(
            labels, languages_countries, sample_index=sample_index
        )
        json_schema = get_pii_sample_schema()

        result = self.llm_client.generate(prompt, json_schema)

        # Handle different response formats
        if isinstance(result, list):
            if len(result) == 0:
                raise ValueError("LLM returned an empty list")
            if len(result) > 1:
                logging.warning(
                    f"LLM returned {len(result)} samples, using the first one"
                )
            result[0].update({"language": language, "country": country})
            return result[0]
        elif isinstance(result, dict) and "samples" in result:
            samples = result["samples"]
            if isinstance(samples, list) and len(samples) > 0:
                samples[0].update({"language": language, "country": country})
                return samples[0]
        elif isinstance(result, dict):
            result.update({"language": language, "country": country})
            return result

        raise ValueError(f"Unexpected response format: {type(result)}")

    def review_sample(self, sample: dict[str, Any]) -> dict[str, Any]:
        """Review a sample using LLM and return corrected JSON if needed."""
        all_labels = LabelUtils.get_pii_labels(all_labels=True)
        expected_labels = ", ".join(all_labels.keys())

        language = sample["language"]
        country = sample["country"]
        prompt = PromptBuilder.build_review_prompt(
            sample, expected_labels, language=language, country=country
        )
        json_schema = get_review_sample_schema()

        result = self.llm_client.review(prompt, json_schema)
        result.update({"language": language, "country": country})
        return result


def process_single_sample(
    sample_index: int,
    gen: TrainingSetGenerator,
) -> tuple[int, str]:
    """
    Process a single training sample (generate, review, convert, save).

    Args:
        sample_index: Index of the sample to generate
        gen: TrainingSetGenerator instance

    Returns:
        Tuple of (sample_index, file_name) for the saved training sample
    """
    # Generate sample
    result = gen.generate_pii_samples(sample_index=sample_index)
    logging.info(f"Sample {sample_index}: Generated PII sample")

    # Save raw sample
    file_name = gen.file_manager.save_sample(result, "data_samples/samples")

    # Review sample
    result = gen.review_sample(result)
    file_name = gen.file_manager.save_sample(
        result, "data_samples/reviewed_samples", file_name
    )
    logging.info(f"Sample {sample_index}: Reviewed sample")

    # Add file_name to result for convert_to_labelstudio
    result["file_name"] = file_name

    # Convert to training sample
    training_sample = convert_to_labelstudio(result)
    file_name = gen.file_manager.save_sample(
        training_sample, "data_samples/annotation_samples", file_name
    )
    logging.info(f"Sample {sample_index}: Converted to training sample")

    return sample_index, file_name


def main(argv):
    """Main function to generate training samples."""
    del argv  # Unused

    logging.set_verbosity(getattr(logging, _LOG_LEVEL.value.upper(), logging.INFO))

    config = TrainingSetConfig(
        num_samples=_NUM_SAMPLES.value,
        output_dir=_TRAINING_OUTPUT_DIR.value,
        api_url=_API_URL.value,
        api_model=_API_MODEL.value,
    )

    is_testing = config.num_samples <= 3
    gen = TrainingSetGenerator(config, is_testing=is_testing)

    max_workers = _MAX_WORKERS.value
    if max_workers is None:
        max_workers = min(12, config.num_samples + 4)

    if config.num_samples > 1 and max_workers > 1:
        logging.info(
            f"Processing {config.num_samples} samples in parallel with {max_workers} workers"
        )

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_index = {
                executor.submit(process_single_sample, i, gen): i
                for i in range(config.num_samples)
            }

            with tqdm(total=config.num_samples, desc="Generating samples") as pbar:
                for future in as_completed(future_to_index):
                    sample_index = future_to_index[future]
                    try:
                        idx, file_name = future.result()
                        pbar.update(1)
                    except Exception as exc:
                        logging.error(
                            f"Sample {sample_index} generated an exception: {exc}"
                        )
                        pbar.update(1)
    else:
        for i in tqdm(range(config.num_samples)):
            idx, file_name = process_single_sample(i, gen)
            print(f"Sample {idx}: Saved training sample to {file_name}")


if __name__ == "__main__":
    app.run(main)
