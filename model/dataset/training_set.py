import random
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from absl import flags, logging
from dotenv import load_dotenv
from tqdm import tqdm
from transformers import AutoTokenizer

try:
    from .api_clients import LLMClient, OllamaClient, OpenAIClient
    from .file_operations import FileManager
    from .label_utils import LabelUtils
    from .prompts import PromptBuilder
    from .schemas import get_pii_sample_schema, get_review_sample_schema
    from ..dataset.to_labelstudio import convert_to_labelstudio
except ImportError:
    # Allow running as a script
    from api_clients import LLMClient, OllamaClient, OpenAIClient
    from file_operations import FileManager
    from label_utils import LabelUtils
    from prompts import PromptBuilder
    from schemas import get_pii_sample_schema, get_review_sample_schema
    from dataset.to_labelstudio import convert_to_labelstudio

# Load .env file from root directory
# Get the root directory (parent of model/ directory)
root_dir = Path(__file__).parent.parent.parent
env_path = root_dir / ".env"
load_dotenv(env_path)

# Define absl flags
FLAGS = flags.FLAGS
flags.DEFINE_integer("num_samples", 5, "Number of samples to generate")
flags.DEFINE_boolean("use_ollama", False, "Whether to use Ollama instead of OpenAI")
flags.DEFINE_string(
    "api_url",
    None,
    "API URL for the LLM client. If OpenAI, don't touch it. This is needed for vLLM backend with OpenAI-compatible API.",
)
flags.DEFINE_string(
    "api_model",
    "openai/gpt-oss-120b",
    "Model name to use when api_url is specified (e.g., 'openai/gpt-oss-120b')",
)
flags.DEFINE_string(
    "output_dir", "model/dataset", "Output directory for generated samples"
)
flags.DEFINE_string(
    "log_level", "WARNING", "Logging level (DEBUG, INFO, WARNING, ERROR)"
)
flags.DEFINE_integer(
    "max_workers",
    None,
    "Maximum number of parallel workers (default: min(32, num_samples + 4))",
)


@dataclass
class TrainingSetConfig:
    """Configuration for training set."""

    use_ollama: bool = False
    split: str = "train"
    num_samples: int = 5
    output_dir: str = "model/dataset"
    api_url: str | None = None  # generator API URL
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


class TrainingSetGenerator:
    """Generator for training sets."""

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
            llm_client: LLM client to use (if None, creates based on config.use_ollama)
            file_manager: File manager for saving samples (if None, creates default)
            is_testing: Whether running in testing mode
            language_count: Number of languages to include in the training set
        """
        self.config = config
        self.is_testing = is_testing
        self.language_count = language_count

        # Initialize LLM client
        if llm_client is None:
            if config.use_ollama:
                self.llm_client = OllamaClient()
            else:
                if config.api_url:
                    model = config.api_model  # FIXME (Eddie): magic number vibez. Should generalize to any HF model that vllm serve <> eats on the server.
                    self.llm_client = OpenAIClient(api_url=config.api_url, model=model)
                else:
                    self.llm_client = OpenAIClient()
        else:
            self.llm_client = llm_client

        # Initialize file manager
        self.file_manager = file_manager or FileManager(
            base_output_dir=config.output_dir
        )

        # Create standard label mappings once
        self.label2id, self.id2label = LabelUtils.create_standard_label2id()

    def generate_pii_samples(self, sample_index: int = 0) -> dict[str, Any]:
        """
        Generate PII samples for a given language.

        Args:
            sample_index: Index of the sample being generated (used for seed variation)
        """
        # Use sample_index as seed to ensure different randomness per sample
        sample_seed = sample_index if not self.is_testing else 42

        languages_countries = self.config.get_languages_countries(
            is_testing=self.is_testing, language_count=self.language_count
        )

        # Pass seed to label selection for variation
        labels = self.config.get_pii_labels(return_count=4, seed=sample_seed)

        prompt = PromptBuilder.build_generation_prompt(
            labels, languages_countries, sample_index=sample_index
        )
        json_schema = get_pii_sample_schema()

        result = self.llm_client.generate(prompt, json_schema)

        # Handle different response formats
        if isinstance(result, list):
            # LLM may return a list even if we asked for one sample
            # Just take the first item
            if len(result) == 0:
                raise ValueError("LLM returned an empty list")
            # Log if we got multiple items (might be useful for debugging)
            if len(result) > 1:
                logging.warning(
                    f"LLM returned {len(result)} samples, using the first one"
                )
            return result[0]
        elif isinstance(result, dict) and "samples" in result:
            samples = result["samples"]
            if isinstance(samples, list) and len(samples) > 0:
                return samples[0]
        elif isinstance(result, dict):
            return result

        raise ValueError(f"Unexpected response format: {type(result)}")

    def review_sample(self, sample: dict[str, Any]) -> dict[str, Any]:
        """Review a sample using LLM and return corrected JSON if needed."""
        all_labels = self.config.get_pii_labels(all_labels=True)
        expected_labels = ", ".join(all_labels.keys())

        prompt = PromptBuilder.build_review_prompt(sample, expected_labels)
        json_schema = get_review_sample_schema()

        return self.llm_client.review(prompt, json_schema)

    def convert_to_training_sample(
        self, result: dict[str, Any], tokenizer: AutoTokenizer | None = None
    ) -> dict[str, Any]:
        """
        Convert the result to a training sample format.

        Note: Tokenization is now done during training, not during dataset creation.
        This method just ensures the sample has the required fields (text, privacy_mask, coreferences).

        Args:
            result: Sample dictionary with text, privacy_mask, and coreferences
            tokenizer: Optional tokenizer (kept for backward compatibility, not used)

        Returns:
            Sample dictionary with text, privacy_mask, and coreferences (no tokenization)
        """
        # Just return the raw data - tokenization will happen during training
        # Ensure required fields are present
        required_fields = ["text", "privacy_mask", "coreferences"]
        for field in required_fields:
            if field not in result:
                raise ValueError(f"Missing required field: {field}")

        # Return the sample as-is (no tokenization)
        return result


def process_single_sample(
    sample_index: int,
    gen: TrainingSetGenerator,
    tokenizer: AutoTokenizer | None = None,
) -> tuple[int, str]:
    """
    Process a single training sample (generate, review, convert, save).
    Args:
        sample_index: Index of the sample to generate
        gen: TrainingSetGenerator instance
        tokenizer: Tokenizer instance (deprecated, kept for backward compatibility)
    Returns:
        Tuple of (sample_index, file_name) for the saved training sample
    """
    # Generate sample with index for seed variation
    result = gen.generate_pii_samples(sample_index=sample_index)
    logging.info(f"Sample {sample_index}: Generated PII sample")

    # Save raw sample
    file_name = gen.file_manager.save_sample(result, "samples")

    # Review sample
    result = gen.review_sample(result)
    file_name = gen.file_manager.save_sample(result, "reviewed_samples", file_name)
    logging.info(f"Sample {sample_index}: Reviewed sample")

    # Convert to training sample by calling convert_to_labelstudio from to_labelstudio.py
    training_sample = convert_to_labelstudio(result)
    file_name = gen.file_manager.save_sample(training_sample, "training_samples", file_name)
    logging.info(f"Sample {sample_index}: Converted to training sample")

    return sample_index, file_name


def main():
    """Main function to generate training samples."""
    # Parse absl flags
    FLAGS(sys.argv)

    # Set log level
    logging.set_verbosity(getattr(logging, FLAGS.log_level.upper(), logging.INFO))

    # Create config from flags
    config = TrainingSetConfig(
        use_ollama=FLAGS.use_ollama,
        num_samples=FLAGS.num_samples,
        output_dir=FLAGS.output_dir,
        api_url=FLAGS.api_url,
        api_model=FLAGS.api_model,
    )
    # Use testing mode only if generating very few samples (for quick testing)
    # Otherwise use full diversity
    is_testing = config.num_samples <= 3
    gen = TrainingSetGenerator(config, is_testing=is_testing)
    # Tokenizer is no longer needed during dataset creation (tokenization happens during training)

    # Determine number of workers
    max_workers = FLAGS.max_workers
    if max_workers is None:
        # Default: use min(12, num_samples + 4) to avoid too many concurrent API calls
        max_workers = min(12, config.num_samples + 4)

    # Use parallel processing if we have multiple samples
    if config.num_samples > 1 and max_workers > 1:
        logging.info(
            f"Processing {config.num_samples} samples in parallel with {max_workers} workers"
        )

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Submit all tasks
            future_to_index = {
                executor.submit(process_single_sample, i, gen, None): i
                for i in range(config.num_samples)
            }

            # Process completed tasks with progress bar
            with tqdm(total=config.num_samples, desc="Generating samples") as pbar:
                for future in as_completed(future_to_index):
                    sample_index = future_to_index[future]
                    try:
                        idx, file_name = future.result()
                        # print(f"Sample {idx}: Saved training sample to {file_name}")
                        pbar.update(1)
                    except Exception as exc:
                        logging.error(
                            f"Sample {sample_index} generated an exception: {exc}"
                        )
                        pbar.update(1)
    else:
        # Sequential processing for single sample or when max_workers <= 1
        for i in tqdm(range(config.num_samples)):
            idx, file_name = process_single_sample(i, gen, None)
            print(f"Sample {idx}: Saved training sample to {file_name}")


if __name__ == "__main__":
    main()
