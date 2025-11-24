import random
import sys
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
    from .tokenization import TokenizationProcessor
except ImportError:
    # Allow running as a script
    from api_clients import LLMClient, OllamaClient, OpenAIClient
    from file_operations import FileManager
    from label_utils import LabelUtils
    from prompts import PromptBuilder
    from schemas import get_pii_sample_schema, get_review_sample_schema
    from tokenization import TokenizationProcessor

# Load .env file from root directory
# Get the root directory (parent of model/ directory)
root_dir = Path(__file__).parent.parent
env_path = root_dir / ".env"
load_dotenv(env_path)

# Define absl flags
FLAGS = flags.FLAGS
flags.DEFINE_integer("num_samples", 5, "Number of samples to generate")
flags.DEFINE_boolean("use_ollama", False, "Whether to use Ollama instead of OpenAI")
flags.DEFINE_string("output_dir", "dataset", "Output directory for generated samples")
flags.DEFINE_string("log_level", "INFO", "Logging level (DEBUG, INFO, WARNING, ERROR)")


@dataclass
class TrainingSetConfig:
    """Configuration for training set."""

    use_ollama: bool = False
    split: str = "train"
    num_samples: int = 5
    output_dir: str = "dataset"
    model_name: str = "distilbert-base-cased"  # Tokenizer model name

    def get_languages(
        self, language_count: int = 10, seed: int = 42, is_testing: bool = False
    ) -> list[str]:
        languages = (
            "English",
            "German",
            "French",
            "Spanish",
            "Italian",
            "Portuguese",
            "Dutch",
            "Swedish",
            "Danish",
            "Norwegian",
            "Finnish",
            "Estonian",
            "Latvian",
            "Lithuanian",
            "Polish",
            "Romanian",
            "Russian",
            "Turkish",
            "Ukrainian",
            "Chinese",
            "Swahili",
            "Arabic",
            "Hausa",
            "Yoruba",
            "Zulu",
            "Amharic",
            "Afrikaans",
            "Thai",
            "Vietnamese",
            "Indonesian",
            "Malay",
            "Tagalog",
            "Burmese",
            "Malayalam",
            "Lao",
            "Khmer",
        )

        # Use a separate Random instance for languages to avoid affecting global random state
        lang_rng = random.Random(seed) if seed else random.Random()

        if is_testing:
            return [languages[0]]
        else:
            return lang_rng.sample(languages, language_count)

    def get_pii_labels(
        self, all_labels: bool = False, return_count: int = 10
    ) -> dict[str, str]:
        """Get PII labels with their human-readable descriptions."""
        # Use the centralized label descriptions from LabelUtils
        labels = LabelUtils.LABEL_DESCRIPTIONS.copy()

        if not all_labels:
            labels = LabelUtils.select_label_subset(labels, return_count)
        return labels


class TrainingSetGenerator:
    """Generator for training sets."""

    def __init__(
        self,
        config: TrainingSetConfig,
        llm_client: LLMClient | None = None,
        file_manager: FileManager | None = None,
        is_testing: bool = False,
    ):
        """
        Initialize the training set generator.

        Args:
            config: Configuration for training set generation
            llm_client: LLM client to use (if None, creates based on config.use_ollama)
            file_manager: File manager for saving samples (if None, creates default)
            is_testing: Whether running in testing mode
        """
        self.config = config
        self.is_testing = is_testing

        # Initialize LLM client
        if llm_client is None:
            if config.use_ollama:
                self.llm_client = OllamaClient()
            else:
                self.llm_client = OpenAIClient()
        else:
            self.llm_client = llm_client

        # Initialize file manager
        self.file_manager = file_manager or FileManager(base_output_dir=config.output_dir)

        # Create standard label mappings once
        self.label2id, self.id2label = LabelUtils.create_standard_label2id()

    def generate_pii_samples(self) -> dict[str, Any]:
        """Generate PII samples for a given language."""
        languages = self.config.get_languages(is_testing=self.is_testing)
        labels = self.config.get_pii_labels(return_count=4)

        prompt = PromptBuilder.build_generation_prompt(labels, languages)
        json_schema = get_pii_sample_schema()

        result = self.llm_client.generate(prompt, json_schema)

        # Handle different response formats
        if isinstance(result, list):
            if len(result) > 1:
                raise ValueError("Expected a single result, got a list with multiple items")
            return result[0] if result else {}
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
        self, result: dict[str, Any], tokenizer: AutoTokenizer
    ) -> dict[str, Any]:
        """Convert the result to a training sample with B-* and I-* labels."""
        text = result["text"]
        privacy_mask = result["privacy_mask"]
        coreferences = result["coreferences"]

        # Create tokenization processor
        processor = TokenizationProcessor(tokenizer, self.label2id, self.id2label)

        # Generate PII and coreference samples
        pii_sample = processor.create_pii_sample(text, privacy_mask)
        coreference_sample = processor.create_coreference_sample(text, coreferences)

        # Validate that tokenization is consistent
        if coreference_sample["input_ids"] != pii_sample["input_ids"]:
            raise ValueError("Input IDs do not match")
        if coreference_sample["attention_mask"] != pii_sample["attention_mask"]:
            raise ValueError("Attention masks do not match")

        input_ids = pii_sample["input_ids"]
        attention_mask = pii_sample["attention_mask"]

        # Remove duplicate keys
        for key in ["input_ids", "attention_mask", "text"]:
            pii_sample.pop(key, None)
            coreference_sample.pop(key, None)

        # Combine results
        result.update({
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "coreference_sample": coreference_sample,
            "pii_sample": pii_sample,
        })
        return result


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
    )
    gen = TrainingSetGenerator(config, is_testing=True)
    tokenizer = AutoTokenizer.from_pretrained(config.model_name)

    for i in tqdm(range(config.num_samples)):
        # Generate sample
        result = gen.generate_pii_samples()
        logging.info(result)

        # Save raw sample
        file_name = gen.file_manager.save_sample(result, "samples")

        # Review sample
        result = gen.review_sample(result)
        file_name = gen.file_manager.save_sample(result, "reviewed_samples", file_name)

        # Convert to training sample
        training_sample = gen.convert_to_training_sample(result, tokenizer)
        file_name = gen.file_manager.save_sample(
            training_sample, "training_samples", file_name
        )
        print(f"Saved training sample to {file_name}")

if __name__ == "__main__":
    main()
