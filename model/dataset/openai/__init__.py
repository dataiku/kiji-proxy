"""OpenAI-based training set generation."""

from .api_clients import LLMClient, OpenAIClient
from .prompts import PromptBuilder
from .schemas import (
    get_coreference_sample_schema,
    get_ner_sample_schema,
    get_pii_sample_schema,
    get_review_sample_schema,
)

__all__ = [
    "LLMClient",
    "OpenAIClient",
    "PromptBuilder",
    "get_coreference_sample_schema",
    "get_ner_sample_schema",
    "get_pii_sample_schema",
    "get_review_sample_schema",
]


def get_training_set_classes():
    """Lazy import of training_set module to avoid circular imports with absl flags."""
    from .training_set import TrainingSetConfig, TrainingSetGenerator

    return TrainingSetConfig, TrainingSetGenerator
