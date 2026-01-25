"""Dataset generation module for PII training sets."""

from .file_operations import FileManager
from .label_utils import LabelUtils
from .labelstudio import convert_to_labelstudio

__all__ = [
    "FileManager",
    "LabelUtils",
    "convert_to_labelstudio",
]


def get_training_set_classes():
    """Lazy import of training_set module to avoid circular imports with absl flags."""
    from .openai.training_set import TrainingSetConfig, TrainingSetGenerator

    return TrainingSetConfig, TrainingSetGenerator
