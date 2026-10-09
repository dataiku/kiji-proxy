"""Multilingual PII dataset generation with pluggable backends.

An LLM backend (Qwen3.8-27B on Workers AI, a local OpenAI-compatible server,
or Doubleword) writes the text and proposes candidate spans; a decision-model
backend (Cloudflare Clef or a locally served Laya) determines the PII label
of every candidate; the result is written as Label Studio tasks.

``python -m model.dataset.multilingual.pipeline`` runs from the command line,
``python -m model.dataset.multilingual.ui`` serves a web UI that starts runs
and tracks their progress.
"""

from .backends import (
    DECISION_BACKENDS,
    GENERATOR_BACKENDS,
    build_decision_model,
    build_generator,
    describe_backends,
)
from .config import GENERATION_LANGUAGES, TEXT_TYPES, PipelineConfig
from .entity_labeler import EntityLabeler
from .labelstudio_tasks import build_labelstudio_task, write_labelstudio_import
from .progress import ProgressTracker

__all__ = [
    "DECISION_BACKENDS",
    "GENERATION_LANGUAGES",
    "GENERATOR_BACKENDS",
    "TEXT_TYPES",
    "EntityLabeler",
    "PipelineConfig",
    "ProgressTracker",
    "build_decision_model",
    "build_generator",
    "build_labelstudio_task",
    "describe_backends",
    "write_labelstudio_import",
]


def get_pipeline_class():
    """Lazy import of the pipeline module to avoid absl flag side effects on import."""
    from .pipeline import GenerationPipeline

    return GenerationPipeline
