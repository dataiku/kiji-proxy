"""Dataset generation module for PII training sets."""

# Lazy imports to avoid pulling in heavy dependencies (absl, transformers)
# when only using lightweight components like api_clients, prompts, schemas

__all__ = ["TrainingSetConfig", "TrainingSetGenerator"]


def __getattr__(name):
    """Lazy import for TrainingSetConfig and TrainingSetGenerator."""
    if name in ("TrainingSetConfig", "TrainingSetGenerator"):
        from .training_set import TrainingSetConfig, TrainingSetGenerator
        return locals()[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
