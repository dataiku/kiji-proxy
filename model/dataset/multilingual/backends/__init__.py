"""Pluggable generator and decision-model backends.

Pick a backend by name (``--generator_backend`` / ``--decision_backend``):

Generators (write the text and candidate spans):
    workers-ai         Qwen3.8-27B on Cloudflare Workers AI
    openai-compatible  Any OpenAI-style server, e.g. vLLM serving Qwen locally
    doubleword         Doubleword's realtime OpenAI-compatible API
    fake               Canned texts, no credentials (for trying the UI)

Decision models (decide the label of every candidate span):
    clef               Cloudflare Clef / Clef-flash on Workers AI
    clef-local         Clef / Clef-flash served locally with ``clef_server.py``
    laya               Laya served locally with ``laya-serve``
    fake               Fixed lookup table, no credentials
"""

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from .clef_local import (
    DEFAULT_CLEF_LOCAL_MODEL,
    DEFAULT_CLEF_LOCAL_URL,
    ClefLocalDecisionModel,
)
from .fake import FakeDecisionModel, FakeGenerator
from .laya import DEFAULT_LAYA_MODEL, DEFAULT_LAYA_URL, LayaDecisionModel
from .openai_compatible import (
    DEFAULT_DOUBLEWORD_MODEL,
    DEFAULT_LOCAL_MODEL,
    DEFAULT_LOCAL_URL,
    DOUBLEWORD_URL,
    DoublewordGenerator,
    OpenAICompatibleGenerator,
)
from .protocols import DecisionModel, TextGenerator, extract_answers
from .workers_ai import (
    DEFAULT_CLEF_MODEL,
    DEFAULT_QWEN_MODEL,
    ClefDecisionModel,
    WorkersAIClient,
    WorkersAIGenerator,
)


@dataclass(frozen=True)
class BackendSpec:
    """Describes a backend for the CLI and the UI."""

    name: str
    description: str
    default_model: str
    default_api_url: str | None
    env_vars: tuple[str, ...]
    factory: Callable[..., Any]


def _workers_ai_generator(model: str, api_url: str | None) -> TextGenerator:
    return WorkersAIGenerator(model, WorkersAIClient(base_url=api_url))


def _openai_generator(model: str, api_url: str | None) -> TextGenerator:
    return OpenAICompatibleGenerator(model, api_url or DEFAULT_LOCAL_URL)


def _doubleword_generator(model: str, api_url: str | None) -> TextGenerator:
    return DoublewordGenerator(model, api_url)


def _fake_generator(model: str, api_url: str | None) -> TextGenerator:
    return FakeGenerator(delay=0.3)


def _clef(model: str, api_url: str | None) -> DecisionModel:
    return ClefDecisionModel(model, WorkersAIClient(base_url=api_url))


def _clef_local(model: str, api_url: str | None) -> DecisionModel:
    return ClefLocalDecisionModel(model, api_url)


def _laya(model: str, api_url: str | None) -> DecisionModel:
    return LayaDecisionModel(model, api_url)


def _fake_decision(model: str, api_url: str | None) -> DecisionModel:
    return FakeDecisionModel(delay=0.1)


GENERATOR_BACKENDS: dict[str, BackendSpec] = {
    "workers-ai": BackendSpec(
        "workers-ai",
        "Qwen3.8-27B on Cloudflare Workers AI",
        DEFAULT_QWEN_MODEL,
        None,
        ("CLOUDFLARE_ACCOUNT_ID", "CLOUDFLARE_API_TOKEN"),
        _workers_ai_generator,
    ),
    "openai-compatible": BackendSpec(
        "openai-compatible",
        "Local Qwen via vLLM / SGLang (OpenAI chat API)",
        DEFAULT_LOCAL_MODEL,
        DEFAULT_LOCAL_URL,
        ("QWEN_API_KEY",),
        _openai_generator,
    ),
    "doubleword": BackendSpec(
        "doubleword",
        "Doubleword realtime API",
        DEFAULT_DOUBLEWORD_MODEL,
        DOUBLEWORD_URL,
        ("DOUBLEWORD_API_KEY",),
        _doubleword_generator,
    ),
    "fake": BackendSpec(
        "fake",
        "Canned texts, no credentials",
        "fake-qwen",
        None,
        (),
        _fake_generator,
    ),
}

DECISION_BACKENDS: dict[str, BackendSpec] = {
    "clef": BackendSpec(
        "clef",
        "Cloudflare Clef-flash on Workers AI",
        DEFAULT_CLEF_MODEL,
        None,
        ("CLOUDFLARE_ACCOUNT_ID", "CLOUDFLARE_API_TOKEN"),
        _clef,
    ),
    "clef-local": BackendSpec(
        "clef-local",
        "Clef-flash served locally (clef_server.py)",
        DEFAULT_CLEF_LOCAL_MODEL,
        DEFAULT_CLEF_LOCAL_URL,
        ("CLEF_API_URL", "CLEF_API_KEY"),
        _clef_local,
    ),
    "laya": BackendSpec(
        "laya",
        "Laya served locally (laya-serve)",
        DEFAULT_LAYA_MODEL,
        DEFAULT_LAYA_URL,
        ("LAYA_API_URL", "LAYA_API_KEY"),
        _laya,
    ),
    "fake": BackendSpec(
        "fake",
        "Fixed lookup table, no credentials",
        "fake-clef",
        None,
        (),
        _fake_decision,
    ),
}


def _build(
    registry: dict[str, BackendSpec],
    kind: str,
    name: str,
    model: str | None,
    api_url: str | None,
) -> Any:
    spec = registry.get(name)
    if spec is None:
        raise ValueError(
            f"Unknown {kind} backend {name!r}. Choose from: {list(registry)}"
        )
    return spec.factory(model or spec.default_model, api_url or spec.default_api_url)


def build_generator(
    name: str, model: str | None = None, api_url: str | None = None
) -> TextGenerator:
    """Instantiate a generator backend by name."""
    return _build(GENERATOR_BACKENDS, "generator", name, model, api_url)


def build_decision_model(
    name: str, model: str | None = None, api_url: str | None = None
) -> DecisionModel:
    """Instantiate a decision backend by name."""
    return _build(DECISION_BACKENDS, "decision", name, model, api_url)


def describe_backends() -> dict[str, list[dict[str, Any]]]:
    """JSON-friendly description of every backend (used by the UI)."""

    def rows(registry: dict[str, BackendSpec]) -> list[dict[str, Any]]:
        return [
            {
                "name": spec.name,
                "description": spec.description,
                "default_model": spec.default_model,
                "default_api_url": spec.default_api_url,
                "env_vars": list(spec.env_vars),
            }
            for spec in registry.values()
        ]

    return {
        "generators": rows(GENERATOR_BACKENDS),
        "decision_models": rows(DECISION_BACKENDS),
    }


__all__ = [
    "DECISION_BACKENDS",
    "GENERATOR_BACKENDS",
    "BackendSpec",
    "ClefDecisionModel",
    "ClefLocalDecisionModel",
    "DecisionModel",
    "DoublewordGenerator",
    "FakeDecisionModel",
    "FakeGenerator",
    "LayaDecisionModel",
    "OpenAICompatibleGenerator",
    "TextGenerator",
    "WorkersAIClient",
    "WorkersAIGenerator",
    "build_decision_model",
    "build_generator",
    "describe_backends",
    "extract_answers",
]
