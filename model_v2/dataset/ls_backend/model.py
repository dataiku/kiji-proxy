"""
Label Studio ML backend for PII pre-annotation.

Adapted from model/syn_dataset/labelstudio/backend/model.py (Ollama-only) to
support BOTH OpenAI and Ollama, selected at runtime via PII_PROVIDER:

    PII_PROVIDER=openai  OPENAI_API_KEY=...  OPENAI_MODEL=gpt-4o        (default)
    PII_PROVIDER=ollama  OLLAMA_HOST=http://localhost:11434  OLLAMA_MODEL=gemma4:e2b

It pulls the entity label set from the project's labeling config, asks the model
for schema-constrained structured extraction, and returns Label Studio
predictions with character offsets located in the source text.

See https://labelstud.io/guide/ml_create for the ML backend integration spec.
"""

from __future__ import annotations

import json
import logging
import os
import re
from typing import Any, Iterable

from label_studio_ml.model import LabelStudioMLBase
from label_studio_ml.response import ModelResponse

logger = logging.getLogger(__name__)

DEFAULT_OPENAI_MODEL = "gpt-4o"
DEFAULT_OLLAMA_HOST = "http://localhost:11434"
DEFAULT_OLLAMA_MODEL = "gemma4:e2b"

# Used only if the labeling config isn't parseable yet (e.g. before Label Studio
# has connected the backend to a project). Keep in sync with label_config.xml.
FALLBACK_LABELS = [
    "FIRST_NAME", "MIDDLE_NAME", "LAST_NAME", "EMAIL", "PHONE", "ORG", "DATE",
    "SSN", "CREDIT_CARD", "BANK_ACCOUNT", "IP_ADDRESS", "URL", "NATIONAL_ID",
    "STREET", "CITY", "STATE", "ZIPCODE", "COUNTRY", "ADDITIONAL_ADDRESS_INFO",
]


def _find_all_occurrences(text: str, value: str) -> list[int]:
    positions = []
    start = 0
    while True:
        pos = text.find(value, start)
        if pos == -1:
            break
        positions.append(pos)
        start = pos + 1
    return positions


class PIIPreAnnotator(LabelStudioMLBase):
    """Predict PII entity spans via OpenAI or Ollama (selected by PII_PROVIDER)."""

    def setup(self):
        self.provider = os.environ.get("PII_PROVIDER", "openai").lower()
        if self.provider == "ollama":
            import ollama  # lazy: only needed for this provider
            self.model_name = os.environ.get("OLLAMA_MODEL", DEFAULT_OLLAMA_MODEL)
            self._client = ollama.Client(
                host=os.environ.get("OLLAMA_HOST", DEFAULT_OLLAMA_HOST)
            )
        else:
            from openai import OpenAI  # lazy
            self.model_name = os.environ.get("OPENAI_MODEL", DEFAULT_OPENAI_MODEL)
            self._client = OpenAI()  # reads OPENAI_API_KEY from env

        self.set("model_version", f"{self.provider}-{self.model_name}")
        logger.info(
            "PIIPreAnnotator ready (provider=%s, model=%s)",
            self.provider,
            self.model_name,
        )

    def _resolve_labeling(self) -> tuple[str, str, list[str]]:
        parsed = getattr(self, "parsed_label_config", None) or {}
        for from_name, info in parsed.items():
            if info.get("type") == "Labels":
                to_name = (info.get("to_name") or ["text"])[0]
                labels = info.get("labels") or FALLBACK_LABELS
                return from_name, to_name, labels
        return "label", "text", FALLBACK_LABELS

    def _build_schema(self, labels: list[str]) -> dict[str, Any]:
        # additionalProperties:false + full required lists -> valid for both
        # OpenAI strict structured outputs and Ollama's `format` schema.
        return {
            "type": "object",
            "properties": {
                "entities": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "label": {"type": "string", "enum": labels},
                            "text": {"type": "string"},
                        },
                        "required": ["label", "text"],
                        "additionalProperties": False,
                    },
                }
            },
            "required": ["entities"],
            "additionalProperties": False,
        }

    def _build_prompt(self, text: str, labels: list[str]) -> str:
        label_list = "\n".join(f"  - {label}" for label in labels)
        return (
            "You extract personally identifiable information (PII) from text.\n"
            "Identify every span that matches one of the following labels:\n"
            f"{label_list}\n\n"
            "Rules:\n"
            "- Return the exact substring from the text (preserve casing and punctuation).\n"
            "- Use one entry per occurrence; do not deduplicate repeated mentions.\n"
            "- Only emit a label if you are confident.\n"
            "- If no PII is present, return an empty entities array.\n\n"
            f"TEXT:\n{text}"
        )

    def _call_llm(self, text: str, labels: list[str]) -> list[dict[str, str]]:
        schema = self._build_schema(labels)
        prompt = self._build_prompt(text, labels)
        logger.info(
            "Extracting (provider=%s, model=%s, text_len=%d, labels=%d)",
            self.provider, self.model_name, len(text), len(labels),
        )
        try:
            if self.provider == "ollama":
                resp = self._client.generate(
                    model=self.model_name,
                    prompt=prompt,
                    format=schema,
                    options={"temperature": 0.0},
                )
                raw = getattr(resp, "response", None)
                if raw is None and isinstance(resp, dict):
                    raw = resp.get("response", "")
            else:
                resp = self._client.chat.completions.create(
                    model=self.model_name,
                    messages=[{"role": "user", "content": prompt}],
                    response_format={
                        "type": "json_schema",
                        "json_schema": {
                            "name": "pii_entities",
                            "schema": schema,
                            "strict": True,
                        },
                    },
                    temperature=0,
                )
                raw = resp.choices[0].message.content
        except Exception as e:
            logger.exception("LLM call failed: %s", e)
            return []

        raw = raw or ""
        try:
            payload = json.loads(raw)
        except json.JSONDecodeError:
            logger.warning("Model returned non-JSON output: %r", raw[:500])
            return []

        entities = payload.get("entities", [])
        valid = [
            e for e in entities
            if isinstance(e, dict) and e.get("label") in labels and e.get("text")
        ]
        logger.info("Model returned %d entities, %d valid", len(entities), len(valid))
        return valid

    def _to_results(
        self,
        entities: Iterable[dict[str, str]],
        source_text: str,
        from_name: str,
        to_name: str,
    ) -> list[dict[str, Any]]:
        results = []
        used_spans: set[tuple[int, int]] = set()
        entity_id = 1

        for entity in entities:
            value = entity["text"]
            label = entity["label"]
            if not value.strip():
                continue

            positions = _find_all_occurrences(source_text, value)
            if not positions:
                pattern = r"\b" + re.escape(value) + r"\b"
                positions = [
                    m.start() for m in re.finditer(pattern, source_text, re.IGNORECASE)
                ]

            for start in positions:
                end = start + len(value)
                if (start, end) in used_spans:
                    continue
                used_spans.add((start, end))
                results.append(
                    {
                        "id": f"pii-{entity_id}",
                        "from_name": from_name,
                        "to_name": to_name,
                        "type": "labels",
                        "value": {
                            "start": start,
                            "end": end,
                            "text": source_text[start:end],
                            "labels": [label],
                        },
                    }
                )
                entity_id += 1

        return results

    def predict(self, tasks, context=None, **kwargs) -> ModelResponse:
        from_name, to_name, labels = self._resolve_labeling()
        logger.info(
            "predict() with %d task(s); labeling=%s/%s, %d labels",
            len(tasks), from_name, to_name, len(labels),
        )
        predictions = []

        for i, task in enumerate(tasks):
            text = (task.get("data") or {}).get("text", "")
            if not text:
                logger.warning("Task %d has no data.text — skipping", i)
                predictions.append(
                    {"model_version": self.get("model_version"), "score": 0.0, "result": []}
                )
                continue

            entities = self._call_llm(text, labels)
            results = self._to_results(entities, text, from_name, to_name)
            logger.info("Task %d: %d spans from %d entities", i, len(results), len(entities))
            predictions.append(
                {
                    "model_version": self.get("model_version"),
                    "score": 1.0 if results else 0.0,
                    "result": results,
                }
            )

        return ModelResponse(predictions=predictions)
