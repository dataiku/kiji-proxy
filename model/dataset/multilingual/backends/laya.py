"""Laya: Convai Innovations' open-source System One decision model, served locally.

Install and start the server (Jev-compatible wire API)::

    pip install "laya[serve]"
    laya-serve            # POST http://localhost:8000/v1/systemone

Laya is an encoder (ModernBERT / mmBERT) with a 512-token window, so the
labeler sends it one span at a time with a text window around the span.
"""

import json
import os
from typing import Any

import httpx

from .http_utils import make_http_client, post_json
from .protocols import extract_answers

DEFAULT_LAYA_URL = "http://localhost:8000/v1/systemone"
DEFAULT_LAYA_MODEL = "laya"
DEFAULT_MAX_STATE_CHARS = 1200


def state_to_text(state: str | dict[str, Any]) -> str:
    """Render the labeler's structured state as plain text for an encoder."""
    if isinstance(state, str):
        return state
    parts = []
    if state.get("task"):
        parts.append(str(state["task"]))
    if state.get("text") is not None:
        parts.append(f"TEXT:\n{state['text']}")
    candidates = state.get("candidates")
    if candidates:
        lines = "\n".join(f"{c.get('id')}. {c.get('value')}" for c in candidates)
        parts.append(f"CANDIDATES:\n{lines}")
    for key, value in state.items():
        if key not in {"task", "text", "candidates"}:
            parts.append(f"{key}: {json.dumps(value, ensure_ascii=False)}")
    return "\n\n".join(parts)


class LayaDecisionModel:
    """Decision backend talking to a local ``laya-serve`` instance."""

    def __init__(
        self,
        model: str = DEFAULT_LAYA_MODEL,
        api_url: str | None = None,
        api_key: str | None = None,
        max_questions_per_request: int = 64,
        max_state_chars: int = DEFAULT_MAX_STATE_CHARS,
        timeout: float = 120.0,
        max_retries: int = 3,
        transport: httpx.BaseTransport | None = None,
    ):
        self.model = model
        self.api_url = api_url or os.getenv("LAYA_API_URL") or DEFAULT_LAYA_URL
        self.api_key = api_key or os.getenv("LAYA_API_KEY")
        self.max_questions_per_request = max_questions_per_request
        self.max_state_chars = max_state_chars
        self.max_retries = max_retries
        self._http = make_http_client(timeout, transport)

    @property
    def headers(self) -> dict[str, str]:
        return {"Authorization": f"Bearer {self.api_key}"} if self.api_key else {}

    def build_payload(
        self, state: str | dict[str, Any], questions: dict[str, dict[str, Any]]
    ) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "state": state_to_text(state),
            "questions": questions,
        }
        if self.model and self.model != DEFAULT_LAYA_MODEL:
            payload["model"] = self.model
        return payload

    def decide(
        self, state: str | dict[str, Any], questions: dict[str, dict[str, Any]]
    ) -> dict[str, dict[str, Any]]:
        if not questions:
            return {}
        body = post_json(
            self._http,
            self.api_url,
            self.build_payload(state, questions),
            self.headers,
            self.max_retries,
        )
        return extract_answers(body, set(questions))
