"""Clef / Clef-flash served locally from the open weights.

Start the server (Jev-compatible wire API, see ``clef_server.py``)::

    uv run model/dataset/multilingual/clef_server.py   # POST http://localhost:8002/v1/systemone

The request and response bodies are the same as the hosted model's, so the
labeler treats it like Clef on Workers AI: whole text, 64 questions per request.
"""

import os
from typing import Any

import httpx

from .http_utils import make_http_client, post_json
from .protocols import extract_answers

DEFAULT_CLEF_LOCAL_URL = "http://localhost:8002/v1/systemone"
DEFAULT_CLEF_LOCAL_MODEL = "Cloudflare/clef-flash"


class ClefLocalDecisionModel:
    """Decision backend talking to a local ``clef_server.py`` instance."""

    max_questions_per_request = 64
    max_state_chars = None  # same large context as the hosted model

    def __init__(
        self,
        model: str = DEFAULT_CLEF_LOCAL_MODEL,
        api_url: str | None = None,
        api_key: str | None = None,
        timeout: float = 600.0,
        max_retries: int = 3,
        transport: httpx.BaseTransport | None = None,
    ):
        self.model = model
        self.api_url = api_url or os.getenv("CLEF_API_URL") or DEFAULT_CLEF_LOCAL_URL
        self.api_key = api_key or os.getenv("CLEF_API_KEY")
        self.max_retries = max_retries
        self._http = make_http_client(timeout, transport)

    @property
    def headers(self) -> dict[str, str]:
        return {"Authorization": f"Bearer {self.api_key}"} if self.api_key else {}

    def decide(
        self, state: str | dict[str, Any], questions: dict[str, dict[str, Any]]
    ) -> dict[str, dict[str, Any]]:
        if not questions:
            return {}
        body = post_json(
            self._http,
            self.api_url,
            {"model": self.model, "state": state, "questions": questions},
            self.headers,
            self.max_retries,
        )
        return extract_answers(body, set(questions))
