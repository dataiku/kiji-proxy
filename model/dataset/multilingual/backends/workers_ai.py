"""Cloudflare Workers AI backends: Qwen text generation and Clef decisions."""

import os
from typing import Any

import httpx

from .chat import build_chat_payload, parse_chat_response
from .http_utils import make_http_client, post_json
from .protocols import extract_answers

DEFAULT_BASE_URL = "https://api.cloudflare.com/client/v4"
DEFAULT_QWEN_MODEL = "@cf/qwen/qwen3.8-27b"
DEFAULT_CLEF_MODEL = "@cf/cloudflare/clef-flash"


class WorkersAIClient:
    """Thin REST client for Cloudflare Workers AI.

    Supports the native ``/ai/run/{model}`` endpoint and the OpenAI-compatible
    ``/ai/v1/chat/completions`` endpoint.
    """

    def __init__(
        self,
        account_id: str | None = None,
        api_token: str | None = None,
        base_url: str | None = None,
        timeout: float = 300.0,
        max_retries: int = 3,
        transport: httpx.BaseTransport | None = None,
    ):
        self.account_id = account_id or os.getenv("CLOUDFLARE_ACCOUNT_ID")
        self.api_token = api_token or os.getenv("CLOUDFLARE_API_TOKEN")
        if not self.account_id or not self.api_token:
            raise ValueError(
                "CLOUDFLARE_ACCOUNT_ID and CLOUDFLARE_API_TOKEN must be set "
                "(environment or .env file)."
            )
        # CLOUDFLARE_API_BASE_URL lets an AI Gateway or a local mock stand in.
        self.base_url = (
            base_url or os.getenv("CLOUDFLARE_API_BASE_URL") or DEFAULT_BASE_URL
        ).rstrip("/")
        self.max_retries = max_retries
        self.headers = {"Authorization": f"Bearer {self.api_token}"}
        self._http = make_http_client(timeout, transport)

    @property
    def account_url(self) -> str:
        return f"{self.base_url}/accounts/{self.account_id}/ai"

    def run(self, model: str, payload: dict[str, Any]) -> Any:
        """Call the native ``/ai/run/{model}`` endpoint and return its ``result``."""
        body = post_json(
            self._http,
            f"{self.account_url}/run/{model}",
            payload,
            self.headers,
            self.max_retries,
        )
        if isinstance(body, dict) and "success" in body:
            if not body.get("success", False):
                raise RuntimeError(f"Workers AI error: {body.get('errors')}")
            return body.get("result")
        return body

    def chat_completions(self, payload: dict[str, Any]) -> dict[str, Any]:
        """Call the OpenAI-compatible chat completions endpoint."""
        return post_json(
            self._http,
            f"{self.account_url}/v1/chat/completions",
            payload,
            self.headers,
            self.max_retries,
        )


class WorkersAIGenerator:
    """Structured text generation with a Workers AI chat model (Qwen3.8-27B)."""

    def __init__(
        self,
        model: str = DEFAULT_QWEN_MODEL,
        client: WorkersAIClient | None = None,
        temperature: float = 0.8,
        max_tokens: int = 2048,
    ):
        self.model = model
        self.client = client or WorkersAIClient()
        self.temperature = temperature
        self.max_tokens = max_tokens

    def generate(self, prompt: str, json_schema: dict[str, Any]) -> dict[str, Any]:
        payload = build_chat_payload(
            self.model, prompt, json_schema, self.temperature, self.max_tokens
        )
        return parse_chat_response(self.client.chat_completions(payload))


class ClefDecisionModel:
    """Cloudflare Clef / Clef-flash on Workers AI.

    Clef reads a ``state`` and a map of typed questions (``noul``, ``choice``,
    ``score``) and returns a probability for every allowed answer.
    """

    max_questions_per_request = 64
    max_state_chars = None  # 27B / 9B models with a large context

    def __init__(
        self, model: str = DEFAULT_CLEF_MODEL, client: WorkersAIClient | None = None
    ):
        self.model = model
        self.client = client or WorkersAIClient()

    def decide(
        self, state: str | dict[str, Any], questions: dict[str, dict[str, Any]]
    ) -> dict[str, dict[str, Any]]:
        if not questions:
            return {}
        result = self.client.run(self.model, {"state": state, "questions": questions})
        return extract_answers(result, set(questions))
