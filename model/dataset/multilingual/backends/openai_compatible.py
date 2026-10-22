"""Generators for OpenAI-compatible chat APIs: local Qwen (vLLM, SGLang) and Doubleword."""

import os
from typing import Any

import httpx

from .chat import build_chat_payload, parse_chat_response
from .http_utils import make_http_client, post_json

DEFAULT_LOCAL_MODEL = "Qwen/Qwen3.8-27B"
DEFAULT_LOCAL_URL = "http://localhost:8000/v1/chat/completions"
DOUBLEWORD_URL = "https://api.doubleword.ai/v1/chat/completions"
DEFAULT_DOUBLEWORD_MODEL = "Qwen/Qwen3.8-27B"


class OpenAICompatibleGenerator:
    """Any server speaking the OpenAI chat completions API.

    Thinking is disabled through ``chat_template_kwargs`` (honoured by vLLM
    and SGLang; other servers ignore the field).
    """

    def __init__(
        self,
        model: str = DEFAULT_LOCAL_MODEL,
        api_url: str = DEFAULT_LOCAL_URL,
        api_key: str | None = None,
        temperature: float = 0.8,
        max_tokens: int = 2048,
        extra_body: dict[str, Any] | None = None,
        disable_thinking: bool = True,
        timeout: float = 300.0,
        max_retries: int = 3,
        transport: httpx.BaseTransport | None = None,
    ):
        self.model = model
        self.api_url = api_url
        self.api_key = (
            api_key or os.getenv("QWEN_API_KEY") or os.getenv("OPENAI_API_KEY")
        )
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.extra_body = dict(extra_body or {})
        if disable_thinking:
            self.extra_body.setdefault(
                "chat_template_kwargs", {"enable_thinking": False}
            )
        self.max_retries = max_retries
        self._http = make_http_client(timeout, transport)

    @property
    def headers(self) -> dict[str, str]:
        return {"Authorization": f"Bearer {self.api_key}"} if self.api_key else {}

    def build_payload(self, prompt: str, json_schema: dict[str, Any]) -> dict[str, Any]:
        return build_chat_payload(
            self.model,
            prompt,
            json_schema,
            self.temperature,
            self.max_tokens,
            self.extra_body,
        )

    def generate(self, prompt: str, json_schema: dict[str, Any]) -> dict[str, Any]:
        body = post_json(
            self._http,
            self.api_url,
            self.build_payload(prompt, json_schema),
            self.headers,
            self.max_retries,
        )
        return parse_chat_response(body)


class DoublewordGenerator(OpenAICompatibleGenerator):
    """Doubleword's OpenAI-compatible realtime endpoint (``service_tier: priority``).

    Uses ``DOUBLEWORD_API_KEY``. The model name is whatever Doubleword lists
    for your account (for example ``Qwen/Qwen3.8-27B``).
    """

    def __init__(
        self,
        model: str = DEFAULT_DOUBLEWORD_MODEL,
        api_url: str | None = None,
        api_key: str | None = None,
        service_tier: str = "priority",
        **kwargs: Any,
    ):
        api_key = api_key or os.getenv("DOUBLEWORD_API_KEY")
        if not api_key:
            raise ValueError(
                "DOUBLEWORD_API_KEY must be set (environment or .env file)."
            )
        extra_body = dict(kwargs.pop("extra_body", None) or {})
        if service_tier:
            extra_body["service_tier"] = service_tier
        super().__init__(
            model=model,
            api_url=api_url or DOUBLEWORD_URL,
            api_key=api_key,
            extra_body=extra_body,
            disable_thinking=kwargs.pop("disable_thinking", False),
            **kwargs,
        )
