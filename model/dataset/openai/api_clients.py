"""API client implementations for LLM providers."""

import json
import os
import re
from abc import ABC, abstractmethod
from typing import Any

import httpx


def fix_malformed_unicode(text: str) -> str:
    """
    Fix malformed Unicode escape sequences in text.

    The LLM sometimes outputs \\u0000XX instead of \\u00XX for Unicode characters.
    For example, \\u0000fc instead of \\u00fc for ü.

    Args:
        text: The text possibly containing malformed Unicode escapes

    Returns:
        Text with fixed Unicode escapes
    """
    # Pattern matches \u0000XX where XX are hex digits
    # Replace with \u00XX (remove the extra 00)
    pattern = r"\\u0000([0-9a-fA-F]{2})"
    return re.sub(pattern, r"\\u00\1", text)


class LLMClient(ABC):
    """Abstract base class for LLM API clients."""

    @abstractmethod
    def generate(self, prompt: str, json_schema: dict[str, Any]) -> dict[str, Any]:
        """Generate a response from the LLM."""
        pass

    @abstractmethod
    def review(self, prompt: str, json_schema: dict[str, Any]) -> dict[str, Any]:
        """Review and correct a sample using the LLM."""
        pass


class OpenAIClient(LLMClient):
    """OpenAI API client (also works with OpenAI-compatible APIs)."""

    def __init__(
        self,
        model: str = "gpt-4.1-2025-04-14",
        api_key: str | None = None,
        api_url: str | None = None,
    ):
        """
        Initialize OpenAI client.

        Args:
            model: Model name to use
            api_key: API key (defaults to OPENAI_API_KEY env var)
            api_url: API URL (defaults to OpenAI API, use for compatible APIs)
        """
        self.model = model
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")

        if not api_url and not self.api_key:
            raise ValueError(
                "OPENAI_API_KEY not found in environment variables. Please set it in .env file."
            )

        self.url = api_url or "https://api.openai.com/v1/chat/completions"
        self.timeout = httpx.Timeout(300.0, connect=10.0)
        self.headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        self._is_responses_api = "v1/responses" in self.url

    def _make_request(
        self, prompt: str, json_schema: dict[str, Any], task_type: str = "generate"
    ) -> dict[str, Any]:
        """Make a request to OpenAI API."""
        if self._is_responses_api:
            return self._make_responses_request(prompt, json_schema, task_type)
        else:
            return self._make_chat_request(prompt, json_schema, task_type)

    def _make_responses_request(
        self, prompt: str, json_schema: dict[str, Any], task_type: str = "generate"
    ) -> dict[str, Any]:
        """Make a request to OpenAI Responses API."""
        # OpenAI requires root type to be "object", so wrap the array in an object
        if json_schema.get("type") == "array":
            openai_schema = {
                "type": "object",
                "properties": {"samples": json_schema},
                "required": ["samples"],
                "additionalProperties": False,
            }
        else:
            openai_schema = json_schema

        payload = {
            "model": self.model,
            "input": [{"role": "user", "content": prompt}],
            "stream": False,
            "text": {
                "format": {
                    "type": "json_schema",
                    "name": f"{task_type}_sample",
                    "schema": openai_schema,
                    "strict": True,
                },
            },
        }

        if task_type == "review":
            payload["temperature"] = 0.3
            payload["max_tokens"] = 2000

        response = httpx.post(
            self.url, json=payload, headers=self.headers, timeout=self.timeout
        )

        try:
            response.raise_for_status()
        except httpx.HTTPStatusError as e:
            error_detail = (
                response.text if hasattr(response, "text") else str(response.content)
            )
            print(f"API Error: {e}")
            print(f"Response: {error_detail}")
            raise

        result = response.json()
        content = result["choices"][0]["message"]["content"]

        if isinstance(content, str):
            # Fix malformed Unicode escapes from LLM output
            content = fix_malformed_unicode(content)
            return json.loads(content)
        return content

    def _make_chat_request(
        self, prompt: str, json_schema: dict[str, Any], task_type: str = "generate"
    ) -> dict[str, Any]:
        """Make a request to OpenAI Chat Completions API."""
        # OpenAI requires root type to be "object", so wrap the array in an object
        if json_schema.get("type") == "array":
            openai_schema = {
                "type": "object",
                "properties": {"samples": json_schema},
                "required": ["samples"],
                "additionalProperties": False,
            }
        else:
            openai_schema = json_schema

        payload = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "stream": False,
            "response_format": {
                "type": "json_schema",
                "json_schema": {
                    "name": f"{task_type}_sample",
                    "strict": True,
                    "schema": openai_schema,
                },
            },
        }

        if task_type == "review":
            payload["temperature"] = 0.3
            payload["max_tokens"] = 2000

        response = httpx.post(
            self.url, json=payload, headers=self.headers, timeout=self.timeout
        )

        try:
            response.raise_for_status()
        except httpx.HTTPStatusError as e:
            error_detail = (
                response.text if hasattr(response, "text") else str(response.content)
            )
            print(f"API Error: {e}")
            print(f"Response: {error_detail}")
            raise

        result = response.json()
        content = result["choices"][0]["message"]["content"]

        if isinstance(content, str):
            # Fix malformed Unicode escapes from LLM output
            content = fix_malformed_unicode(content)
            return json.loads(content)
        return content

    def generate(self, prompt: str, json_schema: dict[str, Any]) -> dict[str, Any]:
        """Generate a response from OpenAI."""
        result = self._make_request(prompt, json_schema, "generate")
        # OpenAI wraps array results in a "samples" key
        if isinstance(result, dict) and "samples" in result:
            return result["samples"]
        return result

    def review(self, prompt: str, json_schema: dict[str, Any]) -> dict[str, Any]:
        """Review and correct a sample using OpenAI."""
        return self._make_request(prompt, json_schema, "review")
