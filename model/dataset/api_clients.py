"""API client implementations for LLM providers."""

import json
import os
from abc import ABC, abstractmethod
from typing import Any

import httpx


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
    """OpenAI API client."""

    def __init__(
        self,
        model: str = "gpt-4.1-2025-04-14",
        api_key: str | None = None,
        api_url: str | None = None,
    ):
        self.model = model
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        if not api_url and not self.api_key:
            raise ValueError(
                "OPENAI_API_KEY not found in environment variables. Please set it in .env file."
            )
        if api_url:
            self.url = api_url
        else:
            self.url = "https://api.openai.com/v1/chat/completions"
        self.timeout = httpx.Timeout(300.0, connect=10.0)
        self.headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        self._is_responses_api = False
        if "v1/responses" in self.url:
            self._is_responses_api = True

    def _make_request(
        self, prompt: str, json_schema: dict[str, Any], task_type: str = "generate"
    ) -> dict[str, Any]:
        """Make a request to OpenAI API."""
        if self._is_responses_api:
            return self._make_responses_request(prompt, json_schema, task_type)
        else:
            return self._make_legacy_request(prompt, json_schema, task_type)

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

        # Content might already be a dict if response_format worked, or a JSON string
        if isinstance(content, str):
            return json.loads(content)
        return content

    def _make_legacy_request(
        self, prompt: str, json_schema: dict[str, Any], task_type: str = "generate"
    ) -> dict[str, Any]:
        """Make a request to OpenAI API."""
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

        # Content might already be a dict if response_format worked, or a JSON string
        if isinstance(content, str):
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


class OllamaClient(LLMClient):
    """Ollama API client."""

    def __init__(
        self, model: str = "gpt-oss:20b", base_url: str = "http://localhost:11434"
    ):
        self.model = model
        self.url = f"{base_url}/api/chat"
        self.timeout = httpx.Timeout(300.0, connect=10.0)

    def _make_request(
        self, prompt: str, json_schema: dict[str, Any], task_type: str = "generate"
    ) -> dict[str, Any]:
        """Make a request to Ollama API."""
        payload = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "stream": False,
            "format": json_schema,
        }

        response = httpx.post(self.url, json=payload, timeout=self.timeout)

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
        content = result["message"]["content"]

        if isinstance(content, str):
            return json.loads(content)
        return content

    def generate(self, prompt: str, json_schema: dict[str, Any]) -> dict[str, Any]:
        """Generate a response from Ollama."""
        return self._make_request(prompt, json_schema, "generate")

    def review(self, prompt: str, json_schema: dict[str, Any]) -> dict[str, Any]:
        """Review and correct a sample using Ollama."""
        return self._make_request(prompt, json_schema, "review")
