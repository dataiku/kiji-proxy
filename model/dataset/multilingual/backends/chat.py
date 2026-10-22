"""Chat-completion payloads and output parsing shared by the LLM generators."""

import json
import re
from typing import Any

SYSTEM_PROMPT = "You are a data generator. Reply with JSON only, no explanation."

_THINK_BLOCK = re.compile(r"<think>.*?</think>", re.DOTALL)
_CODE_FENCE = re.compile(r"```(?:json)?\s*([\s\S]*?)\s*```")


def strip_thinking(content: str) -> str:
    """Remove ``<think>...</think>`` reasoning blocks emitted by Qwen models."""
    cleaned = _THINK_BLOCK.sub("", content)
    # An unterminated think block means the model was cut off before answering.
    if "<think>" in cleaned:
        cleaned = cleaned.split("<think>", 1)[0]
    return cleaned.strip()


def extract_json(content: str) -> Any:
    """Parse JSON from model output that may be wrapped in a code fence or prose."""
    cleaned = strip_thinking(content)
    fenced = _CODE_FENCE.search(cleaned)
    if fenced:
        cleaned = fenced.group(1).strip()
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        pass
    # Fall back to the outermost object or array in the text.
    starts = [i for i in (cleaned.find("{"), cleaned.find("[")) if i != -1]
    if not starts:
        raise ValueError(f"No JSON found in model output: {content[:200]!r}")
    start = min(starts)
    closing = "}" if cleaned[start] == "{" else "]"
    end = cleaned.rfind(closing)
    if end <= start:
        raise ValueError(f"Unterminated JSON in model output: {content[:200]!r}")
    return json.loads(cleaned[start : end + 1])


def build_chat_payload(
    model: str,
    prompt: str,
    json_schema: dict[str, Any],
    temperature: float = 0.8,
    max_tokens: int = 2048,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """OpenAI-style chat completion request with a strict JSON schema."""
    payload: dict[str, Any] = {
        "model": model,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ],
        "temperature": temperature,
        "max_tokens": max_tokens,
        "response_format": {
            "type": "json_schema",
            "json_schema": {
                "name": "pii_sample",
                "strict": True,
                "schema": json_schema,
            },
        },
    }
    if extra:
        payload.update(extra)
    return payload


def parse_chat_response(body: dict[str, Any]) -> dict[str, Any]:
    """Return the JSON object in the first choice of a chat completion."""
    content = body["choices"][0]["message"]["content"]
    parsed = extract_json(content) if isinstance(content, str) else content
    if isinstance(parsed, list):
        if not parsed:
            raise ValueError("Model returned an empty list")
        parsed = parsed[0]
    if not isinstance(parsed, dict):
        raise ValueError(f"Unexpected model output type: {type(parsed)}")
    return parsed
