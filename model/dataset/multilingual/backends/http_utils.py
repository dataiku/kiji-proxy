"""Shared HTTP helpers for the backends."""

import time
from typing import Any

import httpx
from absl import logging

RETRYABLE_STATUS_CODES = {408, 409, 425, 429, 500, 502, 503, 504}


def make_http_client(
    timeout: float = 300.0, transport: httpx.BaseTransport | None = None
) -> httpx.Client:
    """Create an httpx client; ``transport`` lets tests inject a mock."""
    return httpx.Client(
        timeout=httpx.Timeout(timeout, connect=10.0), transport=transport
    )


def post_json(
    client: httpx.Client,
    url: str,
    payload: dict[str, Any],
    headers: dict[str, str] | None = None,
    max_retries: int = 3,
) -> Any:
    """POST JSON and return the decoded body, retrying transient failures."""
    last_error: Exception | None = None
    for attempt in range(1, max_retries + 1):
        try:
            response = client.post(url, json=payload, headers=headers)
            if response.status_code in RETRYABLE_STATUS_CODES:
                raise httpx.HTTPStatusError(
                    f"Retryable status {response.status_code}: {response.text[:300]}",
                    request=response.request,
                    response=response,
                )
            response.raise_for_status()
            return response.json()
        except (httpx.TransportError, httpx.HTTPStatusError) as exc:
            last_error = exc
            status = getattr(getattr(exc, "response", None), "status_code", None)
            if status is not None and status not in RETRYABLE_STATUS_CODES:
                raise
            if attempt == max_retries:
                break
            delay = 2**attempt
            logging.warning(
                f"Request to {url} failed (attempt {attempt}/{max_retries}): {exc}. "
                f"Retrying in {delay}s"
            )
            time.sleep(delay)
    raise RuntimeError(f"Request to {url} failed: {last_error}") from last_error
