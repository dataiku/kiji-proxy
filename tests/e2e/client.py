"""Async HTTP client for the e2e harness.

Bounds in-flight requests with a semaphore to stay under the backend's
hardcoded rate limit (10 RPS + burst 20 in src/backend/server/server.go).
"""

from __future__ import annotations

import asyncio
import time
from typing import Any

import httpx


class ThrottledClient:
    def __init__(
        self,
        base_url: str,
        *,
        concurrency: int = 10,
        timeout: float = 60.0,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self._client = httpx.AsyncClient(timeout=timeout)
        self._sem = asyncio.Semaphore(concurrency)

    async def post_json(
        self,
        path: str,
        json_body: dict[str, Any],
        headers: dict[str, str] | None = None,
    ) -> tuple[httpx.Response | None, float]:
        async with self._sem:
            start = time.perf_counter()
            try:
                response = await self._client.post(
                    f"{self.base_url}{path}",
                    json=json_body,
                    headers=headers,
                )
                elapsed = (time.perf_counter() - start) * 1000
                return response, elapsed
            except httpx.HTTPError:
                elapsed = (time.perf_counter() - start) * 1000
                return None, elapsed

    async def get(self, path: str) -> httpx.Response | None:
        async with self._sem:
            try:
                return await self._client.get(f"{self.base_url}{path}")
            except httpx.HTTPError:
                return None

    async def close(self) -> None:
        await self._client.aclose()
