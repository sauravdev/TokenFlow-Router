"""
Upstream proxy — forwards requests to the selected NIM endpoint,
handles streaming (SSE) and non-streaming responses, and measures
TTFT + E2E latency.
"""

from __future__ import annotations

import time
from typing import Any, AsyncIterator, Optional

import httpx
import structlog

from tokenflow.config import settings
from tokenflow.models import EndpointProfile

logger = structlog.get_logger(__name__)


class UpstreamProxy:
    """
    Async HTTP client wrapper for forwarding requests to NIM endpoints.

    - Non-streaming: buffers full response, returns JSON dict
    - Streaming: yields raw SSE chunks, measures TTFT on first chunk
    """

    def __init__(self) -> None:
        self._client = httpx.AsyncClient(
            timeout=httpx.Timeout(
                connect=settings.upstream_connect_timeout_s,
                read=settings.upstream_timeout_s,
                write=30.0,
                pool=5.0,
            ),
            limits=httpx.Limits(max_connections=200, max_keepalive_connections=50),
        )

    async def close(self) -> None:
        await self._client.aclose()

    async def forward(
        self,
        endpoint: EndpointProfile,
        body: dict[str, Any],
        headers: dict[str, str] | None = None,
    ) -> dict[str, Any]:
        """Forward a non-streaming request. Returns the NIM JSON response."""
        url = endpoint.completions_url
        req_headers = {"Content-Type": "application/json"}
        if headers:
            # Pass through Authorization header if present
            for k in ("Authorization", "authorization", "x-api-key"):
                if k in headers:
                    req_headers[k] = headers[k]

        try:
            response = await self._client.post(url, json=body, headers=req_headers)
            response.raise_for_status()
            return response.json()
        except httpx.HTTPStatusError as exc:
            logger.warning(
                "upstream_http_error",
                endpoint=endpoint.name,
                status=exc.response.status_code,
                detail=exc.response.text[:200],
            )
            raise
        except httpx.RequestError as exc:
            logger.warning(
                "upstream_request_error",
                endpoint=endpoint.name,
                error=str(exc),
            )
            raise

    async def forward_streaming(
        self,
        endpoint: EndpointProfile,
        body: dict[str, Any],
        headers: dict[str, str] | None = None,
    ) -> AsyncIterator[tuple[bytes, Optional[float]]]:
        """
        Forward a streaming request.

        Yields (chunk_bytes, ttft_ms_or_None):
          - ttft_ms is set only on the first data chunk
          - subsequent chunks have ttft=None
        """
        url = endpoint.completions_url
        req_headers = {"Content-Type": "application/json"}
        if headers:
            for k in ("Authorization", "authorization", "x-api-key"):
                if k in headers:
                    req_headers[k] = headers[k]

        body = {**body, "stream": True}
        t_start = time.perf_counter()
        first_chunk = True

        try:
            async with self._client.stream(
                "POST", url, json=body, headers=req_headers
            ) as response:
                response.raise_for_status()
                async for chunk in response.aiter_bytes():
                    if chunk:
                        if first_chunk:
                            ttft_ms = (time.perf_counter() - t_start) * 1000
                            first_chunk = False
                            yield chunk, ttft_ms
                        else:
                            yield chunk, None
        except httpx.HTTPStatusError as exc:
            logger.warning(
                "upstream_stream_http_error",
                endpoint=endpoint.name,
                status=exc.response.status_code,
            )
            raise
        except httpx.RequestError as exc:
            logger.warning(
                "upstream_stream_request_error",
                endpoint=endpoint.name,
                error=str(exc),
            )
            raise
