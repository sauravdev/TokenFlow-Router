"""
NVIDIA NIM adapter client.

Handles:
- Health probing (/v1/health/ready, /v1/health/live)
- Metrics scraping (Prometheus /metrics endpoint if available)
- Model metadata retrieval (/v1/models)
- Telemetry extraction from NIM responses

NIM exposes OpenAI-compatible APIs plus a set of NIM-specific
management endpoints. This adapter bridges them to TokenFlow's
internal telemetry model.
"""

from __future__ import annotations

import re
import time
from typing import Any, Optional

import httpx
import structlog

from tokenflow.models import EndpointProfile, TelemetryUpdate

logger = structlog.get_logger(__name__)

# NIM Prometheus metric names we care about
_METRIC_PATTERNS = {
    # vLLM / NIM standard metrics
    "queue_depth": re.compile(
        r'^(?:vllm|nim):num_requests_waiting\s+([\d.]+)', re.MULTILINE
    ),
    "active_requests": re.compile(
        r'^(?:vllm|nim):num_requests_running\s+([\d.]+)', re.MULTILINE
    ),
    "tokens_per_second": re.compile(
        r'^(?:vllm|nim):generation_tokens_per_second\s+([\d.]+)', re.MULTILINE
    ),
    "p50_ttft_ms": re.compile(
        r'^(?:vllm|nim):time_to_first_token_seconds_bucket.*le="0\.5"\}\s+([\d.]+)',
        re.MULTILINE,
    ),
    "p95_e2e_ms": re.compile(
        r'^(?:vllm|nim):e2e_request_latency_seconds_bucket.*le="5"\}\s+([\d.]+)',
        re.MULTILINE,
    ),
}


class NIMClient:
    """
    Lightweight async client for NIM endpoint management operations.
    Not used for inference — only for health + telemetry.
    """

    def __init__(self, timeout: float = 5.0) -> None:
        self._client = httpx.AsyncClient(timeout=timeout)

    async def close(self) -> None:
        await self._client.aclose()

    # ------------------------------------------------------------------
    # Health
    # ------------------------------------------------------------------

    async def is_ready(self, nim_url: str) -> bool:
        """Check /v1/health/ready — returns True if 200 OK."""
        try:
            resp = await self._client.get(f"{nim_url}/v1/health/ready")
            return resp.status_code == 200
        except Exception:
            return False

    async def is_live(self, nim_url: str) -> bool:
        """Check /v1/health/live — liveness probe."""
        try:
            resp = await self._client.get(f"{nim_url}/v1/health/live")
            return resp.status_code == 200
        except Exception:
            return False

    # ------------------------------------------------------------------
    # Model metadata
    # ------------------------------------------------------------------

    async def list_models(self, nim_url: str) -> list[dict[str, Any]]:
        """Fetch /v1/models and return the list of available models."""
        try:
            resp = await self._client.get(f"{nim_url}/v1/models")
            resp.raise_for_status()
            data = resp.json()
            return data.get("data", [])
        except Exception as exc:
            logger.warning("nim_list_models_failed", nim_url=nim_url, error=str(exc))
            return []

    # ------------------------------------------------------------------
    # Prometheus metrics scraping
    # ------------------------------------------------------------------

    async def scrape_metrics(self, nim_url: str) -> Optional[TelemetryUpdate]:
        """
        Attempt to scrape Prometheus metrics from /metrics.
        Returns a partial TelemetryUpdate or None if unavailable.
        """
        try:
            resp = await self._client.get(f"{nim_url}/metrics")
            if resp.status_code != 200:
                return None
            text = resp.text
            return self._parse_prometheus(text)
        except Exception:
            return None

    def _parse_prometheus(self, text: str) -> TelemetryUpdate:
        """Parse subset of Prometheus exposition format into TelemetryUpdate."""
        values: dict[str, float] = {}
        for field, pattern in _METRIC_PATTERNS.items():
            match = pattern.search(text)
            if match:
                try:
                    values[field] = float(match.group(1))
                except ValueError:
                    pass

        # Convert seconds → milliseconds for latency fields
        for field in ("p50_ttft_ms", "p95_e2e_ms"):
            if field in values:
                values[field] = values[field] * 1000

        return TelemetryUpdate(endpoint_id="", **values)

    # ------------------------------------------------------------------
    # Combined probe: health + optional metrics
    # ------------------------------------------------------------------

    async def probe(self, ep: EndpointProfile) -> TelemetryUpdate:
        """
        Full probe: combines health check timing with metrics scraping.
        Falls back to timing-based estimates if /metrics unavailable.
        """
        t_start = time.perf_counter()
        ready = await self.is_ready(ep.nim_url)
        probe_ms = (time.perf_counter() - t_start) * 1000

        # Try Prometheus metrics first
        metrics_update = await self.scrape_metrics(ep.nim_url)

        if metrics_update is not None:
            metrics_update.endpoint_id = ep.id
            # Fill in error_rate from health
            if not ready:
                metrics_update.error_rate = 0.5
            return metrics_update

        # Fall back to probe timing
        if ready:
            return TelemetryUpdate(
                endpoint_id=ep.id,
                p50_ttft_ms=probe_ms,
                p95_ttft_ms=probe_ms * 1.5,
                p50_e2e_ms=probe_ms,
                p95_e2e_ms=probe_ms * 2.0,
                error_rate=0.0,
                saturation_score=0.0,
            )
        else:
            return TelemetryUpdate(
                endpoint_id=ep.id,
                error_rate=1.0,
                saturation_score=1.0,
            )
