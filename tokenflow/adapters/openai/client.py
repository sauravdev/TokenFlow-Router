"""OpenAI / frontier-API backend adapter.

Used for any OpenAI-compatible API endpoint billed per-token rather than
per-GPU-hour: OpenAI directly, Anthropic (via their OpenAI-compat shim),
OpenRouter, xAI Grok, Together, Fireworks, etc. All of them implement
the OpenAI Chat Completions wire format on `POST /v1/chat/completions`.

Probe behaviour
---------------
There's no per-host telemetry to scrape (no /metrics, no GPU cache, no
queue depth). The probe just verifies the endpoint is reachable and
that the API key authenticates by hitting `GET /v1/models`. Latency and
saturation are reported as conservative synthetic estimates so the
scoring engine has a usable signal alongside the local-GPU backends.

Cost accounting
---------------
The router's scoring engine assumes USD-per-GPU-hour. For frontier
APIs we record per-1k-token pricing in `capability_flags` and convert
to an effective hourly rate at registration time, so cost-aware
routing still works:

    cost_per_gpu_hour = (avg_input_tps * price_in / 1k) +
                       (avg_output_tps * price_out / 1k) * 3600

For most planning purposes you can register with a representative
hourly figure (e.g. gpt-4o-mini at ~$3/hr equivalent for chat-shape
traffic) and adjust if your traffic mix shifts.
"""
from __future__ import annotations

import time
from typing import Any, Optional

import httpx
import structlog

from tokenflow.models import EndpointHealth, EndpointProfile, TelemetryUpdate

logger = structlog.get_logger(__name__)


class OpenAIClient:
    """Async client for OpenAI-compatible frontier-API endpoints."""

    def __init__(self, timeout: float = 5.0) -> None:
        self._client = httpx.AsyncClient(timeout=timeout)

    async def close(self) -> None:
        await self._client.aclose()

    # ── auth helpers ─────────────────────────────────────────────────

    @staticmethod
    def _auth_headers(ep: EndpointProfile) -> dict[str, str]:
        """Build the auth header for OpenAI-compatible endpoints."""
        headers: dict[str, str] = {}
        if ep.api_key:
            headers["Authorization"] = f"Bearer {ep.api_key}"
        # Some providers (e.g. Anthropic via the openai-compat layer) accept
        # the key in `x-api-key` instead. Sending both is harmless when only
        # one is required and lets a single config work across providers.
        if ep.api_key:
            headers["x-api-key"] = ep.api_key
        return headers

    # ── probes ───────────────────────────────────────────────────────

    async def is_ready(self, ep: EndpointProfile) -> bool:
        """GET /v1/models with auth header. 200 = the API is reachable
        and the API key authenticates. 401/403 = auth problem. Anything
        else = treat as unhealthy."""
        try:
            resp = await self._client.get(ep.models_url, headers=self._auth_headers(ep))
            return resp.status_code == 200
        except Exception:
            return False

    async def probe(self, ep: EndpointProfile) -> TelemetryUpdate:
        """Full probe. Frontier APIs don't expose telemetry, so we measure
        round-trip time on /v1/models as a coarse latency proxy and report
        zero saturation / zero queue depth."""
        t_start = time.perf_counter()
        ready = await self.is_ready(ep)
        probe_ms = (time.perf_counter() - t_start) * 1000

        if not ready:
            ep.capability_flags["warm"] = False
            ep.capability_flags["frontier"] = True
            return TelemetryUpdate(
                endpoint_id=ep.id, error_rate=1.0, saturation_score=1.0,
            )

        ep.capability_flags["warm"] = True
        ep.capability_flags["frontier"] = True
        ep.capability_flags["api_authenticated"] = True

        # Conservative synthetic estimates. p95 ≈ 2× round-trip until we have
        # real upstream measurements (the gateway feeds back actual latency
        # via state.policy_engine.record_actual_latency, so over time the
        # scoring engine will use real numbers).
        return TelemetryUpdate(
            endpoint_id=ep.id,
            p50_ttft_ms=probe_ms,
            p95_ttft_ms=probe_ms * 2.0,
            p50_e2e_ms=probe_ms * 3.0,        # frontier APIs typically take ~1-3s end-to-end
            p95_e2e_ms=probe_ms * 5.0,
            error_rate=0.0,
            saturation_score=0.0,
            queue_depth=0,
            active_requests=0,
        )

    # ── inference forward (used by gateway/proxy.py) ─────────────────
    # Note: the gateway proxy already handles forward + streaming for any
    # OpenAI-compatible endpoint. The OpenAIClient only needs to provide
    # auth headers — the proxy now consults `endpoint.api_key` and uses
    # this helper to attach Bearer auth uniformly.

    @staticmethod
    def auth_headers_for(ep: EndpointProfile) -> dict[str, str]:
        """Public helper used by gateway/proxy.py at request time."""
        return OpenAIClient._auth_headers(ep)
