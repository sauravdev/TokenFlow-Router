"""Telemetry store and background collector for endpoint metrics."""

from __future__ import annotations

import asyncio
import time
from collections import defaultdict, deque
from datetime import datetime, timezone
from typing import Optional

import httpx
import structlog

from tokenflow.adapters.dynamo.client import DynamoClient
from tokenflow.adapters.nim.client import NIMClient
from tokenflow.adapters.ollama.client import OllamaClient
from tokenflow.adapters.openai.client import OpenAIClient
from tokenflow.adapters.sglang.client import SGLangClient
from tokenflow.adapters.vllm.client import VLLMClient
from tokenflow.config import settings
from tokenflow.models import BackendType, EndpointHealth, EndpointProfile, EndpointTelemetry, TelemetryUpdate

logger = structlog.get_logger(__name__)

# Number of historical snapshots to retain per endpoint
_HISTORY_SIZE = 60


class TelemetryStore:
    """
    In-memory store for endpoint telemetry with EMA smoothing.

    Scrapes /metrics or /v1/health from each NIM endpoint on a background loop.
    Also accepts pushed updates from NIM sidecars via the admin API.
    """

    def __init__(self) -> None:
        self._current: dict[str, EndpointTelemetry] = {}
        self._history: dict[str, deque[EndpointTelemetry]] = defaultdict(
            lambda: deque(maxlen=_HISTORY_SIZE)
        )
        self._lock = asyncio.Lock()
        self._alpha = settings.telemetry_smoothing_alpha
        self._stale_threshold_s = settings.telemetry_stale_threshold_s

    # ------------------------------------------------------------------
    # Write
    # ------------------------------------------------------------------

    async def upsert(self, update: TelemetryUpdate) -> EndpointTelemetry:
        """Apply a telemetry update (push path), with EMA smoothing."""
        async with self._lock:
            existing = self._current.get(update.endpoint_id)
            if existing is None:
                # First data point — accept as-is
                new_tel = EndpointTelemetry(
                    endpoint_id=update.endpoint_id,
                    **{k: v for k, v in update.model_dump().items() if v is not None and k != "endpoint_id"},
                )
            else:
                new_tel = self._apply_ema(existing, update)

            self._current[update.endpoint_id] = new_tel
            self._history[update.endpoint_id].append(new_tel)
            return new_tel

    def _apply_ema(
        self, existing: EndpointTelemetry, update: TelemetryUpdate
    ) -> EndpointTelemetry:
        """Exponential moving average smoothing for numeric metrics."""
        a = self._alpha

        def ema(old: float, new: Optional[float]) -> float:
            if new is None:
                return old
            return a * new + (1 - a) * old

        return EndpointTelemetry(
            endpoint_id=existing.endpoint_id,
            timestamp=datetime.now(timezone.utc),
            rpm=ema(existing.rpm, update.rpm),
            rph=ema(existing.rph, update.rph),
            queue_depth=int(ema(existing.queue_depth, update.queue_depth)),
            active_requests=int(ema(existing.active_requests, update.active_requests)),
            tokens_per_second=ema(existing.tokens_per_second, update.tokens_per_second),
            p50_ttft_ms=ema(existing.p50_ttft_ms, update.p50_ttft_ms),
            p95_ttft_ms=ema(existing.p95_ttft_ms, update.p95_ttft_ms),
            p50_itl_ms=ema(existing.p50_itl_ms, update.p50_itl_ms),
            p95_itl_ms=ema(existing.p95_itl_ms, update.p95_itl_ms),
            p50_e2e_ms=ema(existing.p50_e2e_ms, update.p50_e2e_ms),
            p95_e2e_ms=ema(existing.p95_e2e_ms, update.p95_e2e_ms),
            error_rate=ema(existing.error_rate, update.error_rate),
            saturation_score=ema(existing.saturation_score, update.saturation_score),
        )

    # ------------------------------------------------------------------
    # Read
    # ------------------------------------------------------------------

    def get(self, endpoint_id: str) -> Optional[EndpointTelemetry]:
        return self._current.get(endpoint_id)

    def get_history(self, endpoint_id: str) -> list[EndpointTelemetry]:
        return list(self._history.get(endpoint_id, []))

    def is_stale(self, endpoint_id: str) -> bool:
        tel = self._current.get(endpoint_id)
        if tel is None:
            return True
        now = datetime.now(timezone.utc)
        ts = tel.timestamp
        if ts.tzinfo is None:
            ts = ts.replace(tzinfo=timezone.utc)
        return (now - ts).total_seconds() > self._stale_threshold_s

    def all_current(self) -> dict[str, EndpointTelemetry]:
        return dict(self._current)

    # ------------------------------------------------------------------
    # Derived health scoring
    # ------------------------------------------------------------------

    def compute_health(self, endpoint_id: str) -> EndpointHealth:
        """Derive health state from telemetry."""
        if self.is_stale(endpoint_id):
            return EndpointHealth.UNKNOWN
        tel = self._current[endpoint_id]
        if tel.error_rate > 0.20 or tel.saturation_score > 0.95:
            return EndpointHealth.UNHEALTHY
        if tel.error_rate > 0.05 or tel.saturation_score > 0.80:
            return EndpointHealth.DEGRADED
        return EndpointHealth.HEALTHY


class TelemetryCollector:
    """
    Background task that polls inference endpoints for live metrics.

    Dispatches to the appropriate backend-specific adapter based on
    ep.backend_type so each serving stack is probed with its native
    health + metrics endpoints:

      NIM    → /v1/health/ready + /metrics (Prometheus, vllm:/nim: prefix)
      vLLM   → /health + /metrics (Prometheus, vllm: prefix)
      SGLang → /health_generate + /get_server_info (JSON, rich cache stats)
      Dynamo → /health + /metrics (Prometheus, vllm: + dynamo: prefix)
    """

    def __init__(self, store: TelemetryStore) -> None:
        self._store = store
        self._endpoints: list[EndpointProfile] = []
        self._task: Optional[asyncio.Task] = None
        # Per-backend clients — created lazily at start()
        self._nim_client: Optional[NIMClient] = None
        self._vllm_client: Optional[VLLMClient] = None
        self._sglang_client: Optional[SGLangClient] = None
        self._dynamo_client: Optional[DynamoClient] = None
        self._ollama_client: Optional[OllamaClient] = None
        self._openai_client: Optional[OpenAIClient] = None

    def register_endpoint(self, ep: EndpointProfile) -> None:
        if not any(e.id == ep.id for e in self._endpoints):
            self._endpoints.append(ep)

    def unregister_endpoint(self, endpoint_id: str) -> None:
        self._endpoints = [e for e in self._endpoints if e.id != endpoint_id]

    async def start(self) -> None:
        self._nim_client = NIMClient(timeout=5.0)
        self._vllm_client = VLLMClient(timeout=5.0)
        self._sglang_client = SGLangClient(timeout=8.0)  # health_generate can be slower
        self._dynamo_client = DynamoClient(timeout=5.0)
        self._ollama_client = OllamaClient(timeout=5.0)
        self._openai_client = OpenAIClient(timeout=5.0)  # frontier-API endpoints
        self._task = asyncio.create_task(self._loop(), name="telemetry_collector")
        logger.info("telemetry_collector_started", interval_s=settings.telemetry_scrape_interval_s)

    async def stop(self) -> None:
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
        for client in (self._nim_client, self._vllm_client, self._sglang_client, self._dynamo_client, self._ollama_client, self._openai_client):
            if client:
                await client.close()
        logger.info("telemetry_collector_stopped")

    async def _loop(self) -> None:
        while True:
            await asyncio.sleep(settings.telemetry_scrape_interval_s)
            tasks = [self._scrape(ep) for ep in self._endpoints]
            if tasks:
                await asyncio.gather(*tasks, return_exceptions=True)

    async def _scrape(self, ep: EndpointProfile) -> None:
        """
        Scrape a single endpoint using its backend-specific adapter.

        Each adapter returns a TelemetryUpdate that may contain rich metrics
        (Prometheus histograms, cache hit rates, queue depths) or fall back to
        probe-timing estimates when native metrics are unavailable.

        Warmup grace: during the first `warmup_grace_seconds` after an
        endpoint is registered (e.g. via dormant-profile auto-activation),
        probe failures do not mark the endpoint UNHEALTHY. This prevents a
        container that is still booting from being locked out of routing
        just because its /health is temporarily unreachable.
        """
        health = EndpointHealth.UNKNOWN

        # Compute endpoint age up front — backend adapters don't raise on
        # probe failure; they RETURN TelemetryUpdate(error_rate=1.0). So we
        # check for "effectively-failed" probes after the fact and still
        # apply warmup grace.
        registered_at = ep.registered_at
        if registered_at.tzinfo is None:
            registered_at = registered_at.replace(tzinfo=timezone.utc)
        age_s = (datetime.now(timezone.utc) - registered_at).total_seconds()
        in_warmup = age_s < settings.endpoint_warmup_grace_s

        try:
            update = await self._probe_by_backend(ep)

            # Treat error_rate >= 0.5 as a probe failure. During warmup,
            # skip the upsert so the endpoint stays in its current (likely
            # UNKNOWN) state and remains routable.
            if in_warmup and (update.error_rate or 0.0) >= 0.5:
                logger.info(
                    "probe_failed_during_warmup",
                    endpoint=ep.name,
                    age_s=round(age_s, 1),
                    grace_s=settings.endpoint_warmup_grace_s,
                    error_rate=update.error_rate,
                )
                return

            await self._store.upsert(update)

            # Derive health from telemetry
            health = self._store.compute_health(ep.id)
            logger.debug(
                "endpoint_scraped",
                endpoint=ep.name,
                backend=ep.backend_type,
                health=health,
                error_rate=update.error_rate,
            )
        except Exception as exc:
            # Same warmup-grace logic for the exception path (connection
            # refused, timeout, etc.).
            if in_warmup:
                logger.info(
                    "scrape_during_warmup",
                    endpoint=ep.name,
                    age_s=round(age_s, 1),
                    grace_s=settings.endpoint_warmup_grace_s,
                    error=str(exc),
                )
                return

            health = EndpointHealth.UNHEALTHY
            logger.warning("scrape_failed", endpoint_id=ep.id, backend=ep.backend_type, error=str(exc))
            await self._store.upsert(
                TelemetryUpdate(endpoint_id=ep.id, error_rate=1.0, saturation_score=1.0)
            )

        ep.health = health

    async def _probe_by_backend(self, ep: EndpointProfile) -> TelemetryUpdate:
        """Dispatch to the correct backend adapter."""
        if ep.backend_type == BackendType.VLLM:
            assert self._vllm_client is not None
            return await self._vllm_client.probe(ep)

        if ep.backend_type == BackendType.SGLANG:
            assert self._sglang_client is not None
            return await self._sglang_client.probe(ep)

        if ep.backend_type == BackendType.DYNAMO:
            assert self._dynamo_client is not None
            return await self._dynamo_client.probe(ep)

        if ep.backend_type == BackendType.OLLAMA:
            assert self._ollama_client is not None
            return await self._ollama_client.probe(ep)

        if ep.backend_type == BackendType.OPENAI:
            assert self._openai_client is not None
            return await self._openai_client.probe(ep)

        # Default: NIM (also handles UNKNOWN backend_type gracefully)
        assert self._nim_client is not None
        return await self._nim_client.probe(ep)
