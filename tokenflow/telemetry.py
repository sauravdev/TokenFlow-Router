"""Telemetry store and background collector for endpoint metrics."""

from __future__ import annotations

import asyncio
import time
from collections import defaultdict, deque
from datetime import datetime, timezone
from typing import Optional

import httpx
import structlog

from tokenflow.config import settings
from tokenflow.models import EndpointHealth, EndpointProfile, EndpointTelemetry, TelemetryUpdate

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
    Background task that polls NIM endpoints for live metrics.

    NIM exposes a /v1/health/ready endpoint and optionally a Prometheus
    /metrics path. We use a lightweight health probe and infer queue
    pressure from response timing.
    """

    def __init__(self, store: TelemetryStore) -> None:
        self._store = store
        self._endpoints: list[EndpointProfile] = []
        self._task: Optional[asyncio.Task] = None
        self._client: Optional[httpx.AsyncClient] = None

    def register_endpoint(self, ep: EndpointProfile) -> None:
        if not any(e.id == ep.id for e in self._endpoints):
            self._endpoints.append(ep)

    def unregister_endpoint(self, endpoint_id: str) -> None:
        self._endpoints = [e for e in self._endpoints if e.id != endpoint_id]

    async def start(self) -> None:
        self._client = httpx.AsyncClient(timeout=5.0)
        self._task = asyncio.create_task(self._loop(), name="telemetry_collector")
        logger.info("telemetry_collector_started", interval_s=settings.telemetry_scrape_interval_s)

    async def stop(self) -> None:
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
        if self._client:
            await self._client.aclose()
        logger.info("telemetry_collector_stopped")

    async def _loop(self) -> None:
        while True:
            await asyncio.sleep(settings.telemetry_scrape_interval_s)
            tasks = [self._scrape(ep) for ep in self._endpoints]
            if tasks:
                await asyncio.gather(*tasks, return_exceptions=True)

    async def _scrape(self, ep: EndpointProfile) -> None:
        """Scrape a single endpoint. Uses timing + health probe as proxy metrics."""
        assert self._client is not None
        start = time.perf_counter()
        health = EndpointHealth.UNKNOWN
        try:
            resp = await self._client.get(f"{ep.nim_url}/v1/health/ready")
            elapsed_ms = (time.perf_counter() - start) * 1000
            if resp.status_code == 200:
                health = EndpointHealth.HEALTHY
                # Use probe latency as TTFT proxy if no richer metrics
                update = TelemetryUpdate(
                    endpoint_id=ep.id,
                    p50_ttft_ms=elapsed_ms,
                    p95_ttft_ms=elapsed_ms * 1.5,
                    p50_e2e_ms=elapsed_ms,
                    p95_e2e_ms=elapsed_ms * 2.0,
                    error_rate=0.0,
                )
            else:
                health = EndpointHealth.DEGRADED
                update = TelemetryUpdate(
                    endpoint_id=ep.id,
                    error_rate=0.5,
                    saturation_score=0.8,
                )
            await self._store.upsert(update)
        except Exception as exc:
            health = EndpointHealth.UNHEALTHY
            logger.warning("scrape_failed", endpoint_id=ep.id, error=str(exc))
            await self._store.upsert(
                TelemetryUpdate(endpoint_id=ep.id, error_rate=1.0, saturation_score=1.0)
            )

        # Registry health update is done via the app's registry reference
        # (injected at startup)
        ep.health = health
