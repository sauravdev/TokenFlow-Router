"""
Observability — structured logging, Prometheus metrics, and request trace store.

Per-request traces are kept in a bounded ring buffer for the /explain API.
Prometheus metrics track routing decisions, latencies, and costs.
"""

from __future__ import annotations

import time
from collections import deque
from typing import Optional

import structlog
from prometheus_client import (
    Counter,
    Gauge,
    Histogram,
    generate_latest,
    CONTENT_TYPE_LATEST,
)

from tokenflow.models import ExplainResponse, RequestProfile, RouteDecision, RoutingPolicy

logger = structlog.get_logger(__name__)

# ---------------------------------------------------------------------------
# Prometheus metrics
# ---------------------------------------------------------------------------

ROUTE_DECISIONS_TOTAL = Counter(
    "tokenflow_route_decisions_total",
    "Total routing decisions",
    ["outcome", "endpoint_name", "workload_type", "priority_tier"],
)

ROUTE_DECISION_LATENCY = Histogram(
    "tokenflow_route_decision_latency_ms",
    "Time spent making a routing decision (ms)",
    buckets=[1, 2, 5, 10, 20, 50, 100],
)

UPSTREAM_REQUEST_LATENCY = Histogram(
    "tokenflow_upstream_request_latency_ms",
    "End-to-end upstream request latency (ms)",
    ["endpoint_name"],
    buckets=[100, 250, 500, 1000, 2000, 5000, 10000, 30000],
)

UPSTREAM_TTFT = Histogram(
    "tokenflow_upstream_ttft_ms",
    "Time to first token from upstream (ms)",
    ["endpoint_name"],
    buckets=[50, 100, 200, 500, 1000, 2000, 5000],
)

ROUTE_COST_USD = Counter(
    "tokenflow_estimated_cost_usd_total",
    "Cumulative estimated routing cost in USD",
    ["tenant_id", "endpoint_name"],
)

FALLBACK_TOTAL = Counter(
    "tokenflow_fallback_total",
    "Number of fallback routing events",
    ["reason"],
)

ACTIVE_REQUESTS = Gauge(
    "tokenflow_active_requests",
    "Currently in-flight upstream requests",
    ["endpoint_name"],
)

ENDPOINT_HEALTH = Gauge(
    "tokenflow_endpoint_health",
    "Endpoint health (1=healthy, 0.5=degraded, 0=unhealthy)",
    ["endpoint_id", "endpoint_name"],
)


def get_metrics_output() -> tuple[bytes, str]:
    return generate_latest(), CONTENT_TYPE_LATEST


# ---------------------------------------------------------------------------
# Request trace store
# ---------------------------------------------------------------------------

_TRACE_BUFFER_SIZE = 10_000


class TraceStore:
    """
    Bounded ring buffer of per-request routing traces.
    Used by the /explain API.
    """

    def __init__(self, maxsize: int = _TRACE_BUFFER_SIZE) -> None:
        self._buffer: deque[ExplainResponse] = deque(maxlen=maxsize)
        self._index: dict[str, ExplainResponse] = {}

    def record(
        self,
        profile: RequestProfile,
        decision: RouteDecision,
        policy_name: str,
    ) -> ExplainResponse:
        trace = ExplainResponse(
            request_id=profile.request_id,
            decision=decision,
            request_profile=profile,
            policy_name=policy_name,
        )
        # Evict oldest from index if buffer wraps
        if len(self._buffer) == self._buffer.maxlen:
            oldest = self._buffer[0]
            self._index.pop(oldest.request_id, None)

        self._buffer.append(trace)
        self._index[profile.request_id] = trace

        # Prometheus instrumentation
        endpoint_name = decision.selected_endpoint_name or "none"
        ROUTE_DECISIONS_TOTAL.labels(
            outcome=decision.outcome.value,
            endpoint_name=endpoint_name,
            workload_type=profile.workload_type.value,
            priority_tier=profile.priority_tier.value,
        ).inc()
        ROUTE_DECISION_LATENCY.observe(decision.decision_latency_ms)

        if decision.estimated_cost_usd > 0:
            ROUTE_COST_USD.labels(
                tenant_id=profile.tenant_id,
                endpoint_name=endpoint_name,
            ).inc(decision.estimated_cost_usd)

        if decision.fallback_used:
            FALLBACK_TOTAL.labels(reason=decision.outcome.value).inc()

        return trace

    def get(self, request_id: str) -> Optional[ExplainResponse]:
        return self._index.get(request_id)

    def recent(self, limit: int = 100) -> list[ExplainResponse]:
        items = list(self._buffer)
        return items[-limit:]

    def record_actual_latency(
        self,
        request_id: str,
        endpoint_name: str,
        ttft_ms: Optional[float],
        e2e_ms: float,
    ) -> None:
        trace = self._index.get(request_id)
        if trace:
            trace.decision.actual_ttft_ms = ttft_ms
            trace.decision.actual_e2e_ms = e2e_ms

        UPSTREAM_REQUEST_LATENCY.labels(endpoint_name=endpoint_name).observe(e2e_ms)
        if ttft_ms is not None:
            UPSTREAM_TTFT.labels(endpoint_name=endpoint_name).observe(ttft_ms)
