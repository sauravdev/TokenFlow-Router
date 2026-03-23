"""
vLLM inference backend adapter.

vLLM (https://github.com/vllm-project/vllm) uses PagedAttention for memory-efficient
KV-cache management and excels at decode-heavy workloads with long outputs.

Health and telemetry endpoints:
- GET /health          → 200 OK when server is ready
- GET /metrics         → Prometheus exposition format
- GET /v1/models       → OpenAI-compatible model list

Key Prometheus metrics (vllm: prefix):
  vllm:num_requests_waiting          — queue depth
  vllm:num_requests_running          — active requests
  vllm:gpu_cache_usage_perc          — KV cache saturation (0–1)
  vllm:cpu_cache_usage_perc          — CPU cache saturation (0–1, swap fallback)
  vllm:time_to_first_token_seconds   — TTFT histogram
  vllm:time_per_output_token_seconds — ITL histogram (inter-token latency)
  vllm:e2e_request_latency_seconds   — end-to-end latency histogram
  vllm:prompt_tokens_total           — cumulative prompt tokens
  vllm:generation_tokens_total       — cumulative generation tokens

Note: vLLM's /metrics is always available; no --enable-metrics flag required.
"""

from __future__ import annotations

import re
import time
from typing import Any, Optional

import httpx
import structlog

from tokenflow.models import EndpointProfile, TelemetryUpdate

logger = structlog.get_logger(__name__)

# vLLM Prometheus metric patterns
# Histograms expose _count, _sum, and _bucket; we use _sum/_count for averages
# where available, and bucket thresholds for p50/p95 approximations.
_VLLM_METRIC_PATTERNS: dict[str, re.Pattern] = {
    "queue_depth": re.compile(
        r'^vllm:num_requests_waiting(?:\{[^}]*\})?\s+([\d.]+)', re.MULTILINE
    ),
    "active_requests": re.compile(
        r'^vllm:num_requests_running(?:\{[^}]*\})?\s+([\d.]+)', re.MULTILINE
    ),
    # KV cache fill fraction — high value = approaching OOM, treat as saturation
    "gpu_cache_usage": re.compile(
        r'^vllm:gpu_cache_usage_perc(?:\{[^}]*\})?\s+([\d.]+)', re.MULTILINE
    ),
    # TTFT: use the _sum/_count pair to derive a running mean (proxy for p50)
    "ttft_sum": re.compile(
        r'^vllm:time_to_first_token_seconds_sum(?:\{[^}]*\})?\s+([\d.]+)', re.MULTILINE
    ),
    "ttft_count": re.compile(
        r'^vllm:time_to_first_token_seconds_count(?:\{[^}]*\})?\s+([\d.]+)', re.MULTILINE
    ),
    # ITL (inter-token latency, a.k.a. time_per_output_token)
    "itl_sum": re.compile(
        r'^vllm:time_per_output_token_seconds_sum(?:\{[^}]*\})?\s+([\d.]+)', re.MULTILINE
    ),
    "itl_count": re.compile(
        r'^vllm:time_per_output_token_seconds_count(?:\{[^}]*\})?\s+([\d.]+)', re.MULTILINE
    ),
    # E2E latency
    "e2e_sum": re.compile(
        r'^vllm:e2e_request_latency_seconds_sum(?:\{[^}]*\})?\s+([\d.]+)', re.MULTILINE
    ),
    "e2e_count": re.compile(
        r'^vllm:e2e_request_latency_seconds_count(?:\{[^}]*\})?\s+([\d.]+)', re.MULTILINE
    ),
    # Throughput: generation_tokens_total is a counter; we snapshot it over time
    # For now, use the running total as a proxy — collector computes delta externally
    "generation_tokens_total": re.compile(
        r'^vllm:generation_tokens_total(?:\{[^}]*\})?\s+([\d.]+)', re.MULTILINE
    ),
}

# p95 bucket approximations: highest bucket threshold we consider "p95"
# vLLM default buckets for TTFT: [0.001,0.005,0.01,0.02,0.04,0.06,0.08,0.1,0.25,0.5,0.75,1.0,2.5,5.0,7.5,10.0]
_TTFT_P95_BUCKET_RE = re.compile(
    r'^vllm:time_to_first_token_seconds_bucket\{.*?le="([\d.]+)".*?\}\s+([\d.]+)',
    re.MULTILINE,
)
_E2E_P95_BUCKET_RE = re.compile(
    r'^vllm:e2e_request_latency_seconds_bucket\{.*?le="([\d.]+)".*?\}\s+([\d.]+)',
    re.MULTILINE,
)
_ITL_P95_BUCKET_RE = re.compile(
    r'^vllm:time_per_output_token_seconds_bucket\{.*?le="([\d.]+)".*?\}\s+([\d.]+)',
    re.MULTILINE,
)


def _extract_p95_from_buckets(pattern: re.Pattern, text: str, total_count: float) -> Optional[float]:
    """
    Approximate p95 from a Prometheus cumulative histogram.

    Finds the smallest bucket whose cumulative count >= 0.95 * total_count
    and returns its upper-bound threshold (converted to ms).
    """
    if total_count <= 0:
        return None
    target = 0.95 * total_count
    matches = pattern.findall(text)
    for le_str, count_str in matches:
        try:
            if float(count_str) >= target:
                return float(le_str) * 1000.0  # seconds → ms
        except ValueError:
            continue
    return None


def _parse_vllm_metrics(text: str) -> dict[str, float]:
    """Parse vLLM Prometheus exposition into a dict of named floats."""
    values: dict[str, float] = {}
    for field, pattern in _VLLM_METRIC_PATTERNS.items():
        m = pattern.search(text)
        if m:
            try:
                values[field] = float(m.group(1))
            except ValueError:
                pass
    return values


class VLLMClient:
    """
    Lightweight async client for vLLM endpoint health and telemetry.

    Used by TelemetryCollector when backend_type == BackendType.VLLM.
    Not used for inference — only health + metrics polling.
    """

    def __init__(self, timeout: float = 5.0) -> None:
        self._client = httpx.AsyncClient(timeout=timeout)

    async def close(self) -> None:
        await self._client.aclose()

    async def is_ready(self, base_url: str) -> bool:
        """GET /health → True if 200 OK."""
        try:
            resp = await self._client.get(f"{base_url}/health")
            return resp.status_code == 200
        except Exception:
            return False

    async def list_models(self, base_url: str) -> list[dict[str, Any]]:
        """GET /v1/models → OpenAI-compatible model list."""
        try:
            resp = await self._client.get(f"{base_url}/v1/models")
            resp.raise_for_status()
            return resp.json().get("data", [])
        except Exception as exc:
            logger.warning("vllm_list_models_failed", base_url=base_url, error=str(exc))
            return []

    async def scrape_metrics(self, base_url: str) -> Optional[TelemetryUpdate]:
        """GET /metrics → parse Prometheus text into TelemetryUpdate."""
        try:
            resp = await self._client.get(f"{base_url}/metrics")
            if resp.status_code != 200:
                return None
            return self._parse_prometheus(resp.text, endpoint_id="")
        except Exception:
            return None

    def _parse_prometheus(self, text: str, endpoint_id: str) -> TelemetryUpdate:
        vals = _parse_vllm_metrics(text)

        # Compute mean latencies from sum/count
        def mean_ms(sum_key: str, count_key: str) -> Optional[float]:
            s = vals.get(sum_key)
            c = vals.get(count_key)
            if s is not None and c and c > 0:
                return (s / c) * 1000.0  # seconds → ms
            return None

        p50_ttft = mean_ms("ttft_sum", "ttft_count")
        p50_itl = mean_ms("itl_sum", "itl_count")
        p50_e2e = mean_ms("e2e_sum", "e2e_count")

        # Approximate p95 from histogram buckets
        ttft_count = vals.get("ttft_count", 0.0)
        e2e_count = vals.get("e2e_count", 0.0)
        itl_count = vals.get("itl_count", 0.0)

        p95_ttft = _extract_p95_from_buckets(_TTFT_P95_BUCKET_RE, text, ttft_count)
        p95_e2e = _extract_p95_from_buckets(_E2E_P95_BUCKET_RE, text, e2e_count)
        p95_itl = _extract_p95_from_buckets(_ITL_P95_BUCKET_RE, text, itl_count)

        # Saturation: GPU KV cache fill fraction is the best proxy for vLLM
        gpu_cache = vals.get("gpu_cache_usage")
        saturation = gpu_cache if gpu_cache is not None else None

        return TelemetryUpdate(
            endpoint_id=endpoint_id,
            queue_depth=int(vals["queue_depth"]) if "queue_depth" in vals else None,
            active_requests=int(vals["active_requests"]) if "active_requests" in vals else None,
            p50_ttft_ms=p50_ttft,
            p95_ttft_ms=p95_ttft,
            p50_itl_ms=p50_itl,
            p95_itl_ms=p95_itl,
            p50_e2e_ms=p50_e2e,
            p95_e2e_ms=p95_e2e,
            saturation_score=saturation,
        )

    async def probe(self, ep: EndpointProfile) -> TelemetryUpdate:
        """
        Full probe: health check + Prometheus metrics.
        Falls back to probe-timing estimates if /metrics is unavailable.
        """
        t_start = time.perf_counter()
        ready = await self.is_ready(ep.nim_url)
        probe_ms = (time.perf_counter() - t_start) * 1000

        metrics = await self.scrape_metrics(ep.nim_url)
        if metrics is not None:
            metrics.endpoint_id = ep.id
            if not ready:
                metrics.error_rate = 0.5
            return metrics

        # Fallback to probe timing
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
        return TelemetryUpdate(
            endpoint_id=ep.id,
            error_rate=1.0,
            saturation_score=1.0,
        )
