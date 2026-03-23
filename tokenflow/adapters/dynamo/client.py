"""
NVIDIA Dynamo inference backend adapter.

NVIDIA Dynamo (https://github.com/ai-dynamo/dynamo) is a distributed inference
serving framework built on vLLM for the disaggregated prefill/decode architecture.
It is designed for large-scale multi-node deployments where KV-cache transfer
between prefill and decode workers is managed explicitly.

Key differentiators vs plain vLLM:
  - Disaggregated prefill/decode: separate GPU pools for each phase
  - KV-cache-aware worker routing: routes to workers with KV overlap
  - Multi-node KV transfer: NVLink / RDMA between prefill and decode nodes
  - Per-request routing hints: TokenFlow injects these via adapters/dynamo/hints.py

Health and telemetry endpoints (Dynamo exposes vLLM-compatible paths):
  GET /health          → 200 OK when router + workers are ready
  GET /metrics         → Prometheus (vllm: prefix, same as vLLM + dynamo-specific)
  GET /v1/models       → OpenAI-compatible model list

Additional Dynamo-specific Prometheus metrics (dynamo: prefix):
  dynamo:kv_cache_transfer_bandwidth_bytes_per_sec  — NVLink/RDMA KV transfer throughput
  dynamo:prefill_worker_queue_depth                 — prefill-side queue
  dynamo:decode_worker_queue_depth                  — decode-side queue
  dynamo:kv_hit_rate                                — KV block reuse rate (like vLLM cache_hit)
"""

from __future__ import annotations

import re
import time
from typing import Optional

import httpx
import structlog

from tokenflow.models import EndpointProfile, TelemetryUpdate

logger = structlog.get_logger(__name__)

# Dynamo uses vllm: prefix for standard metrics plus dynamo: for its own
_DYNAMO_PATTERNS: dict[str, re.Pattern] = {
    # Standard vLLM metrics (Dynamo exposes these too)
    "queue_depth": re.compile(
        r'^vllm:num_requests_waiting(?:\{[^}]*\})?\s+([\d.]+)', re.MULTILINE
    ),
    "active_requests": re.compile(
        r'^vllm:num_requests_running(?:\{[^}]*\})?\s+([\d.]+)', re.MULTILINE
    ),
    "gpu_cache_usage": re.compile(
        r'^vllm:gpu_cache_usage_perc(?:\{[^}]*\})?\s+([\d.]+)', re.MULTILINE
    ),
    "ttft_sum": re.compile(
        r'^vllm:time_to_first_token_seconds_sum(?:\{[^}]*\})?\s+([\d.]+)', re.MULTILINE
    ),
    "ttft_count": re.compile(
        r'^vllm:time_to_first_token_seconds_count(?:\{[^}]*\})?\s+([\d.]+)', re.MULTILINE
    ),
    "itl_sum": re.compile(
        r'^vllm:time_per_output_token_seconds_sum(?:\{[^}]*\})?\s+([\d.]+)', re.MULTILINE
    ),
    "itl_count": re.compile(
        r'^vllm:time_per_output_token_seconds_count(?:\{[^}]*\})?\s+([\d.]+)', re.MULTILINE
    ),
    "e2e_sum": re.compile(
        r'^vllm:e2e_request_latency_seconds_sum(?:\{[^}]*\})?\s+([\d.]+)', re.MULTILINE
    ),
    "e2e_count": re.compile(
        r'^vllm:e2e_request_latency_seconds_count(?:\{[^}]*\})?\s+([\d.]+)', re.MULTILINE
    ),
    # Dynamo-specific metrics
    "prefill_queue_depth": re.compile(
        r'^dynamo:prefill_worker_queue_depth(?:\{[^}]*\})?\s+([\d.]+)', re.MULTILINE
    ),
    "decode_queue_depth": re.compile(
        r'^dynamo:decode_worker_queue_depth(?:\{[^}]*\})?\s+([\d.]+)', re.MULTILINE
    ),
    "kv_hit_rate": re.compile(
        r'^dynamo:kv_hit_rate(?:\{[^}]*\})?\s+([\d.]+)', re.MULTILINE
    ),
    "kv_transfer_bw": re.compile(
        r'^dynamo:kv_cache_transfer_bandwidth_bytes_per_sec(?:\{[^}]*\})?\s+([\d.]+)',
        re.MULTILINE,
    ),
}

# Histogram bucket patterns for p95 estimation (reuse vLLM format)
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


def _p95_from_buckets(pattern: re.Pattern, text: str, total_count: float) -> Optional[float]:
    if total_count <= 0:
        return None
    target = 0.95 * total_count
    for le_str, count_str in pattern.findall(text):
        try:
            if float(count_str) >= target:
                return float(le_str) * 1000.0  # s → ms
        except ValueError:
            continue
    return None


class DynamoClient:
    """
    Lightweight async client for NVIDIA Dynamo endpoint health and telemetry.

    Used by TelemetryCollector when backend_type == BackendType.DYNAMO.
    Not used for inference — only health + metrics polling.

    Dynamo's disaggregated prefill/decode architecture means we track both
    prefill_queue_depth and decode_queue_depth separately. The composite
    queue_depth fed to TokenFlow is max(prefill, decode) to reflect the
    true bottleneck. The kv_hit_rate is stored in capability_flags so the
    scoring engine can reward Dynamo endpoints for KV-warm requests.
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

    async def scrape_metrics(self, base_url: str) -> Optional[str]:
        """Fetch raw Prometheus text from /metrics."""
        try:
            resp = await self._client.get(f"{base_url}/metrics")
            if resp.status_code == 200:
                return resp.text
            return None
        except Exception:
            return None

    def _parse_metrics(self, text: str, endpoint_id: str, ep: EndpointProfile) -> TelemetryUpdate:
        vals: dict[str, float] = {}
        for field, pattern in _DYNAMO_PATTERNS.items():
            m = pattern.search(text)
            if m:
                try:
                    vals[field] = float(m.group(1))
                except ValueError:
                    pass

        def mean_ms(sum_key: str, count_key: str) -> Optional[float]:
            s, c = vals.get(sum_key), vals.get(count_key)
            if s is not None and c and c > 0:
                return (s / c) * 1000.0
            return None

        # For disaggregated serving, effective queue depth = max of prefill+decode queues
        prefill_q = vals.get("prefill_queue_depth", vals.get("queue_depth", 0.0))
        decode_q = vals.get("decode_queue_depth", 0.0)
        effective_queue = int(max(prefill_q, decode_q))

        # KV hit rate → store in capability_flags for bonus scoring on KV-warm requests
        kv_hit_rate = vals.get("kv_hit_rate")
        if kv_hit_rate is not None:
            ep.capability_flags["dynamo_kv_hit_rate"] = kv_hit_rate

        # KV transfer bandwidth → proxy for inter-node health
        kv_bw = vals.get("kv_transfer_bw")
        if kv_bw is not None:
            ep.capability_flags["dynamo_kv_transfer_bw_gbps"] = round(kv_bw / 1e9, 2)

        ttft_count = vals.get("ttft_count", 0.0)
        e2e_count = vals.get("e2e_count", 0.0)
        itl_count = vals.get("itl_count", 0.0)

        return TelemetryUpdate(
            endpoint_id=endpoint_id,
            queue_depth=effective_queue,
            active_requests=int(vals["active_requests"]) if "active_requests" in vals else None,
            p50_ttft_ms=mean_ms("ttft_sum", "ttft_count"),
            p95_ttft_ms=_p95_from_buckets(_TTFT_P95_BUCKET_RE, text, ttft_count),
            p50_itl_ms=mean_ms("itl_sum", "itl_count"),
            p95_itl_ms=_p95_from_buckets(_ITL_P95_BUCKET_RE, text, itl_count),
            p50_e2e_ms=mean_ms("e2e_sum", "e2e_count"),
            p95_e2e_ms=_p95_from_buckets(_E2E_P95_BUCKET_RE, text, e2e_count),
            saturation_score=min(vals.get("gpu_cache_usage", 0.0), 1.0),
        )

    async def probe(self, ep: EndpointProfile) -> TelemetryUpdate:
        """Full probe: health check + Prometheus metrics."""
        t_start = time.perf_counter()
        ready = await self.is_ready(ep.nim_url)
        probe_ms = (time.perf_counter() - t_start) * 1000

        if not ready:
            return TelemetryUpdate(
                endpoint_id=ep.id,
                error_rate=1.0,
                saturation_score=1.0,
            )

        text = await self.scrape_metrics(ep.nim_url)
        if text:
            tel = self._parse_metrics(text, ep.id, ep)
            tel.error_rate = 0.0
            logger.debug(
                "dynamo_probe_ok",
                endpoint=ep.name,
                kv_hit_rate=ep.capability_flags.get("dynamo_kv_hit_rate", "?"),
            )
            return tel

        # Fallback to probe timing
        return TelemetryUpdate(
            endpoint_id=ep.id,
            p50_ttft_ms=probe_ms,
            p95_ttft_ms=probe_ms * 1.5,
            p50_e2e_ms=probe_ms,
            p95_e2e_ms=probe_ms * 2.0,
            error_rate=0.0,
            saturation_score=0.0,
        )
