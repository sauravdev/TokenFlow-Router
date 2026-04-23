"""
SGLang inference backend adapter.

SGLang (https://github.com/sgl-project/sglang) uses RadixAttention for automatic
prefix caching across requests. It excels at:
  - Prefill-heavy workloads (long prompts, RAG pipelines, shared system prompts)
  - Multi-turn conversations with repeated prefixes
  - Tool-calling workloads where the tool schema is reused across requests

The cache_hit_rate from /get_server_info is a uniquely valuable routing signal:
a high hit rate means this endpoint is KV-warm for the incoming prefix, making it
the optimal choice for prefill-heavy traffic.

Health and telemetry endpoints:
  GET /health                 → basic liveness (200 OK)
  GET /health_generate        → readiness with actual inference warmup — use for routing
  GET /get_server_info        → rich JSON: queue stats, cache hit rate, throughput
  GET /v1/models              → OpenAI-compatible model list
  GET /metrics                → Prometheus (if enabled with --enable-metrics-for-all)

/get_server_info response shape (SGLang >= 0.3):
  {
    "model_path": "...",
    "num_running_reqs": 4,
    "num_waiting_reqs": 2,
    "token_usage": 0.42,          # KV cache usage fraction 0–1
    "cache_hit_rate": 0.71,       # RadixAttention prefix cache hit rate 0–1
    "avg_prefill_throughput": 12400.5,  # tokens/sec for prefill
    "avg_decode_throughput": 820.3,     # tokens/sec for decode
    "max_total_num_tokens": 131072,
    "context_len": 8192
  }
"""

from __future__ import annotations

import time
from typing import Any, Optional

import httpx
import structlog

from tokenflow.models import EndpointProfile, TelemetryUpdate

logger = structlog.get_logger(__name__)


class SGLangClient:
    """
    Lightweight async client for SGLang endpoint health and telemetry.

    Used by TelemetryCollector when backend_type == BackendType.SGLANG.
    Not used for inference — only health + metrics polling.

    Key routing insight: SGLang's cache_hit_rate should be fed into the
    TokenFlow scoring engine as a bonus signal for PREFILL_HEAVY workloads.
    A warm cache (>0.5 hit rate) dramatically reduces effective TTFT.
    """

    def __init__(self, timeout: float = 5.0) -> None:
        self._client = httpx.AsyncClient(timeout=timeout)

    async def close(self) -> None:
        await self._client.aclose()

    async def is_live(self, base_url: str) -> bool:
        """GET /health → basic liveness check."""
        try:
            resp = await self._client.get(f"{base_url}/health")
            return resp.status_code == 200
        except Exception:
            return False

    async def is_ready(self, base_url: str) -> bool:
        """
        GET /health_generate → full readiness check with inference warmup.

        This is more accurate than /health for routing purposes because it
        verifies the model is actually loaded and can process requests.
        Falls back to /health if /health_generate is unavailable.
        """
        try:
            resp = await self._client.get(f"{base_url}/health_generate", timeout=10.0)
            return resp.status_code == 200
        except httpx.TimeoutException:
            # Warmup in progress — treat as degraded not failed
            return False
        except Exception:
            # Fall back to basic health
            return await self.is_live(base_url)

    async def get_server_info(self, base_url: str) -> Optional[dict[str, Any]]:
        """
        GET /get_server_info → rich server state including cache hit rate.

        Returns None if the endpoint doesn't support this (older SGLang versions).
        """
        try:
            resp = await self._client.get(f"{base_url}/get_server_info")
            if resp.status_code == 200:
                return resp.json()
            return None
        except Exception:
            return None

    async def list_models(self, base_url: str) -> list[dict[str, Any]]:
        """GET /v1/models → OpenAI-compatible model list."""
        try:
            resp = await self._client.get(f"{base_url}/v1/models")
            resp.raise_for_status()
            return resp.json().get("data", [])
        except Exception as exc:
            logger.warning("sglang_list_models_failed", base_url=base_url, error=str(exc))
            return []

    def _server_info_to_telemetry(
        self, info: dict[str, Any], endpoint_id: str
    ) -> TelemetryUpdate:
        """
        Convert /get_server_info JSON into a TelemetryUpdate.

        token_usage (KV cache fill fraction) maps to saturation_score.
        cache_hit_rate is stored in capability_flags at the endpoint level;
        here we use it to adjust estimated TTFT — a high cache hit rate
        means effective TTFT is much lower (cache hit → skip prefill compute).
        """
        queue_depth = int(info.get("num_waiting_reqs", 0))
        active_requests = int(info.get("num_running_reqs", 0))
        token_usage: float = float(info.get("token_usage", 0.0))
        cache_hit_rate: float = float(info.get("cache_hit_rate", 0.0))

        avg_prefill_tps: float = float(info.get("avg_prefill_throughput", 0.0))
        avg_decode_tps: float = float(info.get("avg_decode_throughput", 0.0))

        # Derive approximate tokens/sec (weighted mix of prefill + decode)
        tokens_per_second: Optional[float] = None
        if avg_decode_tps > 0:
            tokens_per_second = avg_decode_tps

        # Derive TTFT estimate: high cache hit rate = cheaper prefill
        # Without real histogram data, estimate from prefill throughput.
        # Assume a 1K token prompt: ttft_ms = 1000 tokens / prefill_tps * 1000
        p50_ttft: Optional[float] = None
        if avg_prefill_tps > 0:
            base_ttft_ms = (1000.0 / avg_prefill_tps) * 1000.0  # 1K token probe
            # Cache hits reduce effective TTFT proportionally
            effective_ttft_ms = base_ttft_ms * (1.0 - cache_hit_rate * 0.8)
            p50_ttft = max(effective_ttft_ms, 1.0)

        # ITL from decode throughput: ms per output token
        p50_itl: Optional[float] = None
        if avg_decode_tps > 0:
            p50_itl = (1.0 / avg_decode_tps) * 1000.0  # ms per token

        return TelemetryUpdate(
            endpoint_id=endpoint_id,
            queue_depth=queue_depth,
            active_requests=active_requests,
            tokens_per_second=tokens_per_second,
            p50_ttft_ms=p50_ttft,
            p95_ttft_ms=p50_ttft * 2.0 if p50_ttft else None,
            p50_itl_ms=p50_itl,
            p95_itl_ms=p50_itl * 1.5 if p50_itl else None,
            saturation_score=min(token_usage, 1.0),
        )

    async def probe(self, ep: EndpointProfile) -> TelemetryUpdate:
        """
        Full probe: readiness check + server info.

        Injects cache_hit_rate into ep.capability_flags so the scoring engine
        can use it as a bonus signal for PREFILL_HEAVY requests.

        Sets capability_flags:
          - warm: True if endpoint is ready and has capacity
          - sglang_cache_hit_rate: RadixAttention prefix cache hit rate (0-1)
          - sglang_token_usage: KV cache fill fraction (0-1)
          - sglang_context_len: max context length reported by server
        """
        t_start = time.perf_counter()
        ready = await self.is_ready(ep.nim_url)
        probe_ms = (time.perf_counter() - t_start) * 1000

        if not ready:
            ep.capability_flags["warm"] = False
            return TelemetryUpdate(
                endpoint_id=ep.id,
                error_rate=1.0,
                saturation_score=1.0,
            )

        info = await self.get_server_info(ep.nim_url)
        if info:
            cache_hit_rate = float(info.get("cache_hit_rate", 0.0))
            token_usage = float(info.get("token_usage", 0.0))
            context_len = info.get("context_len")

            ep.capability_flags["sglang_cache_hit_rate"] = cache_hit_rate
            ep.capability_flags["sglang_token_usage"] = token_usage
            if context_len:
                ep.capability_flags["sglang_context_len"] = context_len

            # Warm if cache is populated and not saturated.
            # A high cache_hit_rate means the model is loaded and prefix cache
            # is primed — ideal state for routing prefill-heavy traffic.
            ep.capability_flags["warm"] = token_usage < 0.95

            logger.debug(
                "sglang_probe_ok",
                endpoint=ep.name,
                cache_hit_rate=round(cache_hit_rate, 3),
                token_usage=round(token_usage, 3),
                queue=info.get("num_waiting_reqs", "?"),
                warm=ep.capability_flags["warm"],
            )
            tel = self._server_info_to_telemetry(info, ep.id)
            tel.error_rate = 0.0
            return tel

        # Fall back to probe timing only — endpoint is live but no server_info
        ep.capability_flags["warm"] = True
        return TelemetryUpdate(
            endpoint_id=ep.id,
            p50_ttft_ms=probe_ms,
            p95_ttft_ms=probe_ms * 1.5,
            p50_e2e_ms=probe_ms,
            p95_e2e_ms=probe_ms * 2.0,
            error_rate=0.0,
            saturation_score=0.0,
        )
