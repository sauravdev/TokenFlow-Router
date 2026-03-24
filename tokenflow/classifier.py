"""Request classifier — enriches raw requests with routing metadata."""

from __future__ import annotations

from typing import Any

import structlog

from tokenflow.models import (
    LatencyClass,
    PriorityTier,
    RequestProfile,
    TokenBand,
    WorkloadType,
)

logger = structlog.get_logger(__name__)

# ---------------------------------------------------------------------------
# Tiktoken setup — optional; fall back to char-heuristic if unavailable
# ---------------------------------------------------------------------------

try:
    import tiktoken as _tiktoken
    # cl100k_base covers GPT-4, Llama-3, Mistral, Qwen, and most modern models.
    # It won't be byte-perfect for every model but is far more accurate than
    # the 4-chars/token heuristic for routing purposes.
    _TIKTOKEN_ENC = _tiktoken.get_encoding("cl100k_base")
    _HAS_TIKTOKEN = True
    logger.debug("tiktoken_available", encoding="cl100k_base")
except Exception:
    _TIKTOKEN_ENC = None
    _HAS_TIKTOKEN = False
    logger.debug("tiktoken_unavailable_using_heuristic")

# Token band boundaries
_TINY_MAX = 256
_SMALL_MAX = 1_024
_MEDIUM_MAX = 4_096
_LARGE_MAX = 16_384

# Prefill/decode classification thresholds
_PREFILL_HEAVY_RATIO = 3.0   # input_tokens / output_tokens > 3 → prefill heavy
_DECODE_HEAVY_RATIO = 3.0    # output_tokens / input_tokens > 3 → decode heavy


def _token_band(n: int) -> TokenBand:
    if n <= _TINY_MAX:
        return TokenBand.TINY
    if n <= _SMALL_MAX:
        return TokenBand.SMALL
    if n <= _MEDIUM_MAX:
        return TokenBand.MEDIUM
    if n <= _LARGE_MAX:
        return TokenBand.LARGE
    return TokenBand.XLARGE


def _estimate_output_tokens(body: dict[str, Any]) -> int:
    """Best-effort output token estimate from request body."""
    # Explicit max_tokens from caller
    if "max_tokens" in body:
        return int(body["max_tokens"])
    if "max_completion_tokens" in body:
        return int(body["max_completion_tokens"])
    # Default by model hint
    model = body.get("model", "").lower()
    if "reasoning" in model or "o1" in model or "o3" in model:
        return 2048
    return 256


def _count_input_tokens(body: dict[str, Any]) -> int:
    """
    Count input tokens from the request body.

    Uses tiktoken (cl100k_base) when available — accurate to within ~1–2%
    for most modern models. Falls back to the 4-chars/token heuristic when
    tiktoken is not installed, which is good enough for routing decisions
    where relative ordering matters more than exact counts.
    """
    messages = body.get("messages", [])
    system_prompt = str(body["system"]) if "system" in body else ""
    tools = body.get("tools", [])

    if _HAS_TIKTOKEN and _TIKTOKEN_ENC is not None:
        total = 0
        for m in messages:
            if isinstance(m, dict):
                content = m.get("content", "")
                if isinstance(content, str):
                    total += len(_TIKTOKEN_ENC.encode(content))
                elif isinstance(content, list):
                    # Multi-modal / tool-call content blocks
                    for block in content:
                        if isinstance(block, dict) and block.get("type") == "text":
                            total += len(_TIKTOKEN_ENC.encode(block.get("text", "")))
                # Role token overhead (~4 tokens per message)
                total += 4
        if system_prompt:
            total += len(_TIKTOKEN_ENC.encode(system_prompt)) + 4
        for tool in tools:
            total += len(_TIKTOKEN_ENC.encode(str(tool)))
        return max(1, total)

    # Heuristic fallback: ~4 chars per token
    total_chars = sum(
        len(str(m.get("content", ""))) for m in messages if isinstance(m, dict)
    )
    if system_prompt:
        total_chars += len(system_prompt)
    for tool in tools:
        total_chars += len(str(tool))
    return max(1, total_chars // 4)


def _infer_workload_type(
    input_tokens: int, output_tokens: int, model_name: str
) -> WorkloadType:
    prefill_ratio = input_tokens / max(output_tokens, 1)
    decode_ratio = output_tokens / max(input_tokens, 1)
    model_lower = model_name.lower()

    # Reasoning models are always reasoning workloads
    if any(k in model_lower for k in ("reasoning", "o1", "o3", "r1", "qwq")):
        return WorkloadType.REASONING

    if prefill_ratio >= _PREFILL_HEAVY_RATIO:
        return WorkloadType.PREFILL_HEAVY
    if decode_ratio >= _DECODE_HEAVY_RATIO:
        return WorkloadType.DECODE_HEAVY
    return WorkloadType.BALANCED


def _infer_latency_class(
    priority: PriorityTier,
    streaming: bool,
    output_tokens: int,
) -> LatencyClass:
    if priority == PriorityTier.OFFLINE:
        return LatencyClass.OFFLINE
    if priority == PriorityTier.BATCH:
        return LatencyClass.BATCH
    if priority == PriorityTier.PREMIUM or streaming:
        return LatencyClass.INTERACTIVE
    if output_tokens > 1000:
        return LatencyClass.STANDARD
    return LatencyClass.INTERACTIVE


def _burst_class(rpm: float) -> str:
    if rpm > 500:
        return "spike"
    if rpm > 100:
        return "burst"
    return "normal"


class RequestClassifier:
    """
    Classifies an incoming request into a RequestProfile.

    Derives:
    - token counts (exact if provided, estimated otherwise)
    - workload type (prefill-heavy / decode-heavy / balanced / reasoning)
    - latency class (interactive / standard / batch / offline)
    - token bands
    - prefill/decode ratios
    """

    def classify(
        self,
        body: dict[str, Any],
        tenant_id: str = "default",
        app_id: str = "default",
        priority_tier: PriorityTier = PriorityTier.STANDARD,
        budget_sensitivity: float = 0.5,
        current_tenant_rpm: float = 0.0,
    ) -> RequestProfile:
        model_requested = body.get("model", "unknown")
        streaming = bool(body.get("stream", False))

        input_tokens = _count_input_tokens(body)
        predicted_output = _estimate_output_tokens(body)

        workload_type = _infer_workload_type(input_tokens, predicted_output, model_requested)
        latency_class = _infer_latency_class(priority_tier, streaming, predicted_output)

        prefill_ratio = input_tokens / max(predicted_output, 1)
        decode_ratio = predicted_output / max(input_tokens, 1)

        profile = RequestProfile(
            tenant_id=tenant_id,
            app_id=app_id,
            model_requested=model_requested,
            input_tokens=input_tokens,
            predicted_output_tokens=predicted_output,
            priority_tier=priority_tier,
            latency_class=latency_class,
            budget_sensitivity=budget_sensitivity,
            streaming=streaming,
            workload_type=workload_type,
            input_token_band=_token_band(input_tokens),
            output_token_band=_token_band(predicted_output),
            prefill_ratio=prefill_ratio,
            decode_ratio=decode_ratio,
            burst_class=_burst_class(current_tenant_rpm),
            raw_body=body,
        )

        logger.debug(
            "request_classified",
            request_id=profile.request_id,
            workload=workload_type,
            latency_class=latency_class,
            input_tokens=input_tokens,
            output_tokens=predicted_output,
            priority=priority_tier,
        )
        return profile
