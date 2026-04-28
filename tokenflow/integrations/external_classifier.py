"""External-classifier bridge.

TokenFlow's scoring engine combines multiple signals (cost, queue, GPU
affinity, model fit, reliability, SLO) into a weighted utility score.
One of those signals is the inferred *workload type* (chat / reasoning /
prefill_heavy / decode_heavy / summarization / generation). The default
classifier is the lightweight keyword/shape heuristic in
`tokenflow/classifier.py`.

Some teams already run a more accurate classifier in production — for
example NVIDIA AI Blueprints' LLM Router v2 publishes a small Qwen 1.7B
model OR a CLIP+NN that recommends a target model per prompt, and many
teams have their own distilBERT / LLM-as-judge intent classifiers
trained on real traffic.

`ExternalClassifierClient` lets you plug any HTTP-served classifier
into TokenFlow's scoring pipeline without changing the policy DSL or
the scoring engine. The classifier's response is folded into the
request profile as a `workload_type` hint *before* scoring runs, so:

  - Hard constraints (context-fit, tenant allowlist, health) still
    apply.
  - Cost weighting still applies.
  - Per-tenant policies still apply.
  - Live policy swap still applies.

The external classifier just becomes one input — a more accurate input
than the keyword baseline — to the same multi-signal score. If the
classifier is unreachable / slow / errors out, TokenFlow falls back to
the local heuristic.

Wire format
-----------
TokenFlow calls the classifier with:

    POST {classifier_url}/recommendation
    {
      "messages": [...OpenAI chat messages...],
      "model": "<requested_model>",
      "metadata": {"tenant_id": "...", "priority_tier": "..."}
    }

The classifier should return:

    {
      "intent": "reasoning" | "chat" | "summarization" | "generation"
              | "decode_heavy" | "prefill_heavy" | "balanced",
      "recommended_model": "<optional model name>",
      "confidence": 0.0-1.0,
      "model_scores": {"<model>": <score>, ...}    # optional
    }

For the NVIDIA AI Blueprints router (v2 experimental), this maps to
the `/recommendation` endpoint exposed by `nat_sfc_router`. Other
classifiers can adopt the same shape.
"""
from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass
from typing import Any, Optional

import httpx
import structlog

logger = structlog.get_logger(__name__)


@dataclass
class ClassifierResult:
    """What an external classifier returns for a single prompt."""

    intent: Optional[str]                         # canonical workload type
    recommended_model: Optional[str]
    confidence: Optional[float]
    model_scores: dict[str, float]
    latency_ms: float
    source: str                                   # "external" | "fallback" | "error"


class ExternalClassifierClient:
    """Async client for an external classifier service.

    Designed to be swapped into the request-profile pipeline (in
    `tokenflow/classifier.py`) before scoring. Configurable timeout +
    fallback so a slow / down classifier never blocks routing.
    """

    # Map of common classifier-output labels → TokenFlow's WorkloadType enum
    # values. TokenFlow's WorkloadType has 4 canonical values:
    # PREFILL_HEAVY, DECODE_HEAVY, BALANCED, REASONING. NVIDIA v2 emits a
    # broader label set (chit_chat / hard_question / summary_request /
    # creative_writing / image_understanding / code_generation / math) — we
    # collapse those down. If your classifier emits something else, add a
    # row here.
    _INTENT_MAP = {
        # NVIDIA v2 router intent labels
        "hard_question":         "reasoning",
        "math":                  "reasoning",
        "code_generation":       "reasoning",
        "summary_request":       "prefill_heavy",   # long input, short output
        "creative_writing":      "decode_heavy",    # short prompt, long output
        "chit_chat":             "decode_heavy",    # short input, short-to-medium output
        "chitchat":              "decode_heavy",
        "image_understanding":   "balanced",        # multimodal — collapse
        # canonical TokenFlow values pass through
        "reasoning":             "reasoning",
        "decode_heavy":          "decode_heavy",
        "prefill_heavy":         "prefill_heavy",
        "balanced":              "balanced",
    }

    def __init__(
        self,
        classifier_url: str,
        timeout_s: float = 0.5,                   # tight default — classifier is hot path
        path: str = "/recommendation",
        api_key: Optional[str] = None,
    ) -> None:
        self.base_url = classifier_url.rstrip("/")
        self.path = path if path.startswith("/") else f"/{path}"
        self.timeout_s = timeout_s
        self.api_key = api_key
        self._client = httpx.AsyncClient(timeout=timeout_s)

    async def close(self) -> None:
        await self._client.aclose()

    async def classify(
        self,
        messages: list[dict[str, Any]],
        model: str = "",
        metadata: Optional[dict[str, Any]] = None,
    ) -> ClassifierResult:
        """Send the prompt to the external classifier. Always returns a
        ClassifierResult — never raises. On error / timeout the result has
        `source="error"` and intent=None, and the caller falls back to the
        local heuristic."""
        t_start = time.perf_counter()
        body = {
            "messages": messages,
            "model": model,
            "metadata": metadata or {},
        }
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        try:
            resp = await self._client.post(
                f"{self.base_url}{self.path}",
                json=body,
                headers=headers,
            )
            latency_ms = (time.perf_counter() - t_start) * 1000.0
            resp.raise_for_status()
            data = resp.json()
        except (httpx.RequestError, httpx.HTTPStatusError, asyncio.TimeoutError) as exc:
            latency_ms = (time.perf_counter() - t_start) * 1000.0
            logger.debug(
                "external_classifier_failed",
                error=str(exc),
                latency_ms=round(latency_ms, 1),
                falling_back="local heuristic",
            )
            return ClassifierResult(
                intent=None,
                recommended_model=None,
                confidence=None,
                model_scores={},
                latency_ms=latency_ms,
                source="error",
            )

        raw_intent = (data.get("intent") or "").strip().lower()
        canonical = self._INTENT_MAP.get(raw_intent)

        return ClassifierResult(
            intent=canonical,                                              # may be None
            recommended_model=data.get("recommended_model"),
            confidence=float(data["confidence"]) if "confidence" in data else None,
            model_scores=dict(data.get("model_scores") or {}),
            latency_ms=latency_ms,
            source="external",
        )


# ---------------------------------------------------------------------------
# Helper to install a classifier into the running router
# ---------------------------------------------------------------------------


def install_into_state(state: Any, client: ExternalClassifierClient) -> None:
    """Wire an ExternalClassifierClient into a running router.

    The router holds an `app.state` with a `classifier` attribute (the
    local RequestClassifier). This helper attaches the external client
    so the local classifier can consult it during request profiling.

    Use it once at startup in your custom main:

        from tokenflow.integrations.external_classifier import (
            ExternalClassifierClient, install_into_state
        )

        client = ExternalClassifierClient(
            classifier_url="http://nvidia-router-v2:5000",
            timeout_s=0.5,
        )
        install_into_state(app.state, client)

    The local RequestClassifier checks for `state.external_classifier`
    on every request; if set, it calls the external classifier first
    and uses the canonical intent if returned, otherwise falls back to
    the local heuristic.
    """
    state.external_classifier = client
    logger.info(
        "external_classifier_installed",
        url=client.base_url,
        timeout_s=client.timeout_s,
    )
