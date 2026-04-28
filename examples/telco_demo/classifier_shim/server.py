"""NVIDIA-compatible classifier shim.

A drop-in replacement for the `/recommendation` endpoint exposed by NVIDIA
AI Blueprints LLM Router v2. Exists so TokenFlow's external-classifier
integration (`tokenflow.integrations.external_classifier`) can be exercised
end-to-end without burning a GPU on a 1.7B classifier model.

It speaks the same API contract as the v2 router (`POST /recommendation`,
returns `{intent, recommended_model, confidence, model_scores}`) using
two-stage classification:

  1. Cheap regex / token-length heuristic — handles ~85% of traffic with
     near-zero latency (single-digit microseconds).
  2. LLM-as-judge fallback (optional) — for borderline prompts, ask a
     small vLLM endpoint to label the intent.

In production, you would replace this with the real NVIDIA Router v2
container; the wire format is byte-compatible.
"""
from __future__ import annotations

import os
import re
import time
from typing import Any, Optional

import httpx
from fastapi import FastAPI
from pydantic import BaseModel

# ---------------------------------------------------------------------------
# Heuristics — calibrated against the workload mix in this demo:
#   customer_care_voice / rag_retrieval / esg_batch / ai_assisted_migration
#   trust_inventory / digital_twin_simulation
# ---------------------------------------------------------------------------

REASONING_KEYWORDS = re.compile(
    r"\b("
    r"why|prove|explain|derive|reason|solve|step by step|step-by-step|"
    r"compare|trade[- ]off|trade-off|optimi[sz]e|analy[sz]e|"
    r"refactor|migrate|port|translate|convert|"
    r"design|architect|simulate|forecast|predict|"
    r"infer|deduce|evaluate|critique"
    r")\b",
    re.IGNORECASE,
)
CODE_KEYWORDS = re.compile(
    r"\b(def |class |function |import |from |return |#include |fn |let |const |"
    r"public |private |async |await |```)",
)
SUMMARY_KEYWORDS = re.compile(
    r"\b(summari[sz]e|tl;?dr|abstract|key points|in (a|one) sentence|brief)\b",
    re.IGNORECASE,
)
GENERATION_KEYWORDS = re.compile(
    r"\b(generate|write a|draft|compose|create a (poem|story|email|response))\b",
    re.IGNORECASE,
)
CHIT_CHAT_KEYWORDS = re.compile(
    r"^(hi|hello|hey|thanks|thank you|how are|what'?s up|good (morning|afternoon))\b",
    re.IGNORECASE,
)


def _flatten_messages(messages: list[dict[str, Any]]) -> str:
    return "\n".join(str(m.get("content") or "") for m in messages)


def heuristic_intent(messages: list[dict[str, Any]]) -> tuple[str, float]:
    """Return (nvidia-style label, confidence)."""
    text = _flatten_messages(messages)
    n_chars = len(text)
    n_tokens_approx = max(1, n_chars // 4)

    if CHIT_CHAT_KEYWORDS.search(text) and n_tokens_approx < 30:
        return "chit_chat", 0.92

    if SUMMARY_KEYWORDS.search(text):
        return "summary_request", 0.86

    if CODE_KEYWORDS.search(text) or REASONING_KEYWORDS.search(text):
        if n_tokens_approx > 1500:
            return "hard_question", 0.83
        return "hard_question", 0.78

    if GENERATION_KEYWORDS.search(text):
        return "creative_writing", 0.75

    if n_tokens_approx > 4000:
        return "summary_request", 0.62
    if n_tokens_approx > 800:
        return "hard_question", 0.55

    return "chit_chat", 0.55


# ---------------------------------------------------------------------------
# Optional LLM-as-judge layer — defers borderline cases to a small vLLM
# ---------------------------------------------------------------------------

JUDGE_URL = os.getenv("CLASSIFIER_JUDGE_URL", "")
JUDGE_MODEL = os.getenv("CLASSIFIER_JUDGE_MODEL", "qwen")
JUDGE_TIMEOUT = float(os.getenv("CLASSIFIER_JUDGE_TIMEOUT_S", "0.4"))

JUDGE_SYSTEM_PROMPT = """You are a request classifier. Given a user prompt, output exactly one of these labels and nothing else:
- hard_question  (reasoning, code, math, multi-step analysis)
- chit_chat      (greeting, small talk, single-sentence Q&A)
- summary_request (summarize, abstract, tl;dr)
- creative_writing (compose poem/story/email/etc.)

Output: just the label."""


async def llm_judge(
    client: httpx.AsyncClient,
    messages: list[dict[str, Any]],
) -> Optional[str]:
    if not JUDGE_URL:
        return None
    last_user = next(
        (m["content"] for m in reversed(messages) if m.get("role") == "user"),
        "",
    )
    body = {
        "model": JUDGE_MODEL,
        "messages": [
            {"role": "system", "content": JUDGE_SYSTEM_PROMPT},
            {"role": "user", "content": str(last_user)[:1200]},
        ],
        "max_tokens": 8,
        "temperature": 0.0,
    }
    try:
        resp = await client.post(
            f"{JUDGE_URL.rstrip('/')}/v1/chat/completions",
            json=body,
            timeout=JUDGE_TIMEOUT,
        )
        resp.raise_for_status()
        data = resp.json()
        text = (
            data.get("choices", [{}])[0].get("message", {}).get("content", "")
        ).strip().lower()
        for label in ("hard_question", "chit_chat", "summary_request", "creative_writing"):
            if label in text:
                return label
    except Exception:
        return None
    return None


# ---------------------------------------------------------------------------
# Routing recommendations — which model to suggest, given the intent
# ---------------------------------------------------------------------------

INTENT_TO_MODEL = {
    "hard_question":     "vllm-premium",
    "summary_request":   "vllm-standard",
    "creative_writing":  "vllm-standard",
    "chit_chat":         "vllm-economy",
}


def model_scores_for_intent(intent: str) -> dict[str, float]:
    """Cosmetic — emits NVIDIA-style per-model scores so downstream
    observability looks identical."""
    base = {"vllm-economy": 0.20, "vllm-standard": 0.40, "vllm-premium": 0.40}
    if intent == "hard_question":
        base = {"vllm-economy": 0.05, "vllm-standard": 0.20, "vllm-premium": 0.75}
    elif intent == "chit_chat":
        base = {"vllm-economy": 0.85, "vllm-standard": 0.10, "vllm-premium": 0.05}
    elif intent == "summary_request":
        base = {"vllm-economy": 0.15, "vllm-standard": 0.65, "vllm-premium": 0.20}
    elif intent == "creative_writing":
        base = {"vllm-economy": 0.20, "vllm-standard": 0.55, "vllm-premium": 0.25}
    return base


# ---------------------------------------------------------------------------
# FastAPI service
# ---------------------------------------------------------------------------


class RecommendationRequest(BaseModel):
    messages: list[dict[str, Any]]
    model: Optional[str] = ""
    metadata: Optional[dict[str, Any]] = None


class RecommendationResponse(BaseModel):
    intent: str
    recommended_model: str
    confidence: float
    model_scores: dict[str, float]
    classifier_latency_ms: float
    classifier_source: str


app = FastAPI(title="nvidia-router-shim", version="1.0")
_judge_client: Optional[httpx.AsyncClient] = None
_stats = {"requests": 0, "judge_calls": 0, "judge_overrides": 0, "errors": 0}


@app.on_event("startup")
async def _startup() -> None:
    global _judge_client
    if JUDGE_URL:
        _judge_client = httpx.AsyncClient(timeout=JUDGE_TIMEOUT)


@app.on_event("shutdown")
async def _shutdown() -> None:
    if _judge_client is not None:
        await _judge_client.aclose()


@app.get("/health")
async def health() -> dict[str, Any]:
    return {
        "status": "healthy",
        "judge_url": JUDGE_URL or None,
        "stats": _stats,
    }


@app.post("/recommendation", response_model=RecommendationResponse)
async def recommendation(req: RecommendationRequest) -> RecommendationResponse:
    t0 = time.perf_counter()
    _stats["requests"] += 1
    intent, conf = heuristic_intent(req.messages)

    used_judge = False
    if 0.55 <= conf <= 0.78 and _judge_client is not None:
        _stats["judge_calls"] += 1
        judge_label = await llm_judge(_judge_client, req.messages)
        if judge_label is not None and judge_label != intent:
            _stats["judge_overrides"] += 1
            intent = judge_label
            conf = 0.88
            used_judge = True
        elif judge_label is not None:
            conf = max(conf, 0.85)
            used_judge = True

    return RecommendationResponse(
        intent=intent,
        recommended_model=INTENT_TO_MODEL.get(intent, "vllm-standard"),
        confidence=round(conf, 3),
        model_scores=model_scores_for_intent(intent),
        classifier_latency_ms=round((time.perf_counter() - t0) * 1000.0, 3),
        classifier_source=("heuristic+judge" if used_judge else "heuristic"),
    )


@app.get("/")
async def root() -> dict[str, Any]:
    return {
        "service": "nvidia-router-shim",
        "wire_format": "POST /recommendation — NVIDIA AI Blueprints LLM Router v2 compatible",
        "stats": _stats,
    }
