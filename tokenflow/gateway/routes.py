"""
Gateway routes — OpenAI-compatible inference endpoint with routing logic.

POST /v1/chat/completions  — main inference route
GET  /v1/models            — list available models
GET  /health               — liveness probe
"""

from __future__ import annotations

import asyncio
import time
from typing import Any

import structlog
from fastapi import APIRouter, Depends, Header, HTTPException, Request
from fastapi.responses import JSONResponse, StreamingResponse

from tokenflow.classifier import RequestClassifier
from tokenflow.config import settings
from tokenflow.models import OptimizationTarget, PriorityTier, TelemetryUpdate
from tokenflow.observability import ACTIVE_REQUESTS

logger = structlog.get_logger(__name__)

router = APIRouter()
classifier = RequestClassifier()


async def _feed_back_latency(state, endpoint_id: str, ttft_ms: float | None, e2e_ms: float):
    """Push actual measured latency back into the telemetry store.

    This closes the feedback loop: real inference latency from the upstream
    response updates the EMA-smoothed telemetry, which future routing
    decisions use. Without this, the store only has scrape-interval data
    and heuristic estimates.
    """
    try:
        update = TelemetryUpdate(
            endpoint_id=endpoint_id,
            p95_e2e_ms=e2e_ms,
        )
        if ttft_ms is not None:
            update.p95_ttft_ms = ttft_ms
        await state.telemetry_store.upsert(update)
    except Exception:
        pass


def _get_app_state(request: Request):
    return request.app.state


@router.post("/v1/chat/completions")
async def chat_completions(
    request: Request,
    x_tenant_id: str = Header(default="default"),
    x_app_id: str = Header(default="default"),
    x_priority_tier: str = Header(default="standard"),
    x_optimization_target: str = Header(default="auto"),
    x_budget_sensitivity: float = Header(default=0.5),
) -> Any:
    state = _get_app_state(request)
    body = await request.json()

    # Validate priority tier
    try:
        priority = PriorityTier(x_priority_tier.lower())
    except ValueError:
        priority = PriorityTier.STANDARD

    requested_optimization = body.get("routing", {}).get(
        "optimize_for", x_optimization_target
    )
    try:
        optimization_target = OptimizationTarget(str(requested_optimization).lower())
    except ValueError:
        optimization_target = OptimizationTarget.AUTO

    # Classify request
    current_rpm = await state.policy_engine.rpm_tracker.current_rpm(x_tenant_id)
    profile = classifier.classify(
        body=body,
        tenant_id=x_tenant_id,
        app_id=x_app_id,
        priority_tier=priority,
        optimization_target=optimization_target,
        budget_sensitivity=x_budget_sensitivity,
        current_tenant_rpm=current_rpm,
    )

    # Optional: refine workload_type via an external classifier (e.g.
    # NVIDIA AI Blueprints LLM Router v2 in intent profile, or your own
    # distilBERT / LLM-as-judge service). Non-blocking on failure — the
    # local heuristic remains in place if the call errors or times out.
    external_classifier = getattr(state, "external_classifier", None)
    if external_classifier is not None:
        try:
            ext = await external_classifier.classify(
                messages=body.get("messages", []),
                model=profile.model_requested,
                metadata={
                    "tenant_id": x_tenant_id,
                    "priority_tier": getattr(priority, "value", str(priority)),
                },
            )
            if ext.intent:
                from tokenflow.models import WorkloadType
                try:
                    profile.workload_type = WorkloadType(ext.intent)
                    logger.debug(
                        "workload_type_refined_by_external_classifier",
                        request_id=profile.request_id,
                        intent=ext.intent,
                        confidence=ext.confidence,
                        latency_ms=round(ext.latency_ms, 1),
                    )
                except ValueError:
                    pass
        except Exception as exc:
            logger.debug("external_classifier_skipped", error=str(exc))

    # Apply policy
    profile, policy_actions = await state.policy_engine.apply(profile)
    if policy_actions:
        logger.debug("policy_applied", request_id=profile.request_id, actions=policy_actions)

    # Ensure there is at least one matching live endpoint before routing.
    await state.profile_manager.ensure_capacity_for_request(profile)

    # Route decision
    decision = await state.decision_engine.decide(profile)

    if decision.selected_endpoint_id is None:
        raise HTTPException(
            status_code=503,
            detail={
                "error": "no_available_endpoint",
                "message": "No healthy endpoint could serve this request.",
                "request_id": profile.request_id,
                "rejections": decision.hard_rejections,
            },
        )

    endpoint = await state.registry.get(decision.selected_endpoint_id)
    if not endpoint:
        raise HTTPException(status_code=503, detail="Selected endpoint not found in registry.")

    # Record trace
    state.trace_store.record(profile, decision, state.policy_engine.policy.name)

    # Background: lazily activate only profile templates relevant to this request
    asyncio.create_task(
        state.profile_manager.maybe_activate_for_request(profile)
    )

    streaming = body.get("stream", False)
    failed_ids: list[str] = []

    # Attempt with fallback chain
    for attempt in range(settings.max_fallback_attempts):
        try:
            ACTIVE_REQUESTS.labels(endpoint_name=endpoint.name).inc()
            t_start = time.perf_counter()

            if streaming:
                await state.profile_manager.record_endpoint_use(endpoint.id)
                capacity_plan = await state.profile_manager.capacity_plan_for_request(
                    profile, endpoint.id
                )
                return await _stream_response(
                    request, state, profile, decision, endpoint, body, capacity_plan
                )
            else:
                result = await state.proxy.forward(
                    endpoint, body, dict(request.headers)
                )
                await state.profile_manager.record_endpoint_use(endpoint.id)
                capacity_plan = await state.profile_manager.capacity_plan_for_request(
                    profile, endpoint.id
                )
                e2e_ms = (time.perf_counter() - t_start) * 1000
                ACTIVE_REQUESTS.labels(endpoint_name=endpoint.name).dec()
                state.trace_store.record_actual_latency(
                    profile.request_id, endpoint.name, None, e2e_ms
                )
                asyncio.create_task(
                    _feed_back_latency(state, endpoint.id, None, e2e_ms)
                )
                await state.policy_engine.record_cost(
                    profile.tenant_id, decision.estimated_cost_usd
                )
                # Inject routing metadata
                result["_tokenflow"] = {
                    "request_id": profile.request_id,
                    "endpoint": endpoint.name,
                    "backend": endpoint.backend_type.value,
                    "gpu": endpoint.gpu_name.value,
                    "decision_ms": round(decision.decision_latency_ms, 2),
                    "optimization_target": profile.optimization_target.value,
                    "request_shape": {
                        "llm_model": profile.model_requested,
                        "model_family": profile.inferred_model_family,
                        "model_size_b": profile.inferred_model_size_b,
                        "isl_tokens": profile.isl_tokens,
                        "osl_tokens": profile.osl_tokens,
                        "total_tokens": profile.total_tokens,
                        "workload_type": profile.workload_type.value,
                    },
                    "capacity_plan": capacity_plan,
                    "end_user_benefit": (
                        "faster first token and snappier interaction"
                        if profile.optimization_target == OptimizationTarget.LATENCY
                        else "higher sustained throughput and better fleet utilization"
                    ),
                }
                headers = {
                    "X-TokenFlow-Active-Backend": endpoint.backend_type.value,
                    "X-TokenFlow-Active-Endpoint": endpoint.name,
                    "X-TokenFlow-Turn-Down-Candidates": ",".join(
                        c["endpoint"] for c in capacity_plan["turn_down_candidates"]
                    ),
                }
                return JSONResponse(content=result, headers=headers)

        except Exception as exc:
            ACTIVE_REQUESTS.labels(endpoint_name=endpoint.name).dec()
            logger.warning(
                "upstream_failed_trying_fallback",
                attempt=attempt + 1,
                endpoint=endpoint.name,
                error=str(exc),
            )
            failed_ids.append(endpoint.id)

            # Try fallback
            fallback = await state.decision_engine.fallback_chain(profile, failed_ids)
            if fallback.selected_endpoint_id is None:
                raise HTTPException(
                    status_code=503,
                    detail={
                        "error": "all_endpoints_failed",
                        "request_id": profile.request_id,
                        "attempts": attempt + 1,
                    },
                )
            endpoint = await state.registry.get(fallback.selected_endpoint_id)
            if endpoint is None:
                raise HTTPException(
                    status_code=503,
                    detail={
                        "error": "fallback_endpoint_vanished",
                        "message": "Fallback endpoint was removed from registry mid-request.",
                        "request_id": profile.request_id,
                    },
                )
            decision = fallback

    raise HTTPException(status_code=503, detail="Max fallback attempts exceeded.")


async def _stream_response(request, state, profile, decision, endpoint, body, capacity_plan):
    """Return a StreamingResponse for SSE."""
    t_start = time.perf_counter()
    ttft_ms = None

    async def generate():
        nonlocal ttft_ms
        try:
            async for chunk, chunk_ttft in state.proxy.forward_streaming(
                endpoint, body, dict(request.headers)
            ):
                if chunk_ttft is not None:
                    ttft_ms = chunk_ttft
                yield chunk
        finally:
            e2e_ms = (time.perf_counter() - t_start) * 1000
            ACTIVE_REQUESTS.labels(endpoint_name=endpoint.name).dec()
            state.trace_store.record_actual_latency(
                profile.request_id, endpoint.name, ttft_ms, e2e_ms
            )
            asyncio.create_task(
                _feed_back_latency(state, endpoint.id, ttft_ms, e2e_ms)
            )
            await state.policy_engine.record_cost(
                profile.tenant_id, decision.estimated_cost_usd
            )

    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers={
            "X-TokenFlow-Request-ID": profile.request_id,
            "X-TokenFlow-Endpoint": endpoint.name,
            "X-TokenFlow-Active-Backend": endpoint.backend_type.value,
            "X-TokenFlow-Active-Endpoint": endpoint.name,
            "X-TokenFlow-Turn-Down-Candidates": ",".join(
                c["endpoint"] for c in capacity_plan["turn_down_candidates"]
            ),
        },
    )


@router.get("/v1/models")
async def list_models(request: Request) -> JSONResponse:
    state = _get_app_state(request)
    endpoints = await state.registry.list_all(enabled_only=True)
    models = {}
    for ep in endpoints:
        if ep.model_name not in models:
            models[ep.model_name] = {
                "id": ep.model_name,
                "object": "model",
                "owned_by": "tokenflow-router",
                "endpoints": [],
            }
        models[ep.model_name]["endpoints"].append(ep.name)

    return JSONResponse(
        content={"object": "list", "data": list(models.values())}
    )


@router.get("/health")
async def health(request: Request) -> JSONResponse:
    state = _get_app_state(request)
    return JSONResponse(
        content={
            "status": "ok",
            "version": "0.1.0",
            "endpoints_registered": state.registry.count,
            "endpoints_healthy": state.registry.healthy_count,
            "policy": state.policy_engine.policy.name,
        }
    )
