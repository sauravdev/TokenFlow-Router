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
from tokenflow.models import PriorityTier
from tokenflow.observability import ACTIVE_REQUESTS

logger = structlog.get_logger(__name__)

router = APIRouter()
classifier = RequestClassifier()


def _get_app_state(request: Request):
    return request.app.state


@router.post("/v1/chat/completions")
async def chat_completions(
    request: Request,
    x_tenant_id: str = Header(default="default"),
    x_app_id: str = Header(default="default"),
    x_priority_tier: str = Header(default="standard"),
    x_budget_sensitivity: float = Header(default=0.5),
) -> Any:
    state = _get_app_state(request)
    body = await request.json()

    # Validate priority tier
    try:
        priority = PriorityTier(x_priority_tier.lower())
    except ValueError:
        priority = PriorityTier.STANDARD

    # Classify request
    current_rpm = await state.policy_engine.rpm_tracker.current_rpm(x_tenant_id)
    profile = classifier.classify(
        body=body,
        tenant_id=x_tenant_id,
        app_id=x_app_id,
        priority_tier=priority,
        budget_sensitivity=x_budget_sensitivity,
        current_tenant_rpm=current_rpm,
    )

    # Apply policy
    profile, policy_actions = await state.policy_engine.apply(profile)
    if policy_actions:
        logger.debug("policy_applied", request_id=profile.request_id, actions=policy_actions)

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

    # Background: lazily activate any profile templates matching this workload
    asyncio.create_task(
        state.profile_manager.maybe_activate_for_workload(profile.workload_type)
    )

    streaming = body.get("stream", False)
    failed_ids: list[str] = []

    # Attempt with fallback chain
    for attempt in range(settings.max_fallback_attempts):
        try:
            ACTIVE_REQUESTS.labels(endpoint_name=endpoint.name).inc()
            t_start = time.perf_counter()

            if streaming:
                return await _stream_response(
                    request, state, profile, decision, endpoint, body
                )
            else:
                result = await state.proxy.forward(
                    endpoint, body, dict(request.headers)
                )
                e2e_ms = (time.perf_counter() - t_start) * 1000
                ACTIVE_REQUESTS.labels(endpoint_name=endpoint.name).dec()
                state.trace_store.record_actual_latency(
                    profile.request_id, endpoint.name, None, e2e_ms
                )
                await state.policy_engine.record_cost(
                    profile.tenant_id, decision.estimated_cost_usd
                )
                # Inject routing metadata
                result["_tokenflow"] = {
                    "request_id": profile.request_id,
                    "endpoint": endpoint.name,
                    "decision_ms": round(decision.decision_latency_ms, 2),
                }
                return JSONResponse(content=result)

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


async def _stream_response(request, state, profile, decision, endpoint, body):
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
            await state.policy_engine.record_cost(
                profile.tenant_id, decision.estimated_cost_usd
            )

    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers={
            "X-TokenFlow-Request-ID": profile.request_id,
            "X-TokenFlow-Endpoint": endpoint.name,
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
