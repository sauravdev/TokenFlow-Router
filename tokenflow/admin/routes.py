"""
Admin API routes.

All paths under /admin/...

Endpoints:
  POST   /admin/endpoints           — register endpoint
  GET    /admin/endpoints           — list all endpoints
  GET    /admin/endpoints/{id}      — get one endpoint
  DELETE /admin/endpoints/{id}      — delete endpoint
  PUT    /admin/endpoints/{id}/enable
  PUT    /admin/endpoints/{id}/disable

  POST   /admin/telemetry           — push telemetry update
  GET    /admin/telemetry/{id}      — get current telemetry for endpoint
  GET    /admin/telemetry/{id}/history

  GET    /admin/policy              — get active policy
  POST   /admin/policy              — replace active policy (JSON body)
  POST   /admin/policy/preset       — switch preset (latency-first | balanced | cost-first)

  GET    /admin/routes/explain/{request_id}  — explain a past routing decision
  GET    /admin/routes/recent                — last N routing decisions

  GET    /admin/metrics             — Prometheus text metrics
"""

from __future__ import annotations

from typing import Any

import structlog
from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import JSONResponse, PlainTextResponse, Response

from tokenflow.models import (
    EndpointRegisterRequest,
    RoutingPolicy,
    TelemetryUpdate,
)
from tokenflow.observability import get_metrics_output
from tokenflow.router import _apply_preset

logger = structlog.get_logger(__name__)
router = APIRouter(prefix="/admin")


def _state(request: Request):
    return request.app.state


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------


@router.post("/endpoints", status_code=201)
async def register_endpoint(
    body: EndpointRegisterRequest, request: Request
) -> JSONResponse:
    state = _state(request)
    profile = await state.registry.register(body)
    state.telemetry_collector.register_endpoint(profile)
    return JSONResponse(content=profile.model_dump(mode="json"), status_code=201)


@router.get("/endpoints")
async def list_endpoints(
    request: Request, enabled_only: bool = False, healthy_only: bool = False
) -> JSONResponse:
    state = _state(request)
    eps = await state.registry.list_all(
        enabled_only=enabled_only, healthy_only=healthy_only
    )
    return JSONResponse(content=[ep.model_dump(mode="json") for ep in eps])


@router.get("/endpoints/{endpoint_id}")
async def get_endpoint(endpoint_id: str, request: Request) -> JSONResponse:
    state = _state(request)
    ep = await state.registry.get(endpoint_id)
    if not ep:
        raise HTTPException(status_code=404, detail="Endpoint not found")
    return JSONResponse(content=ep.model_dump(mode="json"))


@router.delete("/endpoints/{endpoint_id}", status_code=204)
async def delete_endpoint(endpoint_id: str, request: Request) -> Response:
    state = _state(request)
    deleted = await state.registry.delete(endpoint_id)
    if not deleted:
        raise HTTPException(status_code=404, detail="Endpoint not found")
    state.telemetry_collector.unregister_endpoint(endpoint_id)
    return Response(status_code=204)


@router.put("/endpoints/{endpoint_id}/enable")
async def enable_endpoint(endpoint_id: str, request: Request) -> JSONResponse:
    state = _state(request)
    ok = await state.registry.enable(endpoint_id)
    if not ok:
        raise HTTPException(status_code=404, detail="Endpoint not found")
    return JSONResponse(content={"status": "enabled", "endpoint_id": endpoint_id})


@router.put("/endpoints/{endpoint_id}/disable")
async def disable_endpoint(endpoint_id: str, request: Request) -> JSONResponse:
    state = _state(request)
    ok = await state.registry.disable(endpoint_id)
    if not ok:
        raise HTTPException(status_code=404, detail="Endpoint not found")
    return JSONResponse(content={"status": "disabled", "endpoint_id": endpoint_id})


# ---------------------------------------------------------------------------
# Telemetry
# ---------------------------------------------------------------------------


@router.post("/telemetry")
async def push_telemetry(body: TelemetryUpdate, request: Request) -> JSONResponse:
    state = _state(request)
    tel = await state.telemetry_store.upsert(body)
    health = state.telemetry_store.compute_health(body.endpoint_id)
    await state.registry.update_health(body.endpoint_id, health)
    return JSONResponse(content=tel.model_dump(mode="json"))


@router.get("/telemetry/{endpoint_id}")
async def get_telemetry(endpoint_id: str, request: Request) -> JSONResponse:
    state = _state(request)
    tel = state.telemetry_store.get(endpoint_id)
    if tel is None:
        raise HTTPException(status_code=404, detail="No telemetry for endpoint")
    stale = state.telemetry_store.is_stale(endpoint_id)
    data = tel.model_dump(mode="json")
    data["_stale"] = stale
    return JSONResponse(content=data)


@router.get("/telemetry/{endpoint_id}/history")
async def get_telemetry_history(endpoint_id: str, request: Request) -> JSONResponse:
    state = _state(request)
    history = state.telemetry_store.get_history(endpoint_id)
    return JSONResponse(content=[t.model_dump(mode="json") for t in history])


# ---------------------------------------------------------------------------
# Policy
# ---------------------------------------------------------------------------


@router.get("/policy")
async def get_policy(request: Request) -> JSONResponse:
    state = _state(request)
    return JSONResponse(
        content=state.policy_engine.policy.model_dump(mode="json")
    )


@router.post("/policy")
async def set_policy(body: RoutingPolicy, request: Request) -> JSONResponse:
    state = _state(request)
    applied = _apply_preset(body)
    state.policy_engine.set_policy(applied)
    state.decision_engine.set_policy(applied)
    return JSONResponse(content=applied.model_dump(mode="json"))


@router.post("/policy/preset")
async def switch_preset(request: Request) -> JSONResponse:
    state = _state(request)
    body = await request.json()
    preset = body.get("preset", "balanced")
    if preset not in ("latency-first", "balanced", "cost-first"):
        raise HTTPException(
            status_code=400,
            detail="preset must be one of: latency-first, balanced, cost-first",
        )
    updated = state.policy_engine.policy.model_copy(update={"preset": preset})
    applied = _apply_preset(updated)
    state.policy_engine.set_policy(applied)
    state.decision_engine.set_policy(applied)
    return JSONResponse(content={"preset": preset, "weights": {
        "slo": applied.slo_weight,
        "cost": applied.cost_weight,
        "queue": applied.queue_weight,
        "gpu_affinity": applied.gpu_affinity_weight,
        "model_fit": applied.model_fit_weight,
        "reliability": applied.reliability_weight,
    }})


# ---------------------------------------------------------------------------
# Route explain
# ---------------------------------------------------------------------------


@router.get("/routes/explain/{request_id}")
async def explain_route(request_id: str, request: Request) -> JSONResponse:
    state = _state(request)
    trace = state.trace_store.get(request_id)
    if trace is None:
        raise HTTPException(
            status_code=404,
            detail=f"No routing trace found for request_id={request_id}",
        )
    return JSONResponse(content=trace.model_dump(mode="json"))


@router.get("/routes/recent")
async def recent_routes(request: Request, limit: int = 50) -> JSONResponse:
    state = _state(request)
    traces = state.trace_store.recent(limit=min(limit, 500))
    return JSONResponse(content=[t.model_dump(mode="json") for t in traces])


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------


@router.get("/metrics")
async def prometheus_metrics() -> Response:
    data, content_type = get_metrics_output()
    return Response(content=data, media_type=content_type)
