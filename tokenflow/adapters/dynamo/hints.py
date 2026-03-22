"""
NVIDIA Dynamo downstream routing hint injection.

TokenFlow Router operates at the cross-endpoint/cross-pool layer.
When the selected endpoint runs Dynamo as its serving backend,
we can optionally inject per-request hints to influence Dynamo's
internal worker-level routing — e.g., KV-cache affinity, decode
priority, queue tier.

This is V2 functionality and is opt-in via endpoint capability_flags.

Dynamo supports per-request override patterns including:
- kv_cache_affinity_hint: prefix hash to prefer KV-warm workers
- request_priority: controls queue ordering within a Dynamo worker pool
- routing_mode: prefer TTFT-optimized vs ITL-optimized workers
"""

from __future__ import annotations

from typing import Any

from tokenflow.models import EndpointProfile, RequestProfile, WorkloadType


def build_dynamo_hints(
    ep: EndpointProfile, req: RequestProfile
) -> dict[str, Any]:
    """
    Build Dynamo per-request routing hints to attach to the request body.

    These are injected as extra fields that Dynamo-aware NIM builds
    will forward to the Dynamo router for worker-level routing decisions.

    Returns an empty dict if the endpoint doesn't support Dynamo hints.
    """
    if not ep.capability_flags.get("dynamo_hints_enabled", False):
        return {}

    hints: dict[str, Any] = {}

    # Routing mode: align with TokenFlow's workload classification
    if req.workload_type == WorkloadType.PREFILL_HEAVY:
        # Higher KV cache affinity weight helps TTFT for prefill-heavy traffic
        hints["routing_mode"] = "ttft_optimized"
        hints["kv_cache_affinity_weight"] = 0.8
    elif req.workload_type == WorkloadType.DECODE_HEAVY:
        # Lower KV affinity weight helps ITL for decode-heavy traffic
        hints["routing_mode"] = "itl_optimized"
        hints["kv_cache_affinity_weight"] = 0.2
    else:
        hints["routing_mode"] = "balanced"
        hints["kv_cache_affinity_weight"] = 0.5

    # Request priority within Dynamo's queue
    priority_map = {
        "premium": 0,    # highest priority
        "standard": 5,
        "batch": 8,
        "offline": 10,   # lowest priority
    }
    hints["request_priority"] = priority_map.get(req.priority_tier.value, 5)

    # TokenFlow request ID for end-to-end tracing
    hints["tokenflow_request_id"] = req.request_id

    return hints


def inject_hints(body: dict[str, Any], hints: dict[str, Any]) -> dict[str, Any]:
    """Inject Dynamo hints into the request body under 'x_dynamo_hints' key."""
    if not hints:
        return body
    return {**body, "x_dynamo_hints": hints}
