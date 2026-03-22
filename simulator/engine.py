"""
Simulator engine — replays request traces through the routing engine
and reports how decisions would have been made under different policies.

Use cases:
1. Shadow mode: compare current policy vs proposed policy side-by-side
2. Replay: re-route historical requests and compare outcomes
3. Scenario testing: inject synthetic traffic patterns
4. Policy tuning: find weights that minimise cost while meeting SLO
"""

from __future__ import annotations

import random
import statistics
from dataclasses import dataclass, field
from typing import Any

from tokenflow.classifier import RequestClassifier
from tokenflow.models import (
    CostClass,
    EndpointProfile,
    EndpointTelemetry,
    GPUClass,
    RoutingPolicy,
    WorkloadType,
)
from tokenflow.registry import EndpointRegistry
from tokenflow.router import DecisionEngine, _apply_preset
from tokenflow.telemetry import TelemetryStore


# ---------------------------------------------------------------------------
# Synthetic endpoint factory
# ---------------------------------------------------------------------------


def make_synthetic_endpoint(
    name: str,
    model: str,
    gpu: GPUClass,
    cost_class: CostClass,
    cost_per_gpu_hour: float = 3.0,
    supports_reasoning: bool = False,
) -> EndpointProfile:
    return EndpointProfile(
        name=name,
        nim_url=f"http://mock-{name}:8000",
        model_name=model,
        gpu_name=gpu,
        cost_class=cost_class,
        cost_per_gpu_hour=cost_per_gpu_hour,
        supports_reasoning=supports_reasoning,
        max_context_tokens=32768,
    )


def make_synthetic_telemetry(
    endpoint_id: str,
    gpu: GPUClass,
    queue_depth: int = 5,
    error_rate: float = 0.01,
) -> EndpointTelemetry:
    """Generate realistic synthetic telemetry based on GPU class."""
    gpu_perf = {
        GPUClass.H100: dict(ttft=80, itl=15, e2e=1200, tps=1200),
        GPUClass.A100: dict(ttft=120, itl=22, e2e=1800, tps=800),
        GPUClass.L40S: dict(ttft=200, itl=35, e2e=3000, tps=450),
        GPUClass.L40: dict(ttft=250, itl=40, e2e=3800, tps=380),
        GPUClass.A10G: dict(ttft=350, itl=55, e2e=5500, tps=260),
        GPUClass.L4: dict(ttft=500, itl=80, e2e=8000, tps=150),
        GPUClass.UNKNOWN: dict(ttft=300, itl=50, e2e=4000, tps=300),
    }.get(gpu, dict(ttft=300, itl=50, e2e=4000, tps=300))

    jitter = lambda v: v * (1 + random.uniform(-0.1, 0.15))

    return EndpointTelemetry(
        endpoint_id=endpoint_id,
        rpm=random.uniform(10, 80),
        rph=random.uniform(600, 4800),
        queue_depth=queue_depth,
        active_requests=max(1, queue_depth // 2),
        tokens_per_second=jitter(gpu_perf["tps"]),
        p50_ttft_ms=jitter(gpu_perf["ttft"]),
        p95_ttft_ms=jitter(gpu_perf["ttft"] * 1.8),
        p50_itl_ms=jitter(gpu_perf["itl"]),
        p95_itl_ms=jitter(gpu_perf["itl"] * 1.6),
        p50_e2e_ms=jitter(gpu_perf["e2e"]),
        p95_e2e_ms=jitter(gpu_perf["e2e"] * 1.7),
        error_rate=error_rate,
        saturation_score=min(0.95, queue_depth / 50),
    )


# ---------------------------------------------------------------------------
# Standard fleet preset
# ---------------------------------------------------------------------------


def standard_fleet() -> list[EndpointProfile]:
    """A realistic heterogeneous fleet for testing."""
    return [
        make_synthetic_endpoint(
            "nim-h100-llama3-70b", "meta/llama-3.1-70b-instruct",
            GPUClass.H100, CostClass.PREMIUM, cost_per_gpu_hour=8.0,
            supports_reasoning=True,
        ),
        make_synthetic_endpoint(
            "nim-a100-llama3-70b", "meta/llama-3.1-70b-instruct",
            GPUClass.A100, CostClass.PREMIUM, cost_per_gpu_hour=5.0,
        ),
        make_synthetic_endpoint(
            "nim-l40s-llama3-8b", "meta/llama-3.1-8b-instruct",
            GPUClass.L40S, CostClass.STANDARD, cost_per_gpu_hour=2.5,
        ),
        make_synthetic_endpoint(
            "nim-l40s-llama3-70b", "meta/llama-3.1-70b-instruct",
            GPUClass.L40S, CostClass.STANDARD, cost_per_gpu_hour=3.5,
        ),
        make_synthetic_endpoint(
            "nim-l4-llama3-8b", "meta/llama-3.1-8b-instruct",
            GPUClass.L4, CostClass.ECONOMY, cost_per_gpu_hour=0.8,
        ),
        make_synthetic_endpoint(
            "nim-a10g-llama3-8b", "meta/llama-3.1-8b-instruct",
            GPUClass.A10G, CostClass.STANDARD, cost_per_gpu_hour=1.5,
        ),
    ]


# ---------------------------------------------------------------------------
# Synthetic request generator
# ---------------------------------------------------------------------------


def make_request_body(
    model: str = "meta/llama-3.1-8b-instruct",
    workload: str = "balanced",
    input_tokens: int = 500,
    output_tokens: int = 256,
) -> dict[str, Any]:
    """Generate a synthetic OpenAI-style request body."""
    prompt = "x " * input_tokens  # approximate token count
    return {
        "model": model,
        "messages": [
            {"role": "user", "content": prompt},
        ],
        "max_tokens": output_tokens,
        "stream": False,
    }


# ---------------------------------------------------------------------------
# Simulation run
# ---------------------------------------------------------------------------


@dataclass
class SimulationResult:
    policy_name: str
    total_requests: int
    routed: int
    rejected: int
    fallback_used: int
    avg_decision_ms: float
    avg_predicted_ttft_ms: float
    avg_predicted_e2e_ms: float
    avg_estimated_cost_usd: float
    total_estimated_cost_usd: float
    endpoint_distribution: dict[str, int] = field(default_factory=dict)
    slo_attainment_rate: float = 0.0  # fraction meeting SLO targets


async def run_simulation(
    requests: list[dict[str, Any]],
    policy: RoutingPolicy,
    fleet: list[EndpointProfile] | None = None,
    slo_ttft_ms: float = 500.0,
    slo_e2e_ms: float = 5000.0,
) -> SimulationResult:
    """
    Run a set of synthetic or historical requests through the routing engine
    under a given policy and collect aggregate statistics.
    """
    if fleet is None:
        fleet = standard_fleet()

    # Build in-memory registry + telemetry
    registry = EndpointRegistry()
    store = TelemetryStore()
    classifier = RequestClassifier()

    for ep in fleet:
        await registry.register(
            type(
                "Req",
                (),
                {
                    "model_dump": lambda self, ep=ep: {
                        k: v
                        for k, v in ep.model_dump().items()
                        if k
                        in {
                            "name",
                            "nim_url",
                            "model_name",
                            "model_family",
                            "model_version",
                            "gpu_name",
                            "gpu_count",
                            "region",
                            "cost_class",
                            "max_context_tokens",
                            "supports_streaming",
                            "supports_reasoning",
                            "tenant_tags",
                            "capability_flags",
                            "cost_per_gpu_hour",
                        }
                    }
                },
            )()
        )
        tel = make_synthetic_telemetry(ep.id, ep.gpu_name)
        from tokenflow.models import TelemetryUpdate
        await store.upsert(TelemetryUpdate(endpoint_id=ep.id, **{
            k: getattr(tel, k) for k in TelemetryUpdate.model_fields
            if k != "endpoint_id" and hasattr(tel, k)
        }))

    engine = DecisionEngine(registry=registry, store=store)
    engine.set_policy(_apply_preset(policy))

    decision_latencies: list[float] = []
    ttfts: list[float] = []
    e2es: list[float] = []
    costs: list[float] = []
    endpoint_dist: dict[str, int] = {}
    rejected = 0
    fallback_used = 0
    slo_met = 0

    for body in requests:
        profile = classifier.classify(body)
        decision = await engine.decide(profile)

        if decision.selected_endpoint_id is None:
            rejected += 1
            continue

        if decision.fallback_used:
            fallback_used += 1

        decision_latencies.append(decision.decision_latency_ms)
        ttfts.append(decision.predicted_ttft_ms)
        e2es.append(decision.predicted_e2e_ms)
        costs.append(decision.estimated_cost_usd)

        name = decision.selected_endpoint_name or "unknown"
        endpoint_dist[name] = endpoint_dist.get(name, 0) + 1

        if (
            decision.predicted_ttft_ms <= slo_ttft_ms
            and decision.predicted_e2e_ms <= slo_e2e_ms
        ):
            slo_met += 1

    routed = len(requests) - rejected
    return SimulationResult(
        policy_name=policy.name,
        total_requests=len(requests),
        routed=routed,
        rejected=rejected,
        fallback_used=fallback_used,
        avg_decision_ms=statistics.mean(decision_latencies) if decision_latencies else 0.0,
        avg_predicted_ttft_ms=statistics.mean(ttfts) if ttfts else 0.0,
        avg_predicted_e2e_ms=statistics.mean(e2es) if e2es else 0.0,
        avg_estimated_cost_usd=statistics.mean(costs) if costs else 0.0,
        total_estimated_cost_usd=sum(costs),
        endpoint_distribution=endpoint_dist,
        slo_attainment_rate=slo_met / routed if routed > 0 else 0.0,
    )
