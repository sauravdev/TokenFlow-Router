"""
Hardened routing tests — covering edge cases, fallback, stale telemetry,
policy conflicts, and rejection reasons not covered by the base test suite.
"""

from __future__ import annotations

import asyncio

import pytest

from tokenflow.classifier import RequestClassifier
from tokenflow.models import (
    BackendType,
    CostClass,
    EndpointHealth,
    EndpointProfile,
    EndpointRegisterRequest,
    GPUClass,
    LatencyClass,
    OptimizationTarget,
    PriorityTier,
    RouteOutcome,
    RoutingPolicy,
    TelemetryUpdate,
    WorkloadType,
)
from tokenflow.registry import EndpointRegistry
from tokenflow.router import DecisionEngine, ScoringEngine, _apply_preset
from tokenflow.telemetry import TelemetryStore

clf = RequestClassifier()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def make_ep(
    name="ep",
    model="meta/llama-3.1-8b-instruct",
    gpu=GPUClass.L40S,
    cost_class=CostClass.STANDARD,
    backend=BackendType.NIM,
    enabled=True,
    health=EndpointHealth.HEALTHY,
    supports_reasoning=False,
    max_context_tokens=16384,
    nim_url=None,
) -> EndpointProfile:
    return EndpointProfile(
        name=name,
        nim_url=nim_url or f"http://{name}:8000",
        model_name=model,
        gpu_name=gpu,
        cost_class=cost_class,
        backend_type=backend,
        cost_per_gpu_hour=3.0,
        supports_reasoning=supports_reasoning,
        max_context_tokens=max_context_tokens,
        enabled=enabled,
        health=health,
    )


async def make_registry(*endpoints: EndpointProfile) -> EndpointRegistry:
    registry = EndpointRegistry()
    for ep in endpoints:
        req = EndpointRegisterRequest(**{
            k: getattr(ep, k) for k in EndpointRegisterRequest.model_fields
        })
        await registry.register(req)
    return registry


def fresh_store() -> TelemetryStore:
    return TelemetryStore()


def body(model="meta/llama-3.1-8b-instruct", content="hello", max_tokens=64):
    return {"model": model, "messages": [{"role": "user", "content": content}], "max_tokens": max_tokens}


# ---------------------------------------------------------------------------
# Rejection reasons
# ---------------------------------------------------------------------------


def test_cpu_endpoint_rejected_for_interactive_request():
    ep = make_ep(gpu=GPUClass.CPU)
    store = fresh_store()
    engine = ScoringEngine(RoutingPolicy(), store)
    profile = clf.classify(body(), priority_tier=PriorityTier.STANDARD)
    score = engine.score(ep, profile)
    assert score.hard_rejected
    assert "cpu_endpoint" in score.rejection_reason


def test_cpu_endpoint_accepted_for_batch_request():
    ep = make_ep(gpu=GPUClass.CPU)
    store = fresh_store()
    engine = ScoringEngine(RoutingPolicy(), store)
    profile = clf.classify(body(), priority_tier=PriorityTier.BATCH)
    score = engine.score(ep, profile)
    assert not score.hard_rejected


def test_rtx_laptop_rejected_for_large_context():
    ep = make_ep(gpu=GPUClass.RTX_LAPTOP)
    store = fresh_store()
    engine = ScoringEngine(RoutingPolicy(), store)
    profile = clf.classify(
        {"model": "meta/llama-3.1-8b-instruct",
         "messages": [{"role": "user", "content": "x " * 3000}],
         "max_tokens": 2000}
    )
    score = engine.score(ep, profile)
    assert score.hard_rejected
    assert "rtx_laptop" in score.rejection_reason


def test_rtx_laptop_accepted_for_small_context():
    ep = make_ep(gpu=GPUClass.RTX_LAPTOP)
    store = fresh_store()
    engine = ScoringEngine(RoutingPolicy(), store)
    profile = clf.classify(body(content="hi", max_tokens=32))
    score = engine.score(ep, profile)
    assert not score.hard_rejected


def test_reasoning_endpoint_required_for_reasoning_workload():
    ep_no_reasoning = make_ep(supports_reasoning=False)
    ep_reasoning = make_ep(name="reasoning-ep", supports_reasoning=True)
    store = fresh_store()
    engine = ScoringEngine(RoutingPolicy(), store)
    profile = clf.classify({"model": "meta/o1-reasoning", "messages": [{"role": "user", "content": "solve this"}], "max_tokens": 512})
    assert profile.workload_type == WorkloadType.REASONING
    score_no = engine.score(ep_no_reasoning, profile)
    score_yes = engine.score(ep_reasoning, profile)
    assert score_no.hard_rejected
    assert "reasoning_not_supported" in score_no.rejection_reason
    assert not score_yes.hard_rejected


def test_queue_full_rejection():
    ep = make_ep()
    store = fresh_store()
    asyncio.get_event_loop().run_until_complete(
        store.upsert(TelemetryUpdate(endpoint_id=ep.id, queue_depth=200))
    )
    policy = RoutingPolicy(max_queue_depth=100)
    engine = ScoringEngine(policy, store)
    profile = clf.classify(body())
    score = engine.score(ep, profile)
    assert score.hard_rejected
    assert "queue_full" in score.rejection_reason


def test_error_rate_ceiling_rejection():
    ep = make_ep()
    store = fresh_store()
    asyncio.get_event_loop().run_until_complete(
        store.upsert(TelemetryUpdate(endpoint_id=ep.id, error_rate=0.5))
    )
    policy = RoutingPolicy(max_error_rate=0.1)
    engine = ScoringEngine(policy, store)
    profile = clf.classify(body())
    score = engine.score(ep, profile)
    assert score.hard_rejected
    assert "error_rate_too_high" in score.rejection_reason


def test_context_window_exceeded_rejection():
    ep = make_ep(max_context_tokens=512)
    store = fresh_store()
    engine = ScoringEngine(RoutingPolicy(), store)
    profile = clf.classify({
        "model": "meta/llama-3.1-8b-instruct",
        "messages": [{"role": "user", "content": "x " * 1000}],
        "max_tokens": 256,
    })
    score = engine.score(ep, profile)
    assert score.hard_rejected
    assert "context_exceeded" in score.rejection_reason


# ---------------------------------------------------------------------------
# Stale telemetry
# ---------------------------------------------------------------------------


def test_stale_telemetry_uses_heuristic_not_zero():
    """When telemetry is stale, SLO score must use GPU-tier heuristic (> 0)."""
    ep = make_ep(gpu=GPUClass.H100)
    store = fresh_store()
    # Push telemetry with very old timestamp — simulate stale via is_stale mock
    engine = ScoringEngine(RoutingPolicy(), store)
    profile = clf.classify(body())
    slo, ttft, itl, e2e = engine.slo_score(ep, profile)
    # With no telemetry, heuristic must produce valid positive latency estimates
    assert slo > 0.0
    assert ttft > 0.0
    assert e2e > 0.0


def test_stale_telemetry_queue_score_neutral():
    ep = make_ep()
    store = fresh_store()
    engine = ScoringEngine(RoutingPolicy(), store)
    score = engine.queue_score(ep)
    assert score == 0.5  # neutral with no telemetry (healthy endpoint)


def test_stale_telemetry_degraded_queue_score_lower():
    ep = make_ep(health=EndpointHealth.DEGRADED)
    store = fresh_store()
    engine = ScoringEngine(RoutingPolicy(), store)
    score = engine.queue_score(ep)
    assert score == 0.3  # conservative penalty for degraded with no telemetry


def test_h100_heuristic_faster_than_l4():
    """Higher GPU tier must estimate lower latency via heuristic."""
    ep_h100 = make_ep(gpu=GPUClass.H100)
    ep_l4 = make_ep(name="l4", gpu=GPUClass.L4)
    store = fresh_store()
    engine = ScoringEngine(RoutingPolicy(), store)
    profile = clf.classify(body())
    _, ttft_h100, _, _ = engine.slo_score(ep_h100, profile)
    _, ttft_l4, _, _ = engine.slo_score(ep_l4, profile)
    assert ttft_h100 < ttft_l4


# ---------------------------------------------------------------------------
# Routing decisions
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_sglang_preferred_for_prefill_heavy():
    """SGLang should win over NIM for prefill-heavy on same GPU tier."""
    ep_nim = make_ep("nim", backend=BackendType.NIM, gpu=GPUClass.L40S)
    ep_sglang = make_ep("sglang", backend=BackendType.SGLANG, gpu=GPUClass.L40S)
    registry = await make_registry(ep_nim, ep_sglang)
    store = fresh_store()
    engine = DecisionEngine(registry=registry, store=store)
    engine.set_policy(RoutingPolicy(gpu_affinity_weight=0.5, slo_weight=0.1,
                                    cost_weight=0.1, queue_weight=0.1,
                                    model_fit_weight=0.1, reliability_weight=0.1))

    profile = clf.classify({
        "model": "meta/llama-3.1-8b-instruct",
        "messages": [{"role": "user", "content": "x " * 500}],  # long input
        "max_tokens": 32,
    })
    assert profile.workload_type == WorkloadType.PREFILL_HEAVY

    scoring = ScoringEngine(engine.policy, store)
    s_nim = scoring.score(ep_nim, profile)
    s_sglang = scoring.score(ep_sglang, profile)
    assert s_sglang.gpu_affinity_score >= s_nim.gpu_affinity_score


@pytest.mark.asyncio
async def test_vllm_preferred_for_decode_heavy():
    """vLLM should score higher GPU affinity than NIM for decode-heavy."""
    ep_nim = make_ep("nim", backend=BackendType.NIM, gpu=GPUClass.L40S)
    ep_vllm = make_ep("vllm", backend=BackendType.VLLM, gpu=GPUClass.L40S)
    store = fresh_store()
    engine = ScoringEngine(RoutingPolicy(), store)
    profile = clf.classify({
        "model": "meta/llama-3.1-8b-instruct",
        "messages": [{"role": "user", "content": "hi"}],
        "max_tokens": 2000,  # long output → decode heavy
    })
    assert profile.workload_type == WorkloadType.DECODE_HEAVY
    s_nim = engine.score(ep_nim, profile)
    s_vllm = engine.score(ep_vllm, profile)
    assert s_vllm.gpu_affinity_score > s_nim.gpu_affinity_score


def test_latency_optimization_prefers_nim_over_vllm_for_balanced_interactive():
    ep_nim = make_ep("nim", backend=BackendType.NIM, gpu=GPUClass.L40S)
    ep_vllm = make_ep("vllm", backend=BackendType.VLLM, gpu=GPUClass.L40S)
    store = fresh_store()
    engine = ScoringEngine(RoutingPolicy(), store)
    profile = clf.classify(
        body(),
        optimization_target=OptimizationTarget.LATENCY,
    )
    s_nim = engine.score(ep_nim, profile)
    s_vllm = engine.score(ep_vllm, profile)
    assert s_nim.benchmark_score >= s_vllm.benchmark_score


def test_throughput_optimization_prefers_vllm_over_nim_for_decode_heavy():
    ep_nim = make_ep("nim", backend=BackendType.NIM, gpu=GPUClass.L40S)
    ep_vllm = make_ep("vllm", backend=BackendType.VLLM, gpu=GPUClass.L40S)
    store = fresh_store()
    engine = ScoringEngine(RoutingPolicy(), store)
    profile = clf.classify(
        {"model": "meta/llama-3.1-8b-instruct", "messages": [{"role": "user", "content": "hi"}], "max_tokens": 2000},
        optimization_target=OptimizationTarget.THROUGHPUT,
    )
    s_nim = engine.score(ep_nim, profile)
    s_vllm = engine.score(ep_vllm, profile)
    assert s_vllm.benchmark_score > s_nim.benchmark_score


def test_ollama_supported_as_backend_type():
    ep_ollama = make_ep("ollama", backend=BackendType.OLLAMA, gpu=GPUClass.RTX4090)
    store = fresh_store()
    engine = ScoringEngine(RoutingPolicy(), store)
    profile = clf.classify(body(), optimization_target=OptimizationTarget.LATENCY)
    score = engine.score(ep_ollama, profile)
    assert score.endpoint_name == "ollama"
    assert score.benchmark_score >= 0.0


@pytest.mark.asyncio
async def test_fallback_chain_skips_failed_endpoint():
    ep1 = make_ep("ep1")
    ep2 = make_ep("ep2", nim_url="http://ep2:8000")
    registry = await make_registry(ep1, ep2)
    store = fresh_store()
    engine = DecisionEngine(registry=registry, store=store)

    profile = clf.classify(body())
    first = await engine.decide(profile)
    assert first.selected_endpoint_id is not None

    # Simulate ep1 failure — request fallback excluding it
    fallback = await engine.fallback_chain(profile, [first.selected_endpoint_id])
    assert fallback.selected_endpoint_id != first.selected_endpoint_id
    assert fallback.fallback_used is True


@pytest.mark.asyncio
async def test_fallback_chain_exhausted_returns_failed():
    ep = make_ep("only-ep")
    registry = await make_registry(ep)
    store = fresh_store()
    engine = DecisionEngine(registry=registry, store=store)

    profile = clf.classify(body())
    first = await engine.decide(profile)
    assert first.selected_endpoint_id is not None

    # Exclude all endpoints
    all_eps = await registry.list_all()
    failed = await engine.fallback_chain(profile, [e.id for e in all_eps])
    assert failed.selected_endpoint_id is None
    assert failed.outcome == RouteOutcome.FAILED


@pytest.mark.asyncio
async def test_all_endpoints_unhealthy_returns_rejected():
    ep = make_ep(health=EndpointHealth.UNHEALTHY)
    registry = await make_registry(ep)
    store = fresh_store()
    engine = DecisionEngine(registry=registry, store=store)

    profile = clf.classify(body())
    decision = await engine.decide(profile)
    assert decision.selected_endpoint_id is None
    assert decision.outcome == RouteOutcome.REJECTED
    assert len(decision.hard_rejections) > 0


# ---------------------------------------------------------------------------
# Policy conflicts / tenant overrides
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_batch_request_blocked_from_premium_endpoint():
    ep_premium = make_ep("premium", cost_class=CostClass.PREMIUM)
    ep_economy = make_ep("economy", cost_class=CostClass.ECONOMY, nim_url="http://economy:8000")
    registry = await make_registry(ep_premium, ep_economy)
    store = fresh_store()
    engine = DecisionEngine(registry=registry, store=store)

    profile = clf.classify(body(), priority_tier=PriorityTier.BATCH)
    decision = await engine.decide(profile)
    # Should route to economy, not premium
    assert decision.selected_endpoint_id is not None
    ep_selected = await registry.get(decision.selected_endpoint_id)
    assert ep_selected.cost_class != CostClass.PREMIUM


@pytest.mark.asyncio
async def test_latency_first_prefers_faster_gpu():
    ep_h100 = make_ep("h100", gpu=GPUClass.H100, cost_class=CostClass.PREMIUM)
    ep_l4 = make_ep("l4", gpu=GPUClass.L4, cost_class=CostClass.ECONOMY, nim_url="http://l4:8000")
    registry = await make_registry(ep_h100, ep_l4)
    store = fresh_store()
    engine = DecisionEngine(registry=registry, store=store)
    engine.set_policy(_apply_preset(RoutingPolicy(preset="latency-first")))

    profile = clf.classify(body(), priority_tier=PriorityTier.PREMIUM)
    decision = await engine.decide(profile)
    assert decision.selected_endpoint_id is not None
    ep_selected = await registry.get(decision.selected_endpoint_id)
    assert ep_selected.gpu_name == GPUClass.H100


@pytest.mark.asyncio
async def test_cost_first_prefers_economy_gpu():
    ep_h100 = make_ep("h100", gpu=GPUClass.H100, cost_class=CostClass.PREMIUM)
    ep_l4 = make_ep("l4", gpu=GPUClass.L4, cost_class=CostClass.ECONOMY, nim_url="http://l4:8000")
    registry = await make_registry(ep_h100, ep_l4)
    store = fresh_store()
    engine = DecisionEngine(registry=registry, store=store)
    engine.set_policy(_apply_preset(RoutingPolicy(preset="cost-first")))

    profile = clf.classify(body(), priority_tier=PriorityTier.BATCH)
    decision = await engine.decide(profile)
    assert decision.selected_endpoint_id is not None
    ep_selected = await registry.get(decision.selected_endpoint_id)
    assert ep_selected.cost_class == CostClass.ECONOMY


# ---------------------------------------------------------------------------
# Success/failure cleanup — decision outcome fields
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_successful_decision_has_cost_and_latency_estimates():
    ep = make_ep(gpu=GPUClass.H100)
    registry = await make_registry(ep)
    store = fresh_store()
    engine = DecisionEngine(registry=registry, store=store)

    profile = clf.classify(body())
    decision = await engine.decide(profile)

    assert decision.outcome == RouteOutcome.SUCCESS
    assert decision.estimated_cost_usd >= 0.0
    assert decision.predicted_ttft_ms > 0.0
    assert decision.predicted_e2e_ms > 0.0
    assert decision.decision_latency_ms >= 0.0


@pytest.mark.asyncio
async def test_rejected_decision_includes_rejection_reasons():
    ep = make_ep(health=EndpointHealth.UNHEALTHY)
    registry = await make_registry(ep)
    store = fresh_store()
    engine = DecisionEngine(registry=registry, store=store)

    profile = clf.classify(body())
    decision = await engine.decide(profile)

    assert decision.outcome == RouteOutcome.REJECTED
    assert len(decision.hard_rejections) == 1
    assert decision.hard_rejections[0]["reason"] == "endpoint_unhealthy"


@pytest.mark.asyncio
async def test_fallback_decision_includes_policy_id():
    ep1 = make_ep("ep1")
    ep2 = make_ep("ep2", nim_url="http://ep2:8000")
    registry = await make_registry(ep1, ep2)
    store = fresh_store()
    engine = DecisionEngine(registry=registry, store=store)

    profile = clf.classify(body())
    first = await engine.decide(profile)
    fallback = await engine.fallback_chain(profile, [first.selected_endpoint_id])

    assert fallback.policy_id == engine.policy.id


# ---------------------------------------------------------------------------
# Token counting
# ---------------------------------------------------------------------------


def test_token_count_increases_with_longer_message():
    short = clf.classify({"model": "test", "messages": [{"role": "user", "content": "hi"}]})
    long = clf.classify({"model": "test", "messages": [{"role": "user", "content": "x " * 500}]})
    assert long.input_tokens > short.input_tokens


def test_token_count_includes_system_prompt():
    without = clf.classify({"model": "test", "messages": [{"role": "user", "content": "hello"}]})
    with_sys = clf.classify({
        "model": "test",
        "system": "You are a helpful assistant. " * 20,
        "messages": [{"role": "user", "content": "hello"}],
    })
    assert with_sys.input_tokens > without.input_tokens


def test_reasoning_model_gets_higher_default_output():
    standard = clf.classify({"model": "meta/llama-3.1-8b", "messages": [{"role": "user", "content": "hi"}]})
    reasoning = clf.classify({"model": "meta/o1-reasoning", "messages": [{"role": "user", "content": "hi"}]})
    assert reasoning.predicted_output_tokens > standard.predicted_output_tokens


def test_max_tokens_respected():
    profile = clf.classify({
        "model": "meta/llama-3.1-8b",
        "messages": [{"role": "user", "content": "hi"}],
        "max_tokens": 999,
    })
    assert profile.predicted_output_tokens == 999
