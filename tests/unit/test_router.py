"""Unit tests for the scoring and decision engine."""

import pytest
from tokenflow.classifier import RequestClassifier
from tokenflow.models import (
    CostClass,
    EndpointHealth,
    EndpointProfile,
    GPUClass,
    PriorityTier,
    RouteOutcome,
    RoutingPolicy,
    TelemetryUpdate,
    WorkloadType,
)
from tokenflow.registry import EndpointRegistry
from tokenflow.router import DecisionEngine, ScoringEngine, _apply_preset
from tokenflow.telemetry import TelemetryStore


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def make_endpoint(
    name="test-ep",
    model="meta/llama-3.1-8b-instruct",
    gpu=GPUClass.L40S,
    cost_class=CostClass.STANDARD,
    enabled=True,
    health=EndpointHealth.HEALTHY,
    supports_reasoning=False,
) -> EndpointProfile:
    ep = EndpointProfile(
        name=name,
        nim_url=f"http://{name}:8000",
        model_name=model,
        gpu_name=gpu,
        cost_class=cost_class,
        cost_per_gpu_hour=3.0,
        supports_reasoning=supports_reasoning,
        max_context_tokens=16384,
        enabled=enabled,
        health=health,
    )
    return ep


def make_store_with_telemetry(endpoint_id: str, **kwargs) -> TelemetryStore:
    store = TelemetryStore()
    import asyncio
    update = TelemetryUpdate(endpoint_id=endpoint_id, **kwargs)
    asyncio.get_event_loop().run_until_complete(store.upsert(update))
    return store


clf = RequestClassifier()


# ---------------------------------------------------------------------------
# Preset tests
# ---------------------------------------------------------------------------


def test_latency_first_preset_weights():
    policy = RoutingPolicy(preset="latency-first")
    applied = _apply_preset(policy)
    assert applied.slo_weight == 0.50
    assert applied.cost_weight == 0.05
    assert applied.slo_ttft_ms == 300.0


def test_cost_first_preset_weights():
    policy = RoutingPolicy(preset="cost-first")
    applied = _apply_preset(policy)
    assert applied.cost_weight == 0.45
    assert applied.slo_ttft_ms == 2000.0


def test_balanced_preset_unchanged():
    policy = RoutingPolicy(preset="balanced")
    applied = _apply_preset(policy)
    assert applied.slo_weight == 0.30
    assert applied.cost_weight == 0.20


# ---------------------------------------------------------------------------
# Hard rejection tests
# ---------------------------------------------------------------------------


def test_disabled_endpoint_rejected():
    ep = make_endpoint(enabled=False)
    store = TelemetryStore()
    engine = ScoringEngine(RoutingPolicy(), store)
    body = {"model": "meta/llama-3.1-8b-instruct", "messages": [{"role": "user", "content": "hi"}]}
    profile = clf.classify(body)
    score = engine.score(ep, profile)
    assert score.hard_rejected
    assert "disabled" in score.rejection_reason


def test_unhealthy_endpoint_rejected():
    ep = make_endpoint(health=EndpointHealth.UNHEALTHY)
    store = TelemetryStore()
    engine = ScoringEngine(RoutingPolicy(), store)
    body = {"model": "meta/llama-3.1-8b-instruct", "messages": [{"role": "user", "content": "hi"}]}
    profile = clf.classify(body)
    score = engine.score(ep, profile)
    assert score.hard_rejected


def test_model_mismatch_rejected():
    ep = make_endpoint(model="meta/llama-3.1-70b-instruct")
    store = TelemetryStore()
    engine = ScoringEngine(RoutingPolicy(), store)
    body = {"model": "mistral/mistral-7b", "messages": [{"role": "user", "content": "hi"}]}
    profile = clf.classify(body)
    score = engine.score(ep, profile)
    assert score.hard_rejected
    assert "model_mismatch" in score.rejection_reason


def test_context_exceeded_rejected():
    ep = make_endpoint()
    ep.max_context_tokens = 100
    store = TelemetryStore()
    engine = ScoringEngine(RoutingPolicy(), store)
    body = {
        "model": "meta/llama-3.1-8b-instruct",
        "messages": [{"role": "user", "content": "x " * 1000}],
        "max_tokens": 500,
    }
    profile = clf.classify(body)
    score = engine.score(ep, profile)
    assert score.hard_rejected
    assert "context_exceeded" in score.rejection_reason


def test_batch_request_rejected_on_premium_lane():
    ep = make_endpoint(cost_class=CostClass.PREMIUM)
    store = TelemetryStore()
    engine = ScoringEngine(RoutingPolicy(), store)
    body = {"model": "meta/llama-3.1-8b-instruct", "messages": [{"role": "user", "content": "x"}]}
    profile = clf.classify(body, priority_tier=PriorityTier.BATCH)
    score = engine.score(ep, profile)
    assert score.hard_rejected
    assert "premium_lane" in score.rejection_reason


# ---------------------------------------------------------------------------
# Utility score ordering
# ---------------------------------------------------------------------------


def test_h100_scores_higher_than_l4_for_premium():
    store = TelemetryStore()
    engine = ScoringEngine(RoutingPolicy(), store)
    ep_h100 = make_endpoint("h100", gpu=GPUClass.H100, cost_class=CostClass.PREMIUM)
    ep_l4 = make_endpoint("l4", gpu=GPUClass.L4, cost_class=CostClass.ECONOMY)
    body = {"model": "meta/llama-3.1-8b-instruct", "messages": [{"role": "user", "content": "hi"}]}
    profile = clf.classify(body, priority_tier=PriorityTier.PREMIUM)
    s_h100 = engine.score(ep_h100, profile)
    s_l4 = engine.score(ep_l4, profile)
    assert s_h100.utility_score > s_l4.utility_score


def test_l4_scores_higher_for_batch_cost_first():
    policy = _apply_preset(RoutingPolicy(preset="cost-first"))
    store = TelemetryStore()
    engine = ScoringEngine(policy, store)
    ep_h100 = make_endpoint("h100", gpu=GPUClass.H100, cost_class=CostClass.PREMIUM)
    ep_h100.cost_per_gpu_hour = 8.0
    ep_l4 = make_endpoint("l4", gpu=GPUClass.L4, cost_class=CostClass.ECONOMY)
    ep_l4.cost_per_gpu_hour = 0.8
    body = {"model": "meta/llama-3.1-8b-instruct", "messages": [{"role": "user", "content": "x"}]}
    profile = clf.classify(body, priority_tier=PriorityTier.BATCH)
    # L4 not rejected (batch allowed on economy)
    s_l4 = engine.score(ep_l4, profile)
    assert not s_l4.hard_rejected
    assert s_l4.cost_score > 0


# ---------------------------------------------------------------------------
# Decision engine
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_decision_selects_best_endpoint():
    registry = EndpointRegistry()
    store = TelemetryStore()

    ep1 = make_endpoint("ep1", gpu=GPUClass.H100, cost_class=CostClass.PREMIUM)
    ep2 = make_endpoint("ep2", gpu=GPUClass.L4, cost_class=CostClass.ECONOMY)

    from tokenflow.models import EndpointRegisterRequest
    for ep in [ep1, ep2]:
        req = EndpointRegisterRequest(**{
            k: getattr(ep, k) for k in EndpointRegisterRequest.model_fields
        })
        await registry.register(req)

    engine = DecisionEngine(registry=registry, store=store)
    engine.set_policy(_apply_preset(RoutingPolicy(preset="latency-first")))

    body = {
        "model": "meta/llama-3.1-8b-instruct",
        "messages": [{"role": "user", "content": "hi"}],
        "max_tokens": 64,
    }
    profile = clf.classify(body, priority_tier=PriorityTier.PREMIUM)
    decision = await engine.decide(profile)

    assert decision.selected_endpoint_id is not None
    assert decision.outcome == RouteOutcome.SUCCESS


@pytest.mark.asyncio
async def test_decision_returns_rejected_when_no_endpoints():
    registry = EndpointRegistry()
    store = TelemetryStore()
    engine = DecisionEngine(registry=registry, store=store)

    body = {"model": "nonexistent-model", "messages": [{"role": "user", "content": "hi"}]}
    profile = clf.classify(body)
    decision = await engine.decide(profile)

    assert decision.selected_endpoint_id is None
    assert decision.outcome == RouteOutcome.REJECTED
