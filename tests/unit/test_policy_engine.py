"""Unit tests for the policy engine."""

import pytest
from tokenflow.classifier import RequestClassifier
from tokenflow.models import (
    PolicyRule,
    PriorityTier,
    RoutingPolicy,
    TenantPolicy,
)
from tokenflow.policy_engine.engine import PolicyEngine

clf = RequestClassifier()


def make_engine_with_policy(**policy_kwargs) -> PolicyEngine:
    engine = PolicyEngine()
    engine.set_policy(RoutingPolicy(**policy_kwargs))
    return engine


@pytest.mark.asyncio
async def test_priority_override_for_tenant():
    engine = PolicyEngine()
    policy = RoutingPolicy(
        tenant_policies={
            "vip": TenantPolicy(
                tenant_id="vip",
                priority_tier_override=PriorityTier.PREMIUM,
            )
        }
    )
    engine.set_policy(policy)

    body = {"model": "llama3", "messages": [{"role": "user", "content": "hi"}]}
    profile = clf.classify(body, tenant_id="vip", priority_tier=PriorityTier.STANDARD)

    modified, actions = await engine.apply(profile)
    assert modified.priority_tier == PriorityTier.PREMIUM
    assert any("priority_override" in a for a in actions)


@pytest.mark.asyncio
async def test_rule_set_priority_on_burst():
    engine = PolicyEngine()
    policy = RoutingPolicy(
        rules=[
            PolicyRule(
                name="demote-burst",
                priority=10,
                conditions={"burst_class": "burst"},
                actions={"set_priority": "batch"},
            )
        ]
    )
    engine.set_policy(policy)

    body = {"model": "llama3", "messages": [{"role": "user", "content": "hi"}]}
    profile = clf.classify(body, current_tenant_rpm=200)  # triggers burst
    profile = profile.model_copy(update={"burst_class": "burst"})

    modified, actions = await engine.apply(profile)
    assert modified.priority_tier == PriorityTier.BATCH


@pytest.mark.asyncio
async def test_rule_set_budget_sensitivity():
    engine = PolicyEngine()
    policy = RoutingPolicy(
        rules=[
            PolicyRule(
                name="big-input-cost-aware",
                priority=10,
                conditions={"input_tokens_gt": 100},
                actions={"set_budget_sensitivity": "0.9"},
            )
        ]
    )
    engine.set_policy(policy)

    body = {
        "model": "llama3",
        "messages": [{"role": "user", "content": "x " * 500}],
    }
    profile = clf.classify(body)
    modified, actions = await engine.apply(profile)
    assert modified.budget_sensitivity == pytest.approx(0.9)


@pytest.mark.asyncio
async def test_rpm_throttle_demotes_to_batch():
    engine = PolicyEngine()
    policy = RoutingPolicy(
        tenant_policies={
            "free-tier": TenantPolicy(
                tenant_id="free-tier",
                max_rpm=10,
            )
        }
    )
    engine.set_policy(policy)

    # Simulate 20 requests to push over RPM cap
    body = {"model": "llama3", "messages": [{"role": "user", "content": "hi"}]}
    for _ in range(20):
        await engine.rpm_tracker.record("free-tier")

    profile = clf.classify(body, tenant_id="free-tier")
    modified, actions = await engine.apply(profile)
    assert modified.priority_tier == PriorityTier.BATCH
    assert any("rpm_throttle" in a for a in actions)


@pytest.mark.asyncio
async def test_no_rules_no_modification():
    engine = PolicyEngine()
    engine.set_policy(RoutingPolicy())

    body = {"model": "llama3", "messages": [{"role": "user", "content": "hi"}]}
    profile = clf.classify(body)
    original_priority = profile.priority_tier
    modified, actions = await engine.apply(profile)
    assert modified.priority_tier == original_priority
