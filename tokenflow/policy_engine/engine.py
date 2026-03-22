"""
Policy engine — evaluates tenant rules, budget caps, and policy DSL.

Policies are loaded from YAML and applied both as hard constraints
(reject/override) and as soft adjustments (weight overrides).
"""

from __future__ import annotations

import asyncio
from collections import defaultdict
from datetime import datetime, timezone
from typing import Any, Optional

import structlog
import yaml

from tokenflow.models import (
    PolicyRule,
    PriorityTier,
    RequestProfile,
    RoutingPolicy,
    TenantPolicy,
)

logger = structlog.get_logger(__name__)


# ---------------------------------------------------------------------------
# Budget tracker
# ---------------------------------------------------------------------------


class BudgetTracker:
    """
    Per-tenant rolling cost accumulator.
    Tracks USD spend within a sliding hour window.
    """

    def __init__(self) -> None:
        # (tenant_id) -> list of (timestamp_utc, cost_usd)
        self._records: dict[str, list[tuple[datetime, float]]] = defaultdict(list)
        self._lock = asyncio.Lock()

    async def record(self, tenant_id: str, cost_usd: float) -> None:
        async with self._lock:
            now = datetime.now(timezone.utc)
            self._records[tenant_id].append((now, cost_usd))
            self._prune(tenant_id)

    async def spent_last_hour(self, tenant_id: str) -> float:
        async with self._lock:
            self._prune(tenant_id)
            return sum(c for _, c in self._records[tenant_id])

    def _prune(self, tenant_id: str) -> None:
        now = datetime.now(timezone.utc)
        self._records[tenant_id] = [
            (ts, c)
            for ts, c in self._records[tenant_id]
            if (now - ts).total_seconds() < 3600
        ]


# ---------------------------------------------------------------------------
# RPM tracker
# ---------------------------------------------------------------------------


class RPMTracker:
    """Per-tenant rolling request-per-minute tracker."""

    def __init__(self) -> None:
        self._records: dict[str, list[datetime]] = defaultdict(list)
        self._lock = asyncio.Lock()

    async def record(self, tenant_id: str) -> None:
        async with self._lock:
            now = datetime.now(timezone.utc)
            self._records[tenant_id].append(now)
            self._prune(tenant_id)

    async def current_rpm(self, tenant_id: str) -> float:
        async with self._lock:
            self._prune(tenant_id)
            return len(self._records[tenant_id])

    def _prune(self, tenant_id: str) -> None:
        now = datetime.now(timezone.utc)
        self._records[tenant_id] = [
            ts for ts in self._records[tenant_id]
            if (now - ts).total_seconds() < 60
        ]


# ---------------------------------------------------------------------------
# Policy loader
# ---------------------------------------------------------------------------


def load_policy_from_yaml(path: str) -> RoutingPolicy:
    """Load a RoutingPolicy from a YAML file."""
    with open(path) as f:
        data = yaml.safe_load(f)

    rules = []
    for r in data.pop("rules", []):
        rules.append(PolicyRule(**r))

    tenant_policies = {}
    for tid, tp in data.pop("tenant_policies", {}).items():
        tenant_policies[tid] = TenantPolicy(tenant_id=tid, **tp)

    return RoutingPolicy(
        rules=rules,
        tenant_policies=tenant_policies,
        **{k: v for k, v in data.items() if k in RoutingPolicy.model_fields},
    )


# ---------------------------------------------------------------------------
# Policy engine
# ---------------------------------------------------------------------------


class PolicyEngine:
    """
    Applies routing policy rules and tenant constraints to a request.

    Returns an (optionally) modified RequestProfile and a list of
    policy violations / overrides for logging.
    """

    def __init__(self) -> None:
        self._policy: RoutingPolicy = RoutingPolicy(name="default")
        self.budget_tracker = BudgetTracker()
        self.rpm_tracker = RPMTracker()

    def set_policy(self, policy: RoutingPolicy) -> None:
        self._policy = policy
        logger.info("policy_loaded", policy_name=policy.name, rules=len(policy.rules))

    @property
    def policy(self) -> RoutingPolicy:
        return self._policy

    async def apply(
        self, profile: RequestProfile
    ) -> tuple[RequestProfile, list[str]]:
        """
        Apply policy to a request profile.

        Returns:
            (modified_profile, list_of_applied_actions)
        """
        actions_applied: list[str] = []
        tenant_id = profile.tenant_id

        # --- Record RPM ---
        await self.rpm_tracker.record(tenant_id)
        current_rpm = await self.rpm_tracker.current_rpm(tenant_id)

        # --- Tenant-level policy ---
        tenant_policy = self._policy.tenant_policies.get(tenant_id)
        if tenant_policy:
            profile, acts = await self._apply_tenant_policy(
                profile, tenant_policy, current_rpm
            )
            actions_applied.extend(acts)

        # --- Rule-based DSL ---
        sorted_rules = sorted(
            [r for r in self._policy.rules if r.enabled],
            key=lambda r: r.priority,
        )
        for rule in sorted_rules:
            if self._rule_matches(rule, profile, current_rpm):
                profile, acts = self._apply_rule_actions(rule, profile)
                actions_applied.extend(acts)
                logger.debug("rule_applied", rule=rule.name, actions=acts)

        return profile, actions_applied

    async def _apply_tenant_policy(
        self,
        profile: RequestProfile,
        tp: TenantPolicy,
        current_rpm: float,
    ) -> tuple[RequestProfile, list[str]]:
        actions: list[str] = []

        # Priority tier override
        if tp.priority_tier_override is not None:
            profile = profile.model_copy(
                update={"priority_tier": tp.priority_tier_override}
            )
            actions.append(f"priority_override:{tp.priority_tier_override}")

        # Cost weight override
        if tp.cost_weight_override is not None:
            profile = profile.model_copy(
                update={"budget_sensitivity": tp.cost_weight_override}
            )
            actions.append(f"cost_weight_override:{tp.cost_weight_override}")

        # RPM cap enforcement
        if tp.max_rpm is not None and current_rpm > tp.max_rpm:
            # Demote to batch tier
            profile = profile.model_copy(
                update={"priority_tier": PriorityTier.BATCH}
            )
            actions.append(f"rpm_throttle:{current_rpm:.0f}>{tp.max_rpm}")

        # Budget check
        if tp.budget_usd_per_hour is not None:
            spent = await self.budget_tracker.spent_last_hour(profile.tenant_id)
            if spent >= tp.budget_usd_per_hour:
                profile = profile.model_copy(
                    update={
                        "priority_tier": PriorityTier.BATCH,
                        "budget_sensitivity": 1.0,
                    }
                )
                actions.append(f"budget_exhausted:{spent:.4f}>={tp.budget_usd_per_hour}")

        return profile, actions

    @staticmethod
    def _rule_matches(
        rule: PolicyRule, profile: RequestProfile, current_rpm: float
    ) -> bool:
        cond = rule.conditions
        checks = {
            "tenant_id": lambda v: profile.tenant_id == v,
            "priority_tier": lambda v: profile.priority_tier.value == v,
            "workload_type": lambda v: profile.workload_type.value == v,
            "latency_class": lambda v: profile.latency_class.value == v,
            "input_tokens_gt": lambda v: profile.input_tokens > int(v),
            "input_tokens_lt": lambda v: profile.input_tokens < int(v),
            "output_tokens_gt": lambda v: profile.predicted_output_tokens > int(v),
            "output_tokens_lt": lambda v: profile.predicted_output_tokens < int(v),
            "model_contains": lambda v: v.lower() in profile.model_requested.lower(),
            "rpm_gt": lambda v: current_rpm > float(v),
            "burst_class": lambda v: profile.burst_class == v,
            "budget_sensitivity_gt": lambda v: profile.budget_sensitivity > float(v),
        }
        for key, value in cond.items():
            checker = checks.get(key)
            if checker and not checker(value):
                return False
        return True

    @staticmethod
    def _apply_rule_actions(
        rule: PolicyRule, profile: RequestProfile
    ) -> tuple[RequestProfile, list[str]]:
        actions = rule.actions
        applied: list[str] = []
        updates: dict[str, Any] = {}

        if "set_priority" in actions:
            updates["priority_tier"] = PriorityTier(actions["set_priority"])
            applied.append(f"set_priority:{actions['set_priority']}")

        if "set_budget_sensitivity" in actions:
            updates["budget_sensitivity"] = float(actions["set_budget_sensitivity"])
            applied.append(f"set_budget_sensitivity:{actions['set_budget_sensitivity']}")

        if "set_latency_class" in actions:
            from tokenflow.models import LatencyClass
            updates["latency_class"] = LatencyClass(actions["set_latency_class"])
            applied.append(f"set_latency_class:{actions['set_latency_class']}")

        if updates:
            profile = profile.model_copy(update=updates)

        return profile, applied

    async def record_cost(self, tenant_id: str, cost_usd: float) -> None:
        await self.budget_tracker.record(tenant_id, cost_usd)
