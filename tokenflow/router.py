"""
Scoring and decision engine — the core of TokenFlow Router.

For each request, scores every candidate endpoint using a weighted
utility function and selects the best-fit route.
"""

from __future__ import annotations

import math
import time
from typing import Optional

import structlog

from tokenflow.models import (
    CandidateScore,
    CostClass,
    EndpointHealth,
    EndpointProfile,
    EndpointTelemetry,
    GPUClass,
    LatencyClass,
    PriorityTier,
    RequestProfile,
    RouteDecision,
    RouteOutcome,
    RoutingPolicy,
    WorkloadType,
)
from tokenflow.registry import EndpointRegistry
from tokenflow.telemetry import TelemetryStore

logger = structlog.get_logger(__name__)

# GPU tier ordering — higher = more powerful
_GPU_TIER: dict[GPUClass, int] = {
    GPUClass.H100: 5,
    GPUClass.A100: 4,
    GPUClass.L40S: 3,
    GPUClass.L40: 3,
    GPUClass.A10G: 2,
    GPUClass.L4: 2,
    GPUClass.RTX4090: 2,
    GPUClass.RTX3090: 1,
    GPUClass.UNKNOWN: 2,
}

# Cost tier ordering — lower = cheaper
_COST_TIER: dict[CostClass, float] = {
    CostClass.ECONOMY: 1.0,
    CostClass.STANDARD: 2.5,
    CostClass.PREMIUM: 6.0,
}

# Fallback SLO targets when policy doesn't specify
_DEFAULT_SLO_TTFT_MS = 500.0
_DEFAULT_SLO_ITL_MS = 50.0
_DEFAULT_SLO_E2E_MS = 5000.0


def _clamp(val: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return max(lo, min(hi, val))


class ScoringEngine:
    """
    Scores a single candidate endpoint for a given request.

    Utility(e) = w_slo * SLOScore
               + w_cost * CostScore
               + w_queue * QueueScore
               + w_gpu * GPUAffinityScore
               + w_model * ModelFitScore
               + w_reliability * ReliabilityScore
    """

    def __init__(self, policy: RoutingPolicy, store: TelemetryStore) -> None:
        self.policy = policy
        self.store = store

    # ------------------------------------------------------------------
    # Hard constraint checks
    # ------------------------------------------------------------------

    def hard_reject(
        self, ep: EndpointProfile, req: RequestProfile
    ) -> Optional[str]:
        """Return rejection reason if endpoint cannot serve this request, else None."""
        if not ep.enabled:
            return "endpoint_disabled"
        if ep.health == EndpointHealth.UNHEALTHY:
            return "endpoint_unhealthy"

        # Model compatibility
        from tokenflow.registry import EndpointRegistry
        if not EndpointRegistry._model_matches(ep.model_name, req.model_requested):
            return "model_mismatch"

        # Context window
        total_tokens = req.input_tokens + req.predicted_output_tokens
        if total_tokens > ep.max_context_tokens:
            return f"context_exceeded({total_tokens}>{ep.max_context_tokens})"

        # Streaming support
        if req.streaming and not ep.supports_streaming:
            return "streaming_not_supported"

        # Reasoning support
        if req.workload_type == WorkloadType.REASONING and not ep.supports_reasoning:
            return "reasoning_not_supported"

        # Queue depth hard ceiling
        tel = self.store.get(ep.id)
        if tel and tel.queue_depth > self.policy.max_queue_depth:
            return f"queue_full({tel.queue_depth}>{self.policy.max_queue_depth})"

        # Error rate ceiling
        if tel and tel.error_rate > self.policy.max_error_rate:
            return f"error_rate_too_high({tel.error_rate:.2f})"

        # Premium tier: only allow cost_class=premium or standard
        if req.priority_tier == PriorityTier.BATCH:
            if ep.cost_class == CostClass.PREMIUM:
                return "batch_request_on_premium_lane"

        return None

    # ------------------------------------------------------------------
    # Individual score components
    # ------------------------------------------------------------------

    def slo_score(self, ep: EndpointProfile, req: RequestProfile) -> tuple[float, float, float, float]:
        """Returns (score, est_ttft, est_itl, est_e2e) — all in ms."""
        tel = self.store.get(ep.id)

        # Estimate latencies — use telemetry if fresh, else heuristic
        if tel and not self.store.is_stale(ep.id):
            est_ttft = max(tel.p95_ttft_ms, 1.0)
            est_itl = max(tel.p95_itl_ms, 1.0)
            est_e2e = max(tel.p95_e2e_ms, 1.0)
        else:
            # Conservative heuristic based on GPU tier and token count
            gpu_tier = _GPU_TIER.get(ep.gpu_name, 2)
            base_ttft = 800.0 / gpu_tier
            base_itl = 30.0 / gpu_tier
            base_e2e = base_ttft + req.predicted_output_tokens * base_itl
            est_ttft = base_ttft * (1 + req.input_tokens / 4096)
            est_itl = base_itl
            est_e2e = max(base_e2e, est_ttft + est_itl * req.predicted_output_tokens)

        slo_ttft = self.policy.slo_ttft_ms or _DEFAULT_SLO_TTFT_MS
        slo_itl = self.policy.slo_itl_ms or _DEFAULT_SLO_ITL_MS
        slo_e2e = self.policy.slo_e2e_ms or _DEFAULT_SLO_E2E_MS

        # Weight by workload type
        if req.workload_type == WorkloadType.PREFILL_HEAVY:
            w1, w2, w3 = 0.6, 0.1, 0.3
        elif req.workload_type == WorkloadType.DECODE_HEAVY:
            w1, w2, w3 = 0.1, 0.6, 0.3
        elif req.workload_type == WorkloadType.REASONING:
            w1, w2, w3 = 0.3, 0.2, 0.5
        else:  # balanced
            w1, w2, w3 = 0.35, 0.3, 0.35

        score = (
            w1 * _clamp(slo_ttft / est_ttft)
            + w2 * _clamp(slo_itl / est_itl)
            + w3 * _clamp(slo_e2e / est_e2e)
        )
        return _clamp(score), est_ttft, est_itl, est_e2e

    def cost_score(self, ep: EndpointProfile, req: RequestProfile) -> tuple[float, float]:
        """Returns (score, estimated_usd)."""
        # Estimate tokens processed
        total_tokens = req.input_tokens + req.predicted_output_tokens
        # Rough GPU-hours per request
        gpu_tier = _GPU_TIER.get(ep.gpu_name, 2)
        tokens_per_sec = max(200 * gpu_tier, 100)
        request_duration_s = total_tokens / tokens_per_sec
        gpu_hours = request_duration_s / 3600 * ep.gpu_count
        est_cost = gpu_hours * ep.cost_per_gpu_hour

        # Normalise: economy=1.0, premium ~6x more expensive
        cost_tier = _COST_TIER.get(ep.cost_class, 2.5)
        # Higher score = cheaper — invert normalised cost
        norm_cost = _clamp(cost_tier / 6.0)  # 0=premium, 1=economy (flipped)
        score = 1.0 - norm_cost

        # Budget sensitivity modulates how much cost matters
        # (already applied as weight at utility level, but we also scale score)
        score = score * (1.0 - req.budget_sensitivity * 0.5) + req.budget_sensitivity * 0.5 * (1.0 - norm_cost)

        return _clamp(score), est_cost

    def queue_score(self, ep: EndpointProfile) -> float:
        tel = self.store.get(ep.id)
        if not tel or self.store.is_stale(ep.id):
            return 0.5  # neutral if unknown

        # Saturation score is 0=idle, 1=full
        sat = _clamp(tel.saturation_score)
        # queue_depth relative to ceiling
        queue_pressure = _clamp(tel.queue_depth / max(self.policy.max_queue_depth, 1))
        combined = max(sat, queue_pressure)
        return 1.0 - combined

    def gpu_affinity_score(
        self, ep: EndpointProfile, req: RequestProfile
    ) -> float:
        """Score GPU fit for the request's workload type."""
        gpu_tier = _GPU_TIER.get(ep.gpu_name, 2)

        if req.priority_tier == PriorityTier.PREMIUM:
            # Premium requests want highest-tier GPUs
            return _clamp(gpu_tier / 5.0)

        if req.workload_type == WorkloadType.PREFILL_HEAVY:
            # Needs strong prefill — higher tier helps more
            return _clamp(gpu_tier / 5.0)

        if req.workload_type == WorkloadType.DECODE_HEAVY:
            # Decode is less GPU-tier-sensitive; mid-range is fine
            # Penalise over-provisioning premium GPUs on cheap decode
            over_provision_penalty = 0.1 if ep.cost_class == CostClass.PREMIUM else 0.0
            return _clamp(gpu_tier / 5.0 * 0.7 - over_provision_penalty)

        if req.workload_type == WorkloadType.REASONING:
            return _clamp(gpu_tier / 5.0)

        # Balanced — moderate preference for higher tier
        return _clamp(gpu_tier / 5.0 * 0.8)

    def model_fit_score(self, ep: EndpointProfile, req: RequestProfile) -> float:
        """Exact model match > family match > wildcard."""
        req_lower = req.model_requested.lower()
        served_lower = ep.model_name.lower()
        if served_lower == req_lower:
            return 1.0
        # Family prefix match
        if req_lower and served_lower.startswith(req_lower.split("-")[0]):
            return 0.75
        # Wildcard
        if req.model_requested in ("any", "*", ""):
            return 0.5
        return 0.3

    def reliability_score(self, ep: EndpointProfile) -> float:
        tel = self.store.get(ep.id)
        if not tel or self.store.is_stale(ep.id):
            # Health degraded = lower score
            if ep.health == EndpointHealth.DEGRADED:
                return 0.4
            return 0.6  # neutral/unknown

        err_penalty = _clamp(tel.error_rate / self.policy.max_error_rate)
        # Tail latency instability — p95 >> p50 signals instability
        if tel.p50_ttft_ms > 0:
            ttft_spread = tel.p95_ttft_ms / max(tel.p50_ttft_ms, 1)
            spread_penalty = _clamp((ttft_spread - 1.5) / 3.0)  # penalise >1.5x spread
        else:
            spread_penalty = 0.0

        return _clamp(1.0 - 0.6 * err_penalty - 0.4 * spread_penalty)

    # ------------------------------------------------------------------
    # Full utility
    # ------------------------------------------------------------------

    def score(self, ep: EndpointProfile, req: RequestProfile) -> CandidateScore:
        rejection = self.hard_reject(ep, req)
        if rejection:
            return CandidateScore(
                endpoint_id=ep.id,
                endpoint_name=ep.name,
                utility_score=0.0,
                slo_score=0.0,
                cost_score=0.0,
                queue_score=0.0,
                gpu_affinity_score=0.0,
                model_fit_score=0.0,
                reliability_score=0.0,
                estimated_ttft_ms=9999.0,
                estimated_itl_ms=9999.0,
                estimated_e2e_ms=9999.0,
                estimated_cost_usd=9999.0,
                hard_rejected=True,
                rejection_reason=rejection,
            )

        slo_s, est_ttft, est_itl, est_e2e = self.slo_score(ep, req)
        cost_s, est_cost = self.cost_score(ep, req)
        queue_s = self.queue_score(ep)
        gpu_s = self.gpu_affinity_score(ep, req)
        model_s = self.model_fit_score(ep, req)
        rel_s = self.reliability_score(ep)

        p = self.policy
        utility = (
            p.slo_weight * slo_s
            + p.cost_weight * cost_s
            + p.queue_weight * queue_s
            + p.gpu_affinity_weight * gpu_s
            + p.model_fit_weight * model_s
            + p.reliability_weight * rel_s
        )

        return CandidateScore(
            endpoint_id=ep.id,
            endpoint_name=ep.name,
            utility_score=_clamp(utility),
            slo_score=slo_s,
            cost_score=cost_s,
            queue_score=queue_s,
            gpu_affinity_score=gpu_s,
            model_fit_score=model_s,
            reliability_score=rel_s,
            estimated_ttft_ms=est_ttft,
            estimated_itl_ms=est_itl,
            estimated_e2e_ms=est_e2e,
            estimated_cost_usd=est_cost,
            hard_rejected=False,
        )


class DecisionEngine:
    """
    Orchestrates candidate scoring and selects the winning endpoint.

    Supports:
    - Multi-candidate scoring
    - Fallback chain (next-best if top choice fails)
    - Policy preset application (latency-first / balanced / cost-first)
    """

    def __init__(
        self,
        registry: EndpointRegistry,
        store: TelemetryStore,
    ) -> None:
        self.registry = registry
        self.store = store
        self._policy: RoutingPolicy = RoutingPolicy(name="default")

    def set_policy(self, policy: RoutingPolicy) -> None:
        self._policy = _apply_preset(policy)

    @property
    def policy(self) -> RoutingPolicy:
        return self._policy

    async def decide(self, req: RequestProfile) -> RouteDecision:
        t_start = time.perf_counter()

        candidates = await self.registry.find_by_model(req.model_requested)
        if not candidates:
            # Fall back to all enabled endpoints
            candidates = await self.registry.list_all(enabled_only=True)

        engine = ScoringEngine(self._policy, self.store)
        scores = [engine.score(ep, req) for ep in candidates]

        hard_rejections = [
            {"endpoint_id": s.endpoint_id, "reason": s.rejection_reason or ""}
            for s in scores
            if s.hard_rejected
        ]

        valid = sorted(
            [s for s in scores if not s.hard_rejected],
            key=lambda s: s.utility_score,
            reverse=True,
        )

        decision_ms = (time.perf_counter() - t_start) * 1000

        if not valid:
            logger.warning(
                "no_valid_endpoint",
                request_id=req.request_id,
                model=req.model_requested,
                rejections=len(hard_rejections),
            )
            return RouteDecision(
                request_id=req.request_id,
                selected_endpoint_id=None,
                selected_endpoint_name=None,
                candidate_scores=scores,
                hard_rejections=hard_rejections,
                policy_id=self._policy.id,
                outcome=RouteOutcome.REJECTED,
                decision_latency_ms=decision_ms,
            )

        winner = valid[0]
        logger.info(
            "route_decided",
            request_id=req.request_id,
            endpoint=winner.endpoint_name,
            utility=round(winner.utility_score, 3),
            est_ttft_ms=round(winner.estimated_ttft_ms, 1),
            est_e2e_ms=round(winner.estimated_e2e_ms, 1),
            decision_ms=round(decision_ms, 2),
        )

        return RouteDecision(
            request_id=req.request_id,
            selected_endpoint_id=winner.endpoint_id,
            selected_endpoint_name=winner.endpoint_name,
            candidate_scores=scores,
            hard_rejections=hard_rejections,
            policy_id=self._policy.id,
            estimated_cost_usd=winner.estimated_cost_usd,
            predicted_ttft_ms=winner.estimated_ttft_ms,
            predicted_itl_ms=winner.estimated_itl_ms,
            predicted_e2e_ms=winner.estimated_e2e_ms,
            outcome=RouteOutcome.SUCCESS,
            decision_latency_ms=decision_ms,
        )

    async def fallback_chain(
        self, req: RequestProfile, failed_ids: list[str]
    ) -> RouteDecision:
        """Re-run decision excluding already-failed endpoints."""
        candidates = await self.registry.list_all(enabled_only=True)
        candidates = [c for c in candidates if c.id not in failed_ids]

        engine = ScoringEngine(self._policy, self.store)
        scores = [engine.score(ep, req) for ep in candidates]
        valid = sorted(
            [s for s in scores if not s.hard_rejected],
            key=lambda s: s.utility_score,
            reverse=True,
        )

        if not valid:
            return RouteDecision(
                request_id=req.request_id,
                selected_endpoint_id=None,
                selected_endpoint_name=None,
                candidate_scores=scores,
                outcome=RouteOutcome.FAILED,
                fallback_used=True,
                fallback_count=len(failed_ids),
            )

        winner = valid[0]
        return RouteDecision(
            request_id=req.request_id,
            selected_endpoint_id=winner.endpoint_id,
            selected_endpoint_name=winner.endpoint_name,
            candidate_scores=scores,
            policy_id=self._policy.id,
            estimated_cost_usd=winner.estimated_cost_usd,
            predicted_ttft_ms=winner.estimated_ttft_ms,
            predicted_itl_ms=winner.estimated_itl_ms,
            predicted_e2e_ms=winner.estimated_e2e_ms,
            outcome=RouteOutcome.FALLBACK_USED,
            fallback_used=True,
            fallback_count=len(failed_ids),
        )


def _apply_preset(policy: RoutingPolicy) -> RoutingPolicy:
    """Apply a named preset to override policy weights."""
    presets = {
        "latency-first": dict(
            slo_weight=0.50,
            cost_weight=0.05,
            queue_weight=0.20,
            gpu_affinity_weight=0.15,
            model_fit_weight=0.05,
            reliability_weight=0.05,
            slo_ttft_ms=300.0,
            slo_itl_ms=30.0,
            slo_e2e_ms=3000.0,
        ),
        "balanced": dict(
            slo_weight=0.30,
            cost_weight=0.20,
            queue_weight=0.15,
            gpu_affinity_weight=0.15,
            model_fit_weight=0.10,
            reliability_weight=0.10,
        ),
        "cost-first": dict(
            slo_weight=0.15,
            cost_weight=0.45,
            queue_weight=0.10,
            gpu_affinity_weight=0.10,
            model_fit_weight=0.10,
            reliability_weight=0.10,
            slo_ttft_ms=2000.0,
            slo_itl_ms=100.0,
            slo_e2e_ms=30000.0,
        ),
    }
    if policy.preset in presets:
        updates = presets[policy.preset]
        return policy.model_copy(update=updates)
    return policy
