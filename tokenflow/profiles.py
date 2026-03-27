"""
Dynamic backend profile management.

Allows pre-defining backend profile *templates* that are lazily activated
when matching workload types are detected. The current implementation now
supports three important behaviors:

1. **Dormant backend catalog** — templates describe backends that may exist but
   are not currently running/registered.
2. **Single-owner model residency** — when enabled, activating a template can
   deactivate sibling templates for the same model so the router does not keep
   duplicate live copies everywhere by default.
3. **Idle deactivation** — activated templates can be unregistered after a period
   of inactivity so only the needed backend stays live.

This is intentionally lightweight orchestration: TokenFlow does not yet launch
containers or VM instances itself, but it can control which endpoint templates
become active in the routing plane, which is the repo's current abstraction for
"start only what is needed".
"""

from __future__ import annotations

import asyncio
import uuid
from datetime import datetime, timedelta
from typing import Any

import structlog
from pydantic import BaseModel, Field

from tokenflow.benchmarks import backend_affinity
from tokenflow.models import (
    BackendType,
    CostClass,
    EndpointRegisterRequest,
    GPUClass,
    OptimizationTarget,
    RequestProfile,
    WorkloadType,
)

logger = structlog.get_logger(__name__)

_GPU_MEMORY_GB: dict[GPUClass, float] = {
    GPUClass.B200: 192.0,
    GPUClass.H200: 141.0,
    GPUClass.H100: 80.0,
    GPUClass.A100: 80.0,
    GPUClass.L40S: 48.0,
    GPUClass.L40: 48.0,
    GPUClass.RTX_PRO_6000: 96.0,
    GPUClass.A10G: 24.0,
    GPUClass.L4: 24.0,
    GPUClass.RTX4090: 24.0,
    GPUClass.RTX3090: 24.0,
    GPUClass.RTX_LAPTOP: 12.0,
    GPUClass.CPU: 0.0,
    GPUClass.UNKNOWN: 24.0,
}


class BackendProfileTemplate(BaseModel):
    """
    A backend endpoint template that can be activated on-demand.

    When a request with a matching workload type arrives, this template can be
    registered as a live endpoint. Templates represent *available capacity*,
    not necessarily live endpoints.
    """

    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    name: str
    nim_url: str
    backend_type: BackendType = BackendType.NIM
    model_name: str
    model_family: str = ""
    model_version: str = "latest"
    gpu_name: GPUClass = GPUClass.UNKNOWN
    gpu_count: int = 1
    region: str = "us-east-1"
    cost_class: CostClass = CostClass.STANDARD
    max_context_tokens: int = 8192
    supports_streaming: bool = True
    supports_reasoning: bool = False
    tenant_tags: list[str] = Field(default_factory=list)
    capability_flags: dict[str, Any] = Field(default_factory=dict)
    cost_per_gpu_hour: float = 3.0

    # Profile-specific
    workload_affinity: list[WorkloadType] = Field(
        default_factory=list,
        description=(
            "Workload types this profile is optimised for. "
            "Empty list means activate for any workload."
        ),
    )
    activation_model_names: list[str] = Field(
        default_factory=list,
        description="Optional list of model names/families this template may activate for.",
    )
    auto_activate: bool = Field(
        default=True,
        description="Automatically activate when a matching request arrives.",
    )
    exclusive_model_residency: bool = Field(
        default=True,
        description="If true, deactivate sibling templates for the same model when this template activates.",
    )
    idle_ttl_seconds: int = Field(
        default=900,
        ge=60,
        description="Deactivate the template after this many idle seconds.",
    )
    min_live_seconds: int = Field(
        default=180,
        ge=0,
        description="Minimum dwell time after activation before deactivation is allowed.",
    )
    deactivation_buffer_seconds: int = Field(
        default=120,
        ge=0,
        description="Additional quiet-time buffer after idle TTL before deactivation.",
    )

    # State
    activated: bool = False
    activated_endpoint_id: str | None = None
    activated_at: datetime | None = None
    last_used_at: datetime | None = None
    activation_count: int = 0
    created_at: datetime = Field(default_factory=datetime.utcnow)


class ProfileManager:
    """Manages backend profile templates and their activation/deactivation."""

    def __init__(self) -> None:
        self._templates: dict[str, BackendProfileTemplate] = {}
        self._lock = asyncio.Lock()
        self._registry = None
        self._telemetry_collector = None
        self._telemetry_store = None

    def attach(self, registry, telemetry_collector, telemetry_store=None) -> None:
        self._registry = registry
        self._telemetry_collector = telemetry_collector
        self._telemetry_store = telemetry_store

    # ------------------------------------------------------------------
    # Template CRUD
    # ------------------------------------------------------------------

    async def add_template(self, template: BackendProfileTemplate) -> BackendProfileTemplate:
        async with self._lock:
            self._templates[template.id] = template
            logger.info(
                "profile_template_registered",
                template_id=template.id,
                name=template.name,
                backend=template.backend_type.value,
                workload_affinity=[w.value for w in template.workload_affinity],
                exclusive_model_residency=template.exclusive_model_residency,
                idle_ttl_seconds=template.idle_ttl_seconds,
                min_live_seconds=template.min_live_seconds,
                deactivation_buffer_seconds=template.deactivation_buffer_seconds,
            )
            return template

    async def get_template(self, template_id: str) -> BackendProfileTemplate | None:
        return self._templates.get(template_id)

    async def list_templates(self) -> list[BackendProfileTemplate]:
        return list(self._templates.values())

    async def delete_template(self, template_id: str) -> bool:
        async with self._lock:
            template = self._templates.get(template_id)
        if template is None:
            return False
        if template.activated:
            await self._deactivate(template)
        async with self._lock:
            self._templates.pop(template_id, None)
        return True

    # ------------------------------------------------------------------
    # Request-driven orchestration
    # ------------------------------------------------------------------

    async def ensure_capacity_for_request(self, profile: RequestProfile) -> None:
        """
        Ensure there is at least one live endpoint for the requested model.

        If no matching live endpoint exists, synchronously activate the best dormant
        template for this request so the current request can route successfully.
        """
        if self._registry is None:
            return
        live = await self._registry.find_by_model(profile.model_requested)
        if live:
            return
        candidate = await self._best_template_for_request(profile, dormant_only=True)
        if candidate is None:
            return
        await self._activate(candidate, profile)

    async def maybe_activate_for_request(self, profile: RequestProfile) -> None:
        """Asynchronously pre-warm the best dormant template for future requests."""
        candidate = await self._best_template_for_request(profile, dormant_only=True)
        if candidate is None:
            return
        asyncio.create_task(self._activate(candidate, profile))

    async def record_endpoint_use(self, endpoint_id: str) -> None:
        async with self._lock:
            for template in self._templates.values():
                if template.activated_endpoint_id == endpoint_id:
                    template.last_used_at = datetime.utcnow()
                    break

    async def capacity_plan_for_request(
        self,
        profile: RequestProfile,
        selected_endpoint_id: str,
    ) -> dict[str, Any]:
        """Describe the active backend and duplicate live lanes that may be turned down."""
        if self._registry is None:
            return {
                "active_backend": None,
                "turn_down_candidates": [],
            }

        live = await self._registry.find_by_model(profile.model_requested)
        selected = next((ep for ep in live if ep.id == selected_endpoint_id), None)
        template_by_endpoint = {
            t.activated_endpoint_id: t
            for t in await self.list_templates()
            if t.activated_endpoint_id
        }
        candidates: list[dict[str, Any]] = []
        for ep in live:
            if ep.id == selected_endpoint_id:
                continue
            template = template_by_endpoint.get(ep.id)
            eligible = await self._eligible_for_deactivation(template) if template else False
            if not eligible:
                continue
            candidates.append(
                {
                    "endpoint": ep.name,
                    "backend": ep.backend_type.value,
                    "gpu": ep.gpu_name.value,
                    "reason": "duplicate_model_residency",
                    "buffered_by": {
                        "idle_ttl_seconds": template.idle_ttl_seconds,
                        "min_live_seconds": template.min_live_seconds,
                        "deactivation_buffer_seconds": template.deactivation_buffer_seconds,
                    } if template else None,
                }
            )

        return {
            "active_backend": (
                {
                    "endpoint": selected.name,
                    "backend": selected.backend_type.value,
                    "gpu": selected.gpu_name.value,
                }
                if selected is not None
                else None
            ),
            "turn_down_candidates": candidates,
        }

    async def _eligible_for_deactivation(self, template: BackendProfileTemplate | None) -> bool:
        if template is None or not template.activated or not template.activated_endpoint_id:
            return False

        now = datetime.utcnow()
        last_used = template.last_used_at or template.activated_at or template.created_at
        activated_at = template.activated_at or template.created_at

        if now - activated_at < timedelta(seconds=template.min_live_seconds):
            return False

        quiet_window = template.idle_ttl_seconds + template.deactivation_buffer_seconds
        if now - last_used < timedelta(seconds=quiet_window):
            return False

        if self._telemetry_store is not None:
            tel = self._telemetry_store.get(template.activated_endpoint_id)
            if tel and (tel.active_requests > 0 or tel.queue_depth > 0):
                return False

        return True

    async def reconcile_idle_templates(self) -> int:
        """Deactivate templates whose live endpoint has been idle past TTL + buffer."""
        async with self._lock:
            templates = list(self._templates.values())

        deactivated = 0
        for template in templates:
            if not await self._eligible_for_deactivation(template):
                continue
            await self._deactivate(template)
            deactivated += 1

        return deactivated

    # ------------------------------------------------------------------
    # Activation / deactivation
    # ------------------------------------------------------------------

    async def activate_template(self, template_id: str) -> BackendProfileTemplate | None:
        template = self._templates.get(template_id)
        if template is None:
            return None
        await self._activate(template, None)
        return template

    async def deactivate_template(self, template_id: str) -> BackendProfileTemplate | None:
        template = self._templates.get(template_id)
        if template is None:
            return None
        await self._deactivate(template)
        return template

    async def _activate(
        self,
        template: BackendProfileTemplate,
        profile: RequestProfile | None,
    ) -> None:
        async with self._lock:
            if template.activated:
                template.last_used_at = datetime.utcnow()
                return
            template.activated = True

        if self._registry is None or self._telemetry_collector is None:
            logger.warning("profile_activation_skipped_no_registry", template_id=template.id)
            async with self._lock:
                template.activated = False
            return

        try:
            if template.exclusive_model_residency:
                await self._deactivate_sibling_templates(template)

            req = EndpointRegisterRequest(
                name=template.name,
                nim_url=template.nim_url,
                backend_type=template.backend_type,
                model_name=template.model_name,
                model_family=template.model_family,
                model_version=template.model_version,
                gpu_name=template.gpu_name,
                gpu_count=template.gpu_count,
                region=template.region,
                cost_class=template.cost_class,
                max_context_tokens=template.max_context_tokens,
                supports_streaming=template.supports_streaming,
                supports_reasoning=template.supports_reasoning,
                tenant_tags=template.tenant_tags,
                capability_flags=template.capability_flags,
                cost_per_gpu_hour=template.cost_per_gpu_hour,
            )
            endpoint = await self._registry.register(req)
            self._telemetry_collector.register_endpoint(endpoint)

            now = datetime.utcnow()
            async with self._lock:
                template.activated = True
                template.activated_endpoint_id = endpoint.id
                template.activated_at = now
                template.last_used_at = now
                template.activation_count += 1

            logger.info(
                "profile_template_activated",
                template_id=template.id,
                endpoint_id=endpoint.id,
                name=template.name,
                backend=template.backend_type.value,
                request_model=(profile.model_requested if profile else template.model_name),
            )
        except Exception as exc:
            async with self._lock:
                template.activated = False
            logger.error(
                "profile_template_activation_failed",
                template_id=template.id,
                error=str(exc),
            )

    async def _deactivate(self, template: BackendProfileTemplate) -> None:
        endpoint_id = template.activated_endpoint_id
        if endpoint_id and self._telemetry_collector is not None:
            self._telemetry_collector.unregister_endpoint(endpoint_id)
        if endpoint_id and self._registry is not None:
            await self._registry.delete(endpoint_id)

        async with self._lock:
            template.activated = False
            template.activated_endpoint_id = None
            template.activated_at = None
            template.last_used_at = None

        logger.info(
            "profile_template_deactivated",
            template_id=template.id,
            name=template.name,
            endpoint_id=endpoint_id,
        )

    async def _deactivate_sibling_templates(self, activated_template: BackendProfileTemplate) -> None:
        async with self._lock:
            siblings = [
                t for t in self._templates.values()
                if t.id != activated_template.id
                and t.activated
                and self._same_model_family_or_name(t, activated_template)
            ]
        for sibling in siblings:
            await self._deactivate(sibling)

    # ------------------------------------------------------------------
    # Template selection
    # ------------------------------------------------------------------

    async def _best_template_for_request(
        self,
        profile: RequestProfile,
        dormant_only: bool,
    ) -> BackendProfileTemplate | None:
        async with self._lock:
            candidates = [
                t for t in self._templates.values()
                if t.auto_activate
                and (not t.activated if dormant_only else True)
                and self._template_matches_request(t, profile)
            ]
        if not candidates:
            return None
        return max(candidates, key=lambda t: self._template_score(t, profile))

    @staticmethod
    def _template_matches_request(template: BackendProfileTemplate, profile: RequestProfile) -> bool:
        if template.workload_affinity and profile.workload_type not in template.workload_affinity:
            return False
        if profile.total_tokens > template.max_context_tokens:
            return False
        return ProfileManager._model_matches_template(template, profile.model_requested)

    @staticmethod
    def _model_matches_template(template: BackendProfileTemplate, requested_model: str) -> bool:
        req = requested_model.lower()
        if req in ("", "*", "any"):
            return True
        candidates = list(template.activation_model_names)
        if template.model_name:
            candidates.append(template.model_name)
        if template.model_family:
            candidates.append(template.model_family)
        for candidate in candidates:
            cand = candidate.lower()
            if req == cand or req.startswith(cand) or cand.startswith(req):
                return True
        return False

    @staticmethod
    def _same_model_family_or_name(a: BackendProfileTemplate, b: BackendProfileTemplate) -> bool:
        a_names = {x.lower() for x in ([a.model_name, a.model_family] + a.activation_model_names) if x}
        b_names = {x.lower() for x in ([b.model_name, b.model_family] + b.activation_model_names) if x}
        return bool(a_names & b_names)

    def _template_score(self, template: BackendProfileTemplate, profile: RequestProfile) -> float:
        score = 0.0

        # Model specificity: exact match > family match.
        if template.model_name.lower() == profile.model_requested.lower():
            score += 3.0
        elif profile.inferred_model_family and profile.inferred_model_family in template.model_name.lower():
            score += 2.0
        else:
            score += 1.0

        # Backend affinity by workload and user intent.
        score += backend_affinity(template.backend_type, profile.workload_type) * 3.0

        vram = _GPU_MEMORY_GB.get(template.gpu_name, 24.0) * max(template.gpu_count, 1)
        score += min(vram / 80.0, 2.0)

        if profile.optimization_target == OptimizationTarget.LATENCY:
            if template.backend_type == BackendType.NIM:
                score += 1.0
            if profile.streaming and template.supports_streaming:
                score += 0.1
        elif profile.optimization_target == OptimizationTarget.THROUGHPUT:
            if template.backend_type in (BackendType.VLLM, BackendType.DYNAMO):
                score += 1.0

        if profile.osl_tokens >= 1024 and template.backend_type in (BackendType.VLLM, BackendType.DYNAMO):
            score += 0.75
        if profile.isl_tokens >= 1024 and template.backend_type == BackendType.SGLANG:
            score += 0.75
        if profile.workload_type == WorkloadType.REASONING and template.supports_reasoning:
            score += 0.75

        if template.cost_class == CostClass.ECONOMY and profile.optimization_target == OptimizationTarget.THROUGHPUT:
            score += 0.2
        if template.cost_class == CostClass.PREMIUM and profile.optimization_target == OptimizationTarget.LATENCY:
            score += 0.2

        return score
