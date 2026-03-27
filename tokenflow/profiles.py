"""
Dynamic backend profile management.

Allows pre-defining backend profile *templates* that are lazily activated
when matching workload types are detected. The current request is never
delayed — activation happens asynchronously in the background, making the
new endpoint available for subsequent requests.

Usage:
  1. Register a profile template via POST /admin/profiles
  2. Set workload_affinity to the WorkloadType(s) it should serve
  3. When a request with a matching workload arrives, the profile is
     automatically activated (endpoint registered) in the background

This lets operators define backend diversity without pre-spinning every
endpoint — e.g. only spin up an SGLang profile when the first
prefill-heavy request arrives.
"""

from __future__ import annotations

import asyncio
import uuid
from datetime import datetime
from typing import Any

import structlog
from pydantic import BaseModel, Field

from tokenflow.models import (
    BackendType,
    CostClass,
    EndpointRegisterRequest,
    GPUClass,
    RequestProfile,
    WorkloadType,
)

logger = structlog.get_logger(__name__)


class BackendProfileTemplate(BaseModel):
    """
    A backend endpoint template that can be activated on-demand.

    When a request with a matching workload type arrives, this template
    is registered as a live endpoint — without blocking the request.
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

    # State
    activated: bool = False
    activated_endpoint_id: str | None = None
    activated_at: datetime | None = None
    created_at: datetime = Field(default_factory=datetime.utcnow)


class ProfileManager:
    """
    Manages backend profile templates and their lazy activation.

    Activation is always non-blocking: it spawns a background asyncio task
    so the in-flight request sees zero startup delay.
    """

    def __init__(self) -> None:
        self._templates: dict[str, BackendProfileTemplate] = {}
        self._lock = asyncio.Lock()
        # Set during app startup via attach()
        self._registry = None
        self._telemetry_collector = None

    def attach(self, registry, telemetry_collector) -> None:
        """Wire up registry and telemetry collector after app startup."""
        self._registry = registry
        self._telemetry_collector = telemetry_collector

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
                workload_affinity=[w.value for w in template.workload_affinity],
            )
            return template

    async def get_template(self, template_id: str) -> BackendProfileTemplate | None:
        return self._templates.get(template_id)

    async def list_templates(self) -> list[BackendProfileTemplate]:
        return list(self._templates.values())

    async def delete_template(self, template_id: str) -> bool:
        async with self._lock:
            if template_id not in self._templates:
                return False
            del self._templates[template_id]
            return True

    # ------------------------------------------------------------------
    # Lazy activation
    # ------------------------------------------------------------------

    async def maybe_activate_for_request(self, profile: RequestProfile) -> None:
        """
        Activate only templates relevant to the incoming request.

        This avoids eagerly spinning every backend for a workload class and helps
        keep GPU memory pressure down when multiple engines can serve the same model.
        """
        if self._registry is None:
            return

        async with self._lock:
            to_activate = [
                t for t in self._templates.values()
                if t.auto_activate
                and not t.activated
                and (not t.workload_affinity or profile.workload_type in t.workload_affinity)
                and self._model_matches_template(t, profile.model_requested)
            ]

        for template in to_activate:
            asyncio.create_task(self._activate(template))

    @staticmethod
    def _model_matches_template(template: BackendProfileTemplate, requested_model: str) -> bool:
        if not template.activation_model_names:
            return True
        req = requested_model.lower()
        return any(req == m.lower() or req.startswith(m.lower()) for m in template.activation_model_names)

    async def activate_template(self, template_id: str) -> BackendProfileTemplate | None:
        """Manually activate a specific template (blocking)."""
        template = self._templates.get(template_id)
        if template is None:
            return None
        await self._activate(template)
        return template

    async def _activate(self, template: BackendProfileTemplate) -> None:
        """Register the template as a live endpoint. Idempotent."""
        async with self._lock:
            if template.activated:
                return
            # Mark activated immediately to prevent duplicate activation
            template.activated = True

        if self._registry is None or self._telemetry_collector is None:
            logger.warning("profile_activation_skipped_no_registry", template_id=template.id)
            return

        try:
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
            profile = await self._registry.register(req)
            self._telemetry_collector.register_endpoint(profile)

            template.activated_endpoint_id = profile.id
            template.activated_at = datetime.utcnow()

            logger.info(
                "profile_template_activated",
                template_id=template.id,
                endpoint_id=profile.id,
                name=template.name,
            )
        except Exception as exc:
            # Roll back activated flag so it can be retried
            template.activated = False
            logger.error(
                "profile_template_activation_failed",
                template_id=template.id,
                error=str(exc),
            )
