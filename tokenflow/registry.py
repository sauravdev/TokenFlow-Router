"""Endpoint registry — in-memory store of registered NIM endpoints."""

from __future__ import annotations

import asyncio
from datetime import datetime
from typing import Optional

import structlog

from tokenflow.models import (
    EndpointHealth,
    EndpointProfile,
    EndpointRegisterRequest,
)

logger = structlog.get_logger(__name__)


class EndpointRegistry:
    """
    Thread-safe, in-memory registry of NIM endpoint profiles.

    Endpoints are keyed by their UUID. The registry tracks:
    - static metadata (model, GPU, region, cost class)
    - dynamic health state (updated by telemetry collector)
    """

    def __init__(self) -> None:
        self._endpoints: dict[str, EndpointProfile] = {}
        self._lock = asyncio.Lock()

    # ------------------------------------------------------------------
    # CRUD
    # ------------------------------------------------------------------

    async def register(self, req: EndpointRegisterRequest) -> EndpointProfile:
        """Register a new endpoint. Returns the created profile."""
        async with self._lock:
            # Deduplicate by nim_url + model_name
            for ep in self._endpoints.values():
                if ep.nim_url == req.nim_url.rstrip("/") and ep.model_name == req.model_name:
                    logger.info(
                        "endpoint_already_registered",
                        endpoint_id=ep.id,
                        nim_url=ep.nim_url,
                    )
                    return ep

            profile = EndpointProfile(**req.model_dump())
            self._endpoints[profile.id] = profile
            logger.info(
                "endpoint_registered",
                endpoint_id=profile.id,
                name=profile.name,
                model=profile.model_name,
                gpu=profile.gpu_name,
            )
            return profile

    async def get(self, endpoint_id: str) -> Optional[EndpointProfile]:
        return self._endpoints.get(endpoint_id)

    async def list_all(
        self,
        enabled_only: bool = False,
        healthy_only: bool = False,
    ) -> list[EndpointProfile]:
        endpoints = list(self._endpoints.values())
        if enabled_only:
            endpoints = [e for e in endpoints if e.enabled]
        if healthy_only:
            endpoints = [
                e
                for e in endpoints
                if e.health in (EndpointHealth.HEALTHY, EndpointHealth.UNKNOWN)
            ]
        return endpoints

    async def update_health(
        self, endpoint_id: str, health: EndpointHealth
    ) -> None:
        async with self._lock:
            if endpoint_id in self._endpoints:
                self._endpoints[endpoint_id].health = health
                self._endpoints[endpoint_id].updated_at = datetime.utcnow()
                logger.debug(
                    "endpoint_health_updated",
                    endpoint_id=endpoint_id,
                    health=health,
                )

    async def enable(self, endpoint_id: str) -> bool:
        async with self._lock:
            if endpoint_id not in self._endpoints:
                return False
            self._endpoints[endpoint_id].enabled = True
            self._endpoints[endpoint_id].updated_at = datetime.utcnow()
            return True

    async def disable(self, endpoint_id: str) -> bool:
        async with self._lock:
            if endpoint_id not in self._endpoints:
                return False
            self._endpoints[endpoint_id].enabled = False
            self._endpoints[endpoint_id].updated_at = datetime.utcnow()
            return True

    async def delete(self, endpoint_id: str) -> bool:
        async with self._lock:
            if endpoint_id not in self._endpoints:
                return False
            del self._endpoints[endpoint_id]
            logger.info("endpoint_deleted", endpoint_id=endpoint_id)
            return True

    # ------------------------------------------------------------------
    # Queries
    # ------------------------------------------------------------------

    async def find_by_model(self, model_name: str) -> list[EndpointProfile]:
        """Return all enabled endpoints that serve the requested model."""
        return [
            ep
            for ep in self._endpoints.values()
            if ep.enabled and self._model_matches(ep.model_name, model_name)
        ]

    @staticmethod
    def _model_matches(served: str, requested: str) -> bool:
        """Flexible model name matching (exact, family prefix, or 'any')."""
        if requested in ("any", "*", ""):
            return True
        served_lower = served.lower()
        req_lower = requested.lower()
        return served_lower == req_lower or served_lower.startswith(req_lower)

    @property
    def count(self) -> int:
        return len(self._endpoints)

    @property
    def healthy_count(self) -> int:
        return sum(
            1
            for e in self._endpoints.values()
            if e.health == EndpointHealth.HEALTHY
        )
