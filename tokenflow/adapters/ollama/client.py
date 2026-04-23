"""
Ollama inference backend adapter.

Ollama (https://ollama.com) is a lightweight local model server. It loads
one model at a time into GPU/CPU memory. Switching models incurs a cold-start
penalty (measured at 130–315ms on Apple Silicon, much higher on CPU).

The key routing-relevant signal Ollama exposes is which model is currently
loaded in memory:

  GET /api/tags  → list of downloaded models (health proxy)
  GET /api/ps    → list of models currently LOADED in memory (warm detection)

A model that is already loaded responds in ~100–300ms. A model that needs
to be swapped in pays an additional 130ms–13s depending on model size and
hardware. This adapter surfaces that state so the scoring engine can
avoid unnecessary cold starts.
"""

from __future__ import annotations

import time
from typing import Any, Optional

import httpx
import structlog

from tokenflow.models import EndpointProfile, TelemetryUpdate

logger = structlog.get_logger(__name__)


class OllamaClient:
    """
    Async client for Ollama endpoint health, warm-model detection, and telemetry.

    Used by TelemetryCollector when backend_type == BackendType.OLLAMA.
    """

    def __init__(self, timeout: float = 5.0) -> None:
        self._client = httpx.AsyncClient(timeout=timeout)

    async def close(self) -> None:
        await self._client.aclose()

    async def is_ready(self, base_url: str) -> bool:
        """GET /api/tags — returns True if Ollama is running and responsive."""
        try:
            resp = await self._client.get(f"{base_url}/api/tags")
            return resp.status_code == 200
        except Exception:
            return False

    async def get_running_models(self, base_url: str) -> list[dict[str, Any]]:
        """
        GET /api/ps — returns models currently loaded in memory.

        Response shape:
          {"models": [{"name": "qwen2.5:1.5b", "model": "qwen2.5:1.5b",
                        "size": 986061892, "digest": "...",
                        "details": {"parameter_size": "1.5B", ...},
                        "expires_at": "...", "size_vram": 561745920}]}

        An empty list means no model is loaded (everything will cold-start).
        """
        try:
            resp = await self._client.get(f"{base_url}/api/ps")
            if resp.status_code == 200:
                return resp.json().get("models", [])
            return []
        except Exception:
            return []

    async def get_available_models(self, base_url: str) -> list[dict[str, Any]]:
        """GET /api/tags — list all downloaded models."""
        try:
            resp = await self._client.get(f"{base_url}/api/tags")
            if resp.status_code == 200:
                return resp.json().get("models", [])
            return []
        except Exception:
            return []

    def _is_model_warm(
        self, running_models: list[dict[str, Any]], target_model: str
    ) -> bool:
        """Check if target_model is currently loaded in Ollama's memory."""
        target = target_model.lower().strip()
        for m in running_models:
            name = m.get("name", "").lower().strip()
            model = m.get("model", "").lower().strip()
            if target == name or target == model:
                return True
            # Handle tag-less matching: "qwen2.5:1.5b" matches "qwen2.5:1.5b"
            # and "qwen2.5" matches "qwen2.5:latest"
            if target.split(":")[0] == name.split(":")[0]:
                if ":" not in target or target == name:
                    return True
        return False

    def _extract_vram_usage(
        self, running_models: list[dict[str, Any]], target_model: str
    ) -> Optional[float]:
        """Extract VRAM usage in bytes for the target model, if loaded."""
        target = target_model.lower().strip()
        for m in running_models:
            name = m.get("name", "").lower().strip()
            if target == name or target.split(":")[0] == name.split(":")[0]:
                return m.get("size_vram")
        return None

    async def probe(self, ep: EndpointProfile) -> TelemetryUpdate:
        """
        Full probe: health check + warm model detection.

        Sets ep.capability_flags["warm"] based on whether the endpoint's
        model is currently loaded in Ollama's memory. This flag is consumed
        by the scoring engine's cold-start penalty logic.
        """
        t_start = time.perf_counter()
        ready = await self.is_ready(ep.nim_url)
        probe_ms = (time.perf_counter() - t_start) * 1000

        if not ready:
            ep.capability_flags["warm"] = False
            ep.capability_flags["ollama_loaded_models"] = []
            return TelemetryUpdate(
                endpoint_id=ep.id,
                error_rate=1.0,
                saturation_score=1.0,
            )

        running = await self.get_running_models(ep.nim_url)
        is_warm = self._is_model_warm(running, ep.model_name)
        loaded_names = [m.get("name", "") for m in running]

        ep.capability_flags["warm"] = is_warm
        ep.capability_flags["ollama_loaded_models"] = loaded_names

        vram = self._extract_vram_usage(running, ep.model_name)
        if vram is not None:
            ep.capability_flags["ollama_vram_bytes"] = vram

        # Saturation: Ollama runs one model at a time, so if a different model
        # is loaded, this endpoint is effectively "occupied" for other models.
        if running and not is_warm:
            saturation = 0.8  # loaded with wrong model — high swap probability
        elif running and is_warm:
            saturation = 0.1  # warm and ready
        else:
            saturation = 0.0  # nothing loaded — fast cold start for small models

        logger.debug(
            "ollama_probe_ok",
            endpoint=ep.name,
            model=ep.model_name,
            warm=is_warm,
            loaded=loaded_names,
            probe_ms=round(probe_ms, 1),
        )

        return TelemetryUpdate(
            endpoint_id=ep.id,
            p50_ttft_ms=probe_ms * 1.2,
            p95_ttft_ms=probe_ms * 2.0 if is_warm else probe_ms * 8.0,
            p50_e2e_ms=probe_ms * 2.0,
            p95_e2e_ms=probe_ms * 4.0 if is_warm else probe_ms * 15.0,
            error_rate=0.0,
            saturation_score=saturation,
        )
