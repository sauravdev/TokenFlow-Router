"""
Tests for Ollama-specific optimizations:
  1. OllamaClient adapter — warm model detection via /api/ps
  2. Cold-start penalty scoring — warm endpoints score higher
  3. Telemetry feedback — real inference latency updates scoring
  4. End-to-end: router prefers warm Ollama endpoint over cold one

Requires: Ollama running at localhost:11434 with qwen2.5:1.5b and qwen2.5:3b
"""

from __future__ import annotations

import asyncio
import time

import httpx
import pytest

from tokenflow.adapters.ollama.client import OllamaClient
from tokenflow.classifier import RequestClassifier
from tokenflow.models import (
    BackendType,
    CostClass,
    EndpointHealth,
    EndpointProfile,
    EndpointRegisterRequest,
    GPUClass,
    LatencyClass,
    RoutingPolicy,
    TelemetryUpdate,
    WorkloadType,
)
from tokenflow.registry import EndpointRegistry
from tokenflow.router import DecisionEngine, ScoringEngine, _apply_preset
from tokenflow.telemetry import TelemetryStore

clf = RequestClassifier()

OLLAMA_BASE = "http://localhost:11434"
SMALL_MODEL = "qwen2.5:1.5b"
LARGE_MODEL = "qwen2.5:3b"


async def _ollama_has_both_models() -> bool:
    try:
        async with httpx.AsyncClient(timeout=3.0) as client:
            r = await client.get(f"{OLLAMA_BASE}/api/tags")
            if r.status_code != 200:
                return False
            models = [m["name"] for m in r.json().get("models", [])]
            return SMALL_MODEL in models and LARGE_MODEL in models
    except Exception:
        return False


async def _infer(model: str, prompt: str, max_tokens: int = 8) -> dict:
    body = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "stream": False,
        "options": {"num_predict": max_tokens},
    }
    t0 = time.perf_counter()
    async with httpx.AsyncClient(timeout=120.0) as client:
        r = await client.post(f"{OLLAMA_BASE}/v1/chat/completions", json=body)
        wall_ms = (time.perf_counter() - t0) * 1000
        r.raise_for_status()
    return {"wall_ms": wall_ms}


def make_ep(name: str, model: str, **kwargs) -> EndpointProfile:
    defaults = dict(
        nim_url=OLLAMA_BASE,
        gpu_name=GPUClass.RTX4090,
        backend_type=BackendType.OLLAMA,
        cost_class=CostClass.ECONOMY,
        cost_per_gpu_hour=0.5,
        max_context_tokens=8192,
        health=EndpointHealth.HEALTHY,
    )
    defaults.update(kwargs)
    return EndpointProfile(name=name, model_name=model, **defaults)


async def make_registry(*endpoints: EndpointProfile) -> EndpointRegistry:
    registry = EndpointRegistry()
    for ep in endpoints:
        req = EndpointRegisterRequest(
            **{k: getattr(ep, k) for k in EndpointRegisterRequest.model_fields}
        )
        await registry.register(req)
    return registry


pytestmark = pytest.mark.skipif(
    not asyncio.get_event_loop().run_until_complete(_ollama_has_both_models()),
    reason=f"Ollama not running or missing {SMALL_MODEL} / {LARGE_MODEL}",
)


# ---------------------------------------------------------------------------
# 1. OllamaClient adapter tests
# ---------------------------------------------------------------------------


class TestOllamaClientAdapter:

    @pytest.mark.asyncio
    async def test_is_ready(self):
        client = OllamaClient(timeout=5.0)
        try:
            assert await client.is_ready(OLLAMA_BASE)
        finally:
            await client.close()

    @pytest.mark.asyncio
    async def test_get_running_models_returns_list(self):
        client = OllamaClient(timeout=5.0)
        try:
            running = await client.get_running_models(OLLAMA_BASE)
            assert isinstance(running, list)
            print(f"\n  Running models: {[m.get('name') for m in running]}")
        finally:
            await client.close()

    @pytest.mark.asyncio
    async def test_warm_model_detection_after_inference(self):
        """After running inference on a model, /api/ps should show it as loaded."""
        await _infer(SMALL_MODEL, "hi", max_tokens=4)

        client = OllamaClient(timeout=5.0)
        try:
            running = await client.get_running_models(OLLAMA_BASE)
            loaded_names = [m.get("name", "") for m in running]
            assert any(SMALL_MODEL in n for n in loaded_names), (
                f"Expected {SMALL_MODEL} in running models, got {loaded_names}"
            )
            print(f"\n  After inference on {SMALL_MODEL}:")
            print(f"    Loaded models: {loaded_names}")
            print(f"    is_warm({SMALL_MODEL}): {client._is_model_warm(running, SMALL_MODEL)}")
            print(f"    is_warm({LARGE_MODEL}): {client._is_model_warm(running, LARGE_MODEL)}")
        finally:
            await client.close()

    @pytest.mark.asyncio
    async def test_probe_sets_warm_flag_on_endpoint(self):
        """probe() should set capability_flags['warm'] on the endpoint."""
        await _infer(SMALL_MODEL, "hi", max_tokens=4)

        client = OllamaClient(timeout=5.0)
        try:
            ep = make_ep("test-ollama", model=SMALL_MODEL)
            update = await client.probe(ep)

            assert ep.capability_flags.get("warm") is True
            assert update.error_rate == 0.0
            assert update.saturation_score < 0.5

            print(f"\n  Probe results for warm {SMALL_MODEL}:")
            print(f"    warm: {ep.capability_flags['warm']}")
            print(f"    loaded_models: {ep.capability_flags.get('ollama_loaded_models')}")
            print(f"    saturation: {update.saturation_score}")
            print(f"    p95_ttft_ms: {update.p95_ttft_ms}")
        finally:
            await client.close()

    @pytest.mark.asyncio
    async def test_probe_detects_model_state(self):
        """Probe accurately reflects whether a model is loaded or not."""
        await _infer(SMALL_MODEL, "hi", max_tokens=4)

        client = OllamaClient(timeout=5.0)
        try:
            # The small model should definitely be warm
            ep_warm = make_ep("warm-ollama", model=SMALL_MODEL)
            await client.probe(ep_warm)
            assert ep_warm.capability_flags.get("warm") is True

            # Check a completely nonexistent model
            ep_fake = make_ep("fake-ollama", model="nonexistent-model:0.1b")
            await client.probe(ep_fake)
            assert ep_fake.capability_flags.get("warm") is False

            print(f"\n  Probe model state detection:")
            print(f"    {SMALL_MODEL}: warm={ep_warm.capability_flags['warm']}")
            print(f"    nonexistent: warm={ep_fake.capability_flags['warm']}")
            print(f"    loaded: {ep_warm.capability_flags.get('ollama_loaded_models')}")
        finally:
            await client.close()


# ---------------------------------------------------------------------------
# 2. Scoring: warm vs cold endpoint
# ---------------------------------------------------------------------------


class TestWarmVsColdScoring:

    def test_warm_ollama_scores_higher_than_cold(self):
        """The scoring engine should prefer a warm Ollama endpoint."""
        ep_warm = make_ep("ollama-warm", model=SMALL_MODEL)
        ep_warm.capability_flags["warm"] = True
        ep_warm.capability_flags["ollama_loaded_models"] = []

        ep_cold = make_ep("ollama-cold", model=SMALL_MODEL)
        ep_cold.capability_flags["warm"] = False
        ep_cold.capability_flags["ollama_loaded_models"] = [LARGE_MODEL]

        store = TelemetryStore()
        engine = ScoringEngine(RoutingPolicy(), store)

        req_body = {
            "model": SMALL_MODEL,
            "messages": [{"role": "user", "content": "Hello"}],
            "max_tokens": 32,
        }
        profile = clf.classify(req_body)

        s_warm = engine.score(ep_warm, profile)
        s_cold = engine.score(ep_cold, profile)

        print(f"\n  Warm vs Cold Ollama Scoring:")
        print(f"  {'Metric':<25} {'Warm':>12} {'Cold':>12} {'Delta':>10}")
        print("  " + "-" * 63)
        print(f"  {'Benchmark score':<25} {s_warm.benchmark_score:>12.4f} {s_cold.benchmark_score:>12.4f} {s_warm.benchmark_score - s_cold.benchmark_score:>+10.4f}")
        print(f"  {'Utility score':<25} {s_warm.utility_score:>12.4f} {s_cold.utility_score:>12.4f} {s_warm.utility_score - s_cold.utility_score:>+10.4f}")

        assert s_warm.benchmark_score > s_cold.benchmark_score, (
            f"Warm ({s_warm.benchmark_score:.4f}) should score higher than cold ({s_cold.benchmark_score:.4f})"
        )
        assert s_warm.utility_score > s_cold.utility_score

    def test_cold_penalty_scales_with_latency_class(self):
        """Interactive requests should get a heavier cold-start penalty than batch."""
        ep = make_ep("ollama", model=SMALL_MODEL)
        ep.capability_flags["warm"] = False
        ep.capability_flags["ollama_loaded_models"] = [LARGE_MODEL]

        store = TelemetryStore()
        engine = ScoringEngine(RoutingPolicy(), store)

        from tokenflow.models import PriorityTier
        interactive_body = {
            "model": SMALL_MODEL,
            "messages": [{"role": "user", "content": "Hi"}],
            "max_tokens": 16,
        }
        batch_body = {
            "model": SMALL_MODEL,
            "messages": [{"role": "user", "content": "Hi"}],
            "max_tokens": 16,
        }

        profile_interactive = clf.classify(interactive_body, priority_tier=PriorityTier.PREMIUM)
        profile_batch = clf.classify(batch_body, priority_tier=PriorityTier.BATCH)

        s_interactive = engine.score(ep, profile_interactive)
        s_batch = engine.score(ep, profile_batch)

        print(f"\n  Cold-start penalty by latency class:")
        print(f"    Interactive benchmark_score: {s_interactive.benchmark_score:.4f}")
        print(f"    Batch benchmark_score:       {s_batch.benchmark_score:.4f}")

        assert s_batch.benchmark_score > s_interactive.benchmark_score, (
            "Batch requests should have a milder cold-start penalty"
        )


# ---------------------------------------------------------------------------
# 3. End-to-end: router prefers warm endpoint with real Ollama
# ---------------------------------------------------------------------------


class TestRouterPrefersWarmEndpoint:

    @pytest.mark.asyncio
    async def test_router_picks_warm_model_endpoint(self):
        """
        Register two Ollama endpoints (same Ollama instance, different models).
        Warm up one model. The router should prefer the warm one.
        """
        await _infer(SMALL_MODEL, "warmup", max_tokens=4)

        ep_small = make_ep("ollama-small", model=SMALL_MODEL)
        ep_large = make_ep("ollama-large", model=LARGE_MODEL,
                           nim_url=OLLAMA_BASE)

        client = OllamaClient(timeout=5.0)
        try:
            await client.probe(ep_small)
            await client.probe(ep_large)
        finally:
            await client.close()

        print(f"\n  After probing (with {SMALL_MODEL} warm):")
        print(f"    {ep_small.name}: warm={ep_small.capability_flags.get('warm')}")
        print(f"    {ep_large.name}: warm={ep_large.capability_flags.get('warm')}")

        # Both endpoints serve "any" model — simulate user requesting any
        # by making both serve a common model name for matching purposes
        ep_small.model_name = "qwen2.5"
        ep_large.model_name = "qwen2.5"

        registry = await make_registry(ep_small, ep_large)
        store = TelemetryStore()
        engine = DecisionEngine(registry=registry, store=store)
        engine.set_policy(RoutingPolicy())

        req_body = {
            "model": "qwen2.5",
            "messages": [{"role": "user", "content": "Hello"}],
            "max_tokens": 32,
        }
        profile = clf.classify(req_body)
        decision = await engine.decide(profile)

        scoring = ScoringEngine(engine.policy, store)
        all_eps = await registry.list_all()
        for ep in all_eps:
            s = scoring.score(ep, profile)
            print(f"    {ep.name}: utility={s.utility_score:.4f}, benchmark={s.benchmark_score:.4f}, "
                  f"warm={ep.capability_flags.get('warm')}")

        print(f"\n  Router selected: {decision.selected_endpoint_name}")
        assert decision.selected_endpoint_name == "ollama-small", (
            f"Expected warm endpoint 'ollama-small', got '{decision.selected_endpoint_name}'"
        )

    @pytest.mark.asyncio
    async def test_measured_cold_vs_warm_with_routing(self):
        """
        Measure actual latency difference between routing to warm vs cold model.
        This is the real payoff of warm-aware routing.
        """
        # Warm up the small model
        await _infer(SMALL_MODEL, "warmup", max_tokens=4)

        # Measure warm inference
        warm_result = await _infer(SMALL_MODEL, "What is 2+2?", max_tokens=8)

        # Force a cold switch to large model
        cold_result = await _infer(LARGE_MODEL, "What is 2+2?", max_tokens=8)

        # Warm inference on large model (now loaded)
        warm_large = await _infer(LARGE_MODEL, "What is 2+2?", max_tokens=8)

        cold_penalty = cold_result["wall_ms"] - warm_large["wall_ms"]

        print("\n" + "=" * 70)
        print("  REAL COLD-START PENALTY: What the router avoids")
        print("=" * 70)
        print(f"  Warm 1.5b request:     {warm_result['wall_ms']:>8.1f} ms")
        print(f"  Cold 3b request:       {cold_result['wall_ms']:>8.1f} ms  (includes model swap)")
        print(f"  Warm 3b request:       {warm_large['wall_ms']:>8.1f} ms")
        print(f"  Cold-start penalty:    {cold_penalty:>8.1f} ms")
        print(f"\n  By tracking warm models via /api/ps, the router avoids")
        print(f"  {cold_penalty:.0f}ms of unnecessary model-swap latency per misrouted request.")

        assert cold_result["wall_ms"] > warm_large["wall_ms"]
