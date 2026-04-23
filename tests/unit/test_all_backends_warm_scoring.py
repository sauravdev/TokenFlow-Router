"""
Tests for warm/cold detection and scoring across ALL backends:
  NIM, vLLM, SGLang, Dynamo, Ollama.

Verifies:
  1. Warm endpoints score higher than cold ones (all backends)
  2. Cold-start penalty scales with latency class (all backends)
  3. Per-backend capability_flags affect gpu_affinity scoring
  4. vLLM cache saturation penalizes scoring
  5. Dynamo queue imbalance penalizes scoring
  6. SGLang cache_hit_rate boosts prefill-heavy scoring
"""

from __future__ import annotations

from typing import Any

import pytest

from tokenflow.benchmarks import get_backend_benchmark
from tokenflow.classifier import RequestClassifier
from tokenflow.models import (
    BackendType,
    CostClass,
    EndpointHealth,
    EndpointProfile,
    GPUClass,
    LatencyClass,
    OptimizationTarget,
    PriorityTier,
    RoutingPolicy,
    WorkloadType,
)
from tokenflow.router import ScoringEngine
from tokenflow.telemetry import TelemetryStore

clf = RequestClassifier()


def make_ep(
    name: str = "ep",
    model: str = "meta/llama-3.1-8b-instruct",
    backend: BackendType = BackendType.NIM,
    gpu: GPUClass = GPUClass.L40S,
    capability_flags: dict[str, Any] | None = None,
    **kwargs,
) -> EndpointProfile:
    defaults = dict(
        nim_url=f"http://{name}:8000",
        cost_class=CostClass.STANDARD,
        cost_per_gpu_hour=3.0,
        max_context_tokens=16384,
        health=EndpointHealth.HEALTHY,
    )
    defaults.update(kwargs)
    ep = EndpointProfile(
        name=name,
        model_name=model,
        gpu_name=gpu,
        backend_type=backend,
        capability_flags=capability_flags or {},
        **defaults,
    )
    return ep


def score_ep(ep, content="Hello", max_tokens=64, priority=PriorityTier.STANDARD,
             opt=OptimizationTarget.AUTO, policy=None):
    store = TelemetryStore()
    engine = ScoringEngine(policy or RoutingPolicy(), store)
    body = {
        "model": ep.model_name,
        "messages": [{"role": "user", "content": content}],
        "max_tokens": max_tokens,
    }
    profile = clf.classify(body, priority_tier=priority, optimization_target=opt)
    return engine.score(ep, profile), profile


# ---------------------------------------------------------------------------
# 1. Warm vs Cold — all backends
# ---------------------------------------------------------------------------


class TestWarmVsColdAllBackends:

    @pytest.mark.parametrize("backend", list(BackendType))
    def test_warm_scores_higher_than_cold(self, backend: BackendType):
        ep_warm = make_ep("warm", backend=backend, capability_flags={"warm": True})
        ep_cold = make_ep("cold", backend=backend, capability_flags={"warm": False})

        s_warm, _ = score_ep(ep_warm)
        s_cold, _ = score_ep(ep_cold)

        assert not s_warm.hard_rejected, f"Warm {backend.value} was rejected: {s_warm.rejection_reason}"
        assert not s_cold.hard_rejected, f"Cold {backend.value} was rejected: {s_cold.rejection_reason}"

        assert s_warm.benchmark_score > s_cold.benchmark_score, (
            f"{backend.value}: warm benchmark ({s_warm.benchmark_score:.4f}) "
            f"should > cold ({s_cold.benchmark_score:.4f})"
        )
        print(f"  {backend.value:<10} warm={s_warm.benchmark_score:.4f} cold={s_cold.benchmark_score:.4f} "
              f"delta={s_warm.benchmark_score - s_cold.benchmark_score:+.4f}")

    def test_warm_cold_summary_table(self):
        print("\n" + "=" * 80)
        print("  WARM vs COLD BENCHMARK SCORES — ALL BACKENDS")
        print("=" * 80)
        print(f"  {'Backend':<12} {'Warm':>10} {'Cold':>10} {'Delta':>10} {'Cold Penalty':>14}")
        print("  " + "-" * 60)

        for bt in BackendType:
            bench = get_backend_benchmark(bt)
            ep_w = make_ep("w", backend=bt, capability_flags={"warm": True})
            ep_c = make_ep("c", backend=bt, capability_flags={"warm": False})
            s_w, _ = score_ep(ep_w)
            s_c, _ = score_ep(ep_c)
            if not s_w.hard_rejected and not s_c.hard_rejected:
                delta = s_w.benchmark_score - s_c.benchmark_score
                print(f"  {bt.value:<12} {s_w.benchmark_score:>10.4f} {s_c.benchmark_score:>10.4f} "
                      f"{delta:>+10.4f} {bench.cold_start_penalty_ms:>12.0f}ms")


# ---------------------------------------------------------------------------
# 2. Cold-start penalty scales with latency class
# ---------------------------------------------------------------------------


class TestColdPenaltyScaling:

    @pytest.mark.parametrize("backend", list(BackendType))
    def test_interactive_penalty_heavier_than_batch(self, backend: BackendType):
        ep = make_ep("ep", backend=backend, capability_flags={"warm": False})

        s_interactive, _ = score_ep(ep, priority=PriorityTier.PREMIUM)
        s_batch, _ = score_ep(ep, priority=PriorityTier.BATCH)

        if s_interactive.hard_rejected or s_batch.hard_rejected:
            pytest.skip(f"{backend.value} rejected for this config")

        assert s_batch.benchmark_score >= s_interactive.benchmark_score, (
            f"{backend.value}: batch ({s_batch.benchmark_score:.4f}) should >= "
            f"interactive ({s_interactive.benchmark_score:.4f})"
        )

    def test_penalty_scaling_table(self):
        print("\n" + "=" * 80)
        print("  COLD-START PENALTY BY LATENCY CLASS — ALL BACKENDS")
        print("=" * 80)
        print(f"  {'Backend':<12} {'Interactive':>14} {'Standard':>14} {'Batch':>14}")
        print("  " + "-" * 58)

        for bt in BackendType:
            ep = make_ep("ep", backend=bt, capability_flags={"warm": False})
            s_i, _ = score_ep(ep, priority=PriorityTier.PREMIUM)
            s_s, _ = score_ep(ep, priority=PriorityTier.STANDARD)
            s_b, _ = score_ep(ep, priority=PriorityTier.BATCH)
            if not s_i.hard_rejected:
                print(f"  {bt.value:<12} {s_i.benchmark_score:>14.4f} "
                      f"{s_s.benchmark_score:>14.4f} {s_b.benchmark_score:>14.4f}")


# ---------------------------------------------------------------------------
# 3. vLLM KV cache saturation penalty
# ---------------------------------------------------------------------------


class TestVLLMCacheSaturation:

    def test_high_cache_penalizes_gpu_affinity(self):
        ep_low = make_ep("low-cache", backend=BackendType.VLLM,
                         capability_flags={"warm": True, "vllm_gpu_cache_usage": 0.30})
        ep_high = make_ep("high-cache", backend=BackendType.VLLM,
                          capability_flags={"warm": True, "vllm_gpu_cache_usage": 0.95})

        s_low, _ = score_ep(ep_low)
        s_high, _ = score_ep(ep_high)

        print(f"\n  vLLM GPU cache saturation impact:")
        print(f"    30% cache: gpu_affinity={s_low.gpu_affinity_score:.4f}")
        print(f"    95% cache: gpu_affinity={s_high.gpu_affinity_score:.4f}")

        assert s_low.gpu_affinity_score > s_high.gpu_affinity_score

    def test_vllm_near_oom_scores_lower_than_healthy(self):
        ep_healthy = make_ep("healthy", backend=BackendType.VLLM,
                             capability_flags={"warm": True, "vllm_gpu_cache_usage": 0.30})
        ep_oom = make_ep("oom", backend=BackendType.VLLM,
                         capability_flags={"warm": False, "vllm_gpu_cache_usage": 0.98})

        s_healthy, _ = score_ep(ep_healthy)
        s_oom, _ = score_ep(ep_oom)

        print(f"\n  vLLM healthy vs near-OOM:")
        print(f"    Healthy: benchmark={s_healthy.benchmark_score:.4f} gpu_aff={s_healthy.gpu_affinity_score:.4f}")
        print(f"    OOM:     benchmark={s_oom.benchmark_score:.4f} gpu_aff={s_oom.gpu_affinity_score:.4f}")

        assert s_healthy.benchmark_score > s_oom.benchmark_score
        assert s_healthy.gpu_affinity_score > s_oom.gpu_affinity_score
        assert s_healthy.utility_score > s_oom.utility_score


# ---------------------------------------------------------------------------
# 4. SGLang cache_hit_rate bonus
# ---------------------------------------------------------------------------


class TestSGLangCacheBonus:

    def test_high_cache_hit_boosts_prefill_heavy(self):
        ep_no_cache = make_ep("no-cache", backend=BackendType.SGLANG,
                              capability_flags={"warm": True, "sglang_cache_hit_rate": 0.0})
        ep_cached = make_ep("cached", backend=BackendType.SGLANG,
                            capability_flags={"warm": True, "sglang_cache_hit_rate": 0.85})

        s_no, _ = score_ep(ep_no_cache, content="x " * 500, max_tokens=32)
        s_yes, _ = score_ep(ep_cached, content="x " * 500, max_tokens=32)

        print(f"\n  SGLang cache_hit_rate impact on prefill-heavy:")
        print(f"    0% hit:  gpu_affinity={s_no.gpu_affinity_score:.4f}")
        print(f"    85% hit: gpu_affinity={s_yes.gpu_affinity_score:.4f}")

        assert s_yes.gpu_affinity_score > s_no.gpu_affinity_score

    def test_cache_hit_no_effect_on_decode_heavy(self):
        ep_no_cache = make_ep("no-cache", backend=BackendType.SGLANG,
                              capability_flags={"warm": True, "sglang_cache_hit_rate": 0.0})
        ep_cached = make_ep("cached", backend=BackendType.SGLANG,
                            capability_flags={"warm": True, "sglang_cache_hit_rate": 0.85})

        s_no, _ = score_ep(ep_no_cache, content="hi", max_tokens=2000)
        s_yes, _ = score_ep(ep_cached, content="hi", max_tokens=2000)

        # Cache hit rate should NOT boost decode-heavy (it only helps prefill)
        assert abs(s_yes.gpu_affinity_score - s_no.gpu_affinity_score) < 0.01


# ---------------------------------------------------------------------------
# 5. Dynamo queue imbalance penalty
# ---------------------------------------------------------------------------


class TestDynamoQueueImbalance:

    def test_balanced_queues_score_higher(self):
        ep_balanced = make_ep("balanced", backend=BackendType.DYNAMO,
                              capability_flags={"warm": True,
                                                "dynamo_prefill_queue": 10,
                                                "dynamo_decode_queue": 12})
        ep_imbalanced = make_ep("imbalanced", backend=BackendType.DYNAMO,
                                capability_flags={"warm": True,
                                                  "dynamo_prefill_queue": 5,
                                                  "dynamo_decode_queue": 40})

        s_bal, _ = score_ep(ep_balanced)
        s_imbal, _ = score_ep(ep_imbalanced)

        print(f"\n  Dynamo queue imbalance impact:")
        print(f"    Balanced (10/12):   gpu_affinity={s_bal.gpu_affinity_score:.4f}")
        print(f"    Imbalanced (5/40):  gpu_affinity={s_imbal.gpu_affinity_score:.4f}")

        assert s_bal.gpu_affinity_score > s_imbal.gpu_affinity_score

    def test_dynamo_kv_hit_boosts_prefill(self):
        ep_no_kv = make_ep("no-kv", backend=BackendType.DYNAMO,
                           capability_flags={"warm": True, "dynamo_kv_hit_rate": 0.0})
        ep_kv = make_ep("kv-warm", backend=BackendType.DYNAMO,
                        capability_flags={"warm": True, "dynamo_kv_hit_rate": 0.80})

        s_no, _ = score_ep(ep_no_kv, content="x " * 500, max_tokens=32)
        s_yes, _ = score_ep(ep_kv, content="x " * 500, max_tokens=32)

        assert s_yes.gpu_affinity_score > s_no.gpu_affinity_score


# ---------------------------------------------------------------------------
# 6. NIM warm/cold
# ---------------------------------------------------------------------------


class TestNIMWarmCold:

    def test_nim_warm_vs_cold(self):
        ep_warm = make_ep("nim-warm", backend=BackendType.NIM,
                          capability_flags={"warm": True, "nim_has_metrics": True})
        ep_cold = make_ep("nim-cold", backend=BackendType.NIM,
                          capability_flags={"warm": False, "nim_has_metrics": False})

        s_w, _ = score_ep(ep_warm)
        s_c, _ = score_ep(ep_cold)

        print(f"\n  NIM warm vs cold:")
        print(f"    Warm: benchmark={s_w.benchmark_score:.4f} utility={s_w.utility_score:.4f}")
        print(f"    Cold: benchmark={s_c.benchmark_score:.4f} utility={s_c.utility_score:.4f}")

        assert s_w.benchmark_score > s_c.benchmark_score
        assert s_w.utility_score > s_c.utility_score


# ---------------------------------------------------------------------------
# 7. Ollama model-swap amplified penalty
# ---------------------------------------------------------------------------


class TestOllamaModelSwap:

    def test_ollama_model_swap_penalty_amplified(self):
        ep_warm = make_ep("warm", backend=BackendType.OLLAMA, gpu=GPUClass.RTX4090,
                          capability_flags={"warm": True})
        ep_cold = make_ep("cold", backend=BackendType.OLLAMA, gpu=GPUClass.RTX4090,
                          capability_flags={"warm": False, "ollama_loaded_models": []})
        ep_swap = make_ep("swap", backend=BackendType.OLLAMA, gpu=GPUClass.RTX4090,
                          capability_flags={"warm": False,
                                            "ollama_loaded_models": ["other-model:7b"]})

        s_warm, _ = score_ep(ep_warm)
        s_cold, _ = score_ep(ep_cold)
        s_swap, _ = score_ep(ep_swap)

        print(f"\n  Ollama model swap penalty:")
        print(f"    Warm (model loaded):      benchmark={s_warm.benchmark_score:.4f}")
        print(f"    Cold (nothing loaded):     benchmark={s_cold.benchmark_score:.4f}")
        print(f"    Swap (wrong model loaded): benchmark={s_swap.benchmark_score:.4f}")

        assert s_warm.benchmark_score > s_cold.benchmark_score
        assert s_cold.benchmark_score > s_swap.benchmark_score, (
            "Model swap should be worse than cold start (swap has unload + load)"
        )


# ---------------------------------------------------------------------------
# 8. Cross-backend comparison with capabilities
# ---------------------------------------------------------------------------


class TestCrossBackendWithCapabilities:

    def test_full_fleet_scoring_with_capabilities(self):
        fleet = [
            make_ep("nim-h100", backend=BackendType.NIM, gpu=GPUClass.H100,
                     cost_class=CostClass.PREMIUM,
                     capability_flags={"warm": True, "nim_has_metrics": True}),
            make_ep("vllm-l40s", backend=BackendType.VLLM, gpu=GPUClass.L40S,
                     capability_flags={"warm": True, "vllm_gpu_cache_usage": 0.40}),
            make_ep("sglang-l40s", backend=BackendType.SGLANG, gpu=GPUClass.L40S,
                     capability_flags={"warm": True, "sglang_cache_hit_rate": 0.70}),
            make_ep("dynamo-a100", backend=BackendType.DYNAMO, gpu=GPUClass.A100,
                     capability_flags={"warm": True, "dynamo_kv_hit_rate": 0.50,
                                       "dynamo_prefill_queue": 5, "dynamo_decode_queue": 8}),
            make_ep("ollama-rtx", backend=BackendType.OLLAMA, gpu=GPUClass.RTX4090,
                     cost_class=CostClass.ECONOMY,
                     capability_flags={"warm": True}),
        ]

        workloads = [
            ("prefill_heavy", "Summarize: " + "word " * 400, 32),
            ("decode_heavy", "Write a story", 1500),
            ("balanced", "What is Python?", 128),
        ]

        print("\n" + "=" * 100)
        print("  FULL FLEET SCORING WITH CAPABILITIES — ALL BACKENDS")
        print("=" * 100)

        for wl_name, content, max_tok in workloads:
            print(f"\n  Workload: {wl_name}")
            print(f"  {'Endpoint':<18} {'Backend':<10} {'Utility':>10} {'Bench':>8} "
                  f"{'GPU Aff':>10} {'SLO':>8} {'Warm':>6}")
            print("  " + "-" * 76)

            for ep in fleet:
                s, profile = score_ep(ep, content=content, max_tokens=max_tok)
                if not s.hard_rejected:
                    warm = ep.capability_flags.get("warm", "?")
                    print(f"  {ep.name:<18} {ep.backend_type.value:<10} {s.utility_score:>10.4f} "
                          f"{s.benchmark_score:>8.4f} {s.gpu_affinity_score:>10.4f} "
                          f"{s.slo_score:>8.4f} {str(warm):>6}")
                else:
                    print(f"  {ep.name:<18} {ep.backend_type.value:<10} REJECTED: {s.rejection_reason}")
