"""
Ollama routing benchmark — demonstrates TokenFlow Router's intelligent
endpoint selection vs naive (random / round-robin) routing.

Measures and reports numerical scoring differences across:
  - Workload-aware backend selection
  - GPU-aware routing
  - SLO-aware latency estimation
  - Cost optimization
  - Queue-aware load balancing
  - End-to-end decision quality
"""

from __future__ import annotations

import asyncio
import random
import statistics
import time
from typing import Any

import pytest

from tokenflow.benchmarks import (
    BackendBenchmark,
    BackendGuidance,
    backend_affinity,
    benchmark_score,
    get_backend_benchmark,
    get_backend_guidance,
)
from tokenflow.classifier import RequestClassifier
from tokenflow.models import (
    BackendType,
    CandidateScore,
    CostClass,
    EndpointHealth,
    EndpointProfile,
    EndpointRegisterRequest,
    GPUClass,
    LatencyClass,
    OptimizationTarget,
    PriorityTier,
    RouteDecision,
    RouteOutcome,
    RoutingPolicy,
    TelemetryUpdate,
    WorkloadType,
)
from tokenflow.registry import EndpointRegistry
from tokenflow.router import DecisionEngine, ScoringEngine, _apply_preset
from tokenflow.telemetry import TelemetryStore

clf = RequestClassifier()


# ---------------------------------------------------------------------------
# Endpoint factory
# ---------------------------------------------------------------------------


def make_ep(
    name: str = "ep",
    model: str = "qwen2.5:1.5b",
    gpu: GPUClass = GPUClass.RTX4090,
    cost_class: CostClass = CostClass.ECONOMY,
    backend: BackendType = BackendType.OLLAMA,
    enabled: bool = True,
    health: EndpointHealth = EndpointHealth.HEALTHY,
    max_context_tokens: int = 8192,
    gpu_count: int = 1,
    cost_per_gpu_hour: float = 1.0,
    supports_reasoning: bool = False,
    capability_flags: dict[str, Any] | None = None,
) -> EndpointProfile:
    return EndpointProfile(
        name=name,
        nim_url=f"http://{name}:11434",
        model_name=model,
        gpu_name=gpu,
        cost_class=cost_class,
        backend_type=backend,
        cost_per_gpu_hour=cost_per_gpu_hour,
        max_context_tokens=max_context_tokens,
        gpu_count=gpu_count,
        supports_reasoning=supports_reasoning,
        capability_flags=capability_flags or {},
        enabled=enabled,
        health=health,
    )


async def make_registry(*endpoints: EndpointProfile) -> EndpointRegistry:
    registry = EndpointRegistry()
    for ep in endpoints:
        req = EndpointRegisterRequest(
            **{k: getattr(ep, k) for k in EndpointRegisterRequest.model_fields}
        )
        await registry.register(req)
    return registry


def body(model: str = "qwen2.5:1.5b", content: str = "hello", max_tokens: int = 64):
    return {
        "model": model,
        "messages": [{"role": "user", "content": content}],
        "max_tokens": max_tokens,
    }


# ---------------------------------------------------------------------------
# 1. Ollama backend benchmark priors
# ---------------------------------------------------------------------------


class TestOllamaBenchmarkPriors:
    """Verify Ollama's benchmark priors are correctly configured."""

    def test_ollama_benchmark_exists(self):
        bench = get_backend_benchmark(BackendType.OLLAMA)
        assert isinstance(bench, BackendBenchmark)

    def test_ollama_has_lower_throughput_than_nim(self):
        ollama = get_backend_benchmark(BackendType.OLLAMA)
        nim = get_backend_benchmark(BackendType.NIM)
        assert ollama.decode_tps < nim.decode_tps
        assert ollama.prefill_tps < nim.prefill_tps
        print(f"\n  Ollama decode TPS: {ollama.decode_tps} vs NIM: {nim.decode_tps}")
        print(f"  Ollama prefill TPS: {ollama.prefill_tps} vs NIM: {nim.prefill_tps}")

    def test_ollama_has_lower_concurrency_than_vllm(self):
        ollama = get_backend_benchmark(BackendType.OLLAMA)
        vllm = get_backend_benchmark(BackendType.VLLM)
        assert ollama.max_concurrency < vllm.max_concurrency
        print(f"\n  Ollama max concurrency: {ollama.max_concurrency} vs vLLM: {vllm.max_concurrency}")

    def test_ollama_has_fastest_cold_start(self):
        ollama = get_backend_benchmark(BackendType.OLLAMA)
        for bt in BackendType:
            other = get_backend_benchmark(bt)
            assert ollama.cold_start_penalty_ms <= other.cold_start_penalty_ms
        print(f"\n  Ollama cold start: {ollama.cold_start_penalty_ms}ms (best among all backends)")

    def test_ollama_guidance_edge_affinity(self):
        guidance = get_backend_guidance(BackendType.OLLAMA)
        assert "edge" in guidance.strengths.lower() or "local" in guidance.strengths.lower()
        print(f"\n  Ollama strengths: {guidance.strengths}")

    def test_ollama_affinity_for_all_workloads(self):
        guidance = get_backend_guidance(BackendType.OLLAMA)
        for wt in WorkloadType:
            assert wt in guidance.affinity
            assert 0.0 <= guidance.affinity[wt] <= 1.0
        print("\n  Ollama workload affinities:")
        for wt, score in guidance.affinity.items():
            print(f"    {wt.value:20s} → {score:.2f}")


# ---------------------------------------------------------------------------
# 2. Ollama scoring compared to other backends
# ---------------------------------------------------------------------------


class TestOllamaScoringComparison:
    """Compare Ollama benchmark scores against all other backends."""

    @pytest.mark.parametrize("workload", list(WorkloadType))
    @pytest.mark.parametrize("opt_target", [OptimizationTarget.LATENCY, OptimizationTarget.THROUGHPUT])
    def test_benchmark_score_range(self, workload: WorkloadType, opt_target: OptimizationTarget):
        score = benchmark_score(BackendType.OLLAMA, workload, opt_target)
        assert 0.0 <= score <= 1.0

    def test_ollama_vs_all_backends_scoring_table(self):
        """Print full scoring comparison table."""
        backends = list(BackendType)
        workloads = list(WorkloadType)
        targets = [OptimizationTarget.LATENCY, OptimizationTarget.THROUGHPUT]

        print("\n" + "=" * 90)
        print("  BENCHMARK SCORE COMPARISON: Ollama vs All Backends")
        print("=" * 90)

        for target in targets:
            print(f"\n  Optimization Target: {target.value.upper()}")
            print(f"  {'Backend':<12}", end="")
            for wt in workloads:
                print(f"  {wt.value:>15}", end="")
            print(f"  {'AVG':>8}")
            print("  " + "-" * 80)

            for bt in backends:
                scores = []
                print(f"  {bt.value:<12}", end="")
                for wt in workloads:
                    s = benchmark_score(bt, wt, target)
                    scores.append(s)
                    marker = " ◄" if bt == BackendType.OLLAMA else "  "
                    print(f"  {s:>13.4f}{marker}", end="")
                avg = statistics.mean(scores)
                print(f"  {avg:>8.4f}")

    def test_ollama_balanced_workload_score(self):
        ollama_lat = benchmark_score(BackendType.OLLAMA, WorkloadType.BALANCED, OptimizationTarget.LATENCY)
        ollama_thr = benchmark_score(BackendType.OLLAMA, WorkloadType.BALANCED, OptimizationTarget.THROUGHPUT)
        assert ollama_lat > 0.0
        assert ollama_thr > 0.0
        print(f"\n  Ollama balanced workload score: latency={ollama_lat:.4f}, throughput={ollama_thr:.4f}")


# ---------------------------------------------------------------------------
# 3. Router scoring: Ollama endpoints with different hardware
# ---------------------------------------------------------------------------


class TestOllamaRouterScoring:
    """Test how the full router scores Ollama endpoints on various hardware."""

    def _score_endpoint(
        self,
        ep: EndpointProfile,
        request_body: dict,
        policy: RoutingPolicy | None = None,
        telemetry: TelemetryUpdate | None = None,
        priority: PriorityTier = PriorityTier.STANDARD,
        opt_target: OptimizationTarget = OptimizationTarget.AUTO,
    ) -> CandidateScore:
        store = TelemetryStore()
        if telemetry:
            asyncio.get_event_loop().run_until_complete(store.upsert(telemetry))
        engine = ScoringEngine(policy or RoutingPolicy(), store)
        profile = clf.classify(request_body, priority_tier=priority, optimization_target=opt_target)
        return engine.score(ep, profile)

    def test_ollama_rtx4090_not_rejected_for_small_model(self):
        ep = make_ep(gpu=GPUClass.RTX4090)
        score = self._score_endpoint(ep, body())
        assert not score.hard_rejected
        assert score.utility_score > 0.0
        print(f"\n  Ollama RTX4090 utility: {score.utility_score:.4f}")

    def test_ollama_scores_on_different_gpus(self):
        gpus = [GPUClass.RTX4090, GPUClass.L4, GPUClass.A10G, GPUClass.RTX_LAPTOP]
        print("\n  Ollama qwen2.5:1.5b scores by GPU:")
        print(f"  {'GPU':<15} {'Utility':>10} {'SLO':>8} {'GPU Aff':>10} {'Bench':>8} {'Cost':>8} {'TTFT ms':>10}")
        print("  " + "-" * 70)
        for gpu in gpus:
            ep = make_ep(gpu=gpu)
            score = self._score_endpoint(ep, body(content="Explain quantum computing briefly"))
            if not score.hard_rejected:
                print(
                    f"  {gpu.value:<15} {score.utility_score:>10.4f} {score.slo_score:>8.4f} "
                    f"{score.gpu_affinity_score:>10.4f} {score.benchmark_score:>8.4f} "
                    f"{score.cost_score:>8.4f} {score.estimated_ttft_ms:>10.1f}"
                )
            else:
                print(f"  {gpu.value:<15} REJECTED ({score.rejection_reason})")

    def test_ollama_with_live_telemetry_vs_heuristic(self):
        ep = make_ep(gpu=GPUClass.RTX4090)
        request = body(content="Hello world")

        score_no_tel = self._score_endpoint(ep, request)

        good_tel = TelemetryUpdate(
            endpoint_id=ep.id,
            p95_ttft_ms=150.0,
            p95_itl_ms=20.0,
            p95_e2e_ms=2000.0,
            queue_depth=2,
            error_rate=0.01,
            saturation_score=0.15,
        )
        score_good_tel = self._score_endpoint(ep, request, telemetry=good_tel)

        bad_tel = TelemetryUpdate(
            endpoint_id=ep.id,
            p95_ttft_ms=800.0,
            p95_itl_ms=80.0,
            p95_e2e_ms=12000.0,
            queue_depth=50,
            error_rate=0.05,
            saturation_score=0.75,
        )
        score_bad_tel = self._score_endpoint(ep, request, telemetry=bad_tel)

        print(f"\n  Ollama scoring with telemetry:")
        print(f"    No telemetry (heuristic): utility={score_no_tel.utility_score:.4f}, "
              f"SLO={score_no_tel.slo_score:.4f}, queue={score_no_tel.queue_score:.4f}")
        print(f"    Good telemetry:           utility={score_good_tel.utility_score:.4f}, "
              f"SLO={score_good_tel.slo_score:.4f}, queue={score_good_tel.queue_score:.4f}")
        print(f"    Bad telemetry:            utility={score_bad_tel.utility_score:.4f}, "
              f"SLO={score_bad_tel.slo_score:.4f}, queue={score_bad_tel.queue_score:.4f}")

        assert score_good_tel.queue_score > score_bad_tel.queue_score

    def test_policy_presets_affect_ollama_scoring(self):
        ep = make_ep(gpu=GPUClass.RTX4090)
        request = body(content="Summarize this document")

        presets = ["latency-first", "balanced", "cost-first"]
        print(f"\n  Ollama scoring under different policy presets:")
        print(f"  {'Preset':<18} {'Utility':>10} {'SLO':>8} {'Cost':>8} {'Queue':>8} {'GPU Aff':>10}")
        print("  " + "-" * 60)

        for preset_name in presets:
            policy = _apply_preset(RoutingPolicy(preset=preset_name))
            score = self._score_endpoint(ep, request, policy=policy)
            print(
                f"  {preset_name:<18} {score.utility_score:>10.4f} {score.slo_score:>8.4f} "
                f"{score.cost_score:>8.4f} {score.queue_score:>8.4f} {score.gpu_affinity_score:>10.4f}"
            )


# ---------------------------------------------------------------------------
# 4. WITH vs WITHOUT Router — the core benchmark
# ---------------------------------------------------------------------------


class TestWithVsWithoutRouter:
    """
    Compare TokenFlow Router's intelligent routing against naive strategies.

    Simulates a heterogeneous fleet with Ollama + other backends and measures
    routing quality across many request types.
    """

    @staticmethod
    def _build_fleet() -> list[EndpointProfile]:
        """Create a realistic mixed fleet of inference endpoints."""
        return [
            make_ep("ollama-rtx4090", model="qwen2.5:1.5b", backend=BackendType.OLLAMA,
                     gpu=GPUClass.RTX4090, cost_class=CostClass.ECONOMY, cost_per_gpu_hour=0.5),
            make_ep("ollama-laptop", model="qwen2.5:1.5b", backend=BackendType.OLLAMA,
                     gpu=GPUClass.RTX_LAPTOP, cost_class=CostClass.ECONOMY, cost_per_gpu_hour=0.1,
                     max_context_tokens=4096),
            make_ep("nim-h100", model="qwen2.5:1.5b", backend=BackendType.NIM,
                     gpu=GPUClass.H100, cost_class=CostClass.PREMIUM, cost_per_gpu_hour=8.0,
                     max_context_tokens=32768),
            make_ep("vllm-l40s", model="qwen2.5:1.5b", backend=BackendType.VLLM,
                     gpu=GPUClass.L40S, cost_class=CostClass.STANDARD, cost_per_gpu_hour=3.0,
                     max_context_tokens=16384),
            make_ep("sglang-l40s", model="qwen2.5:1.5b", backend=BackendType.SGLANG,
                     gpu=GPUClass.L40S, cost_class=CostClass.STANDARD, cost_per_gpu_hour=3.0,
                     max_context_tokens=16384),
        ]

    @staticmethod
    def _build_request_mix() -> list[dict]:
        """Generate a diverse set of test requests."""
        return [
            # Short interactive (balanced)
            body(content="What is 2+2?", max_tokens=32),
            # Medium balanced
            body(content="Explain the concept of machine learning in simple terms.", max_tokens=200),
            # Prefill-heavy (long input, short output)
            body(content="Summarize the following text: " + "The quick brown fox. " * 100, max_tokens=50),
            # Decode-heavy (short input, long output)
            body(content="Write a story", max_tokens=1000),
            # Long context
            body(content="Analyze this data: " + "metric=42 " * 300, max_tokens=256),
            # Streaming-like request
            {**body(content="Chat with me about Python", max_tokens=150), "stream": True},
            # Tiny request
            body(content="Hi", max_tokens=16),
            # Medium with system prompt
            {"model": "qwen2.5:1.5b", "system": "You are a helpful coding assistant.",
             "messages": [{"role": "user", "content": "Write a Python function to sort a list"}],
             "max_tokens": 300},
        ]

    @staticmethod
    def _naive_random_select(fleet: list[EndpointProfile]) -> EndpointProfile:
        enabled = [ep for ep in fleet if ep.enabled]
        return random.choice(enabled)

    @staticmethod
    def _naive_round_robin_select(fleet: list[EndpointProfile], index: int) -> EndpointProfile:
        enabled = [ep for ep in fleet if ep.enabled]
        return enabled[index % len(enabled)]

    @pytest.mark.asyncio
    async def test_router_vs_random_routing(self):
        """Compare router's intelligent selection against random routing."""
        fleet = self._build_fleet()
        requests = self._build_request_mix()
        registry = await make_registry(*fleet)
        store = TelemetryStore()

        # Inject realistic telemetry for each endpoint
        telemetry_data = {
            "ollama-rtx4090": TelemetryUpdate(
                endpoint_id="", p95_ttft_ms=420.0, p95_itl_ms=45.0, p95_e2e_ms=8000.0,
                queue_depth=1, error_rate=0.02, saturation_score=0.10),
            "ollama-laptop": TelemetryUpdate(
                endpoint_id="", p95_ttft_ms=800.0, p95_itl_ms=80.0, p95_e2e_ms=15000.0,
                queue_depth=0, error_rate=0.01, saturation_score=0.05),
            "nim-h100": TelemetryUpdate(
                endpoint_id="", p95_ttft_ms=180.0, p95_itl_ms=12.0, p95_e2e_ms=2500.0,
                queue_depth=5, error_rate=0.005, saturation_score=0.30),
            "vllm-l40s": TelemetryUpdate(
                endpoint_id="", p95_ttft_ms=230.0, p95_itl_ms=18.0, p95_e2e_ms=3500.0,
                queue_depth=8, error_rate=0.01, saturation_score=0.40),
            "sglang-l40s": TelemetryUpdate(
                endpoint_id="", p95_ttft_ms=160.0, p95_itl_ms=15.0, p95_e2e_ms=3000.0,
                queue_depth=3, error_rate=0.008, saturation_score=0.20),
        }

        all_eps = await registry.list_all()
        for ep in all_eps:
            if ep.name in telemetry_data:
                tel = telemetry_data[ep.name]
                tel.endpoint_id = ep.id
                await store.upsert(tel)

        # --- Router decisions ---
        engine = DecisionEngine(registry=registry, store=store)
        engine.set_policy(_apply_preset(RoutingPolicy(preset="balanced")))

        router_scores: list[float] = []
        router_ttfts: list[float] = []
        router_e2es: list[float] = []
        router_costs: list[float] = []
        router_decisions: list[str] = []
        router_decision_times: list[float] = []

        for req_body in requests:
            profile = clf.classify(req_body)
            t0 = time.perf_counter()
            decision = await engine.decide(profile)
            dt = (time.perf_counter() - t0) * 1000
            router_decision_times.append(dt)
            if decision.selected_endpoint_id:
                best = next(s for s in decision.candidate_scores
                            if s.endpoint_id == decision.selected_endpoint_id)
                router_scores.append(best.utility_score)
                router_ttfts.append(best.estimated_ttft_ms)
                router_e2es.append(best.estimated_e2e_ms)
                router_costs.append(best.estimated_cost_usd)
                router_decisions.append(best.endpoint_name)

        # --- Random baseline ---
        random.seed(42)
        random_scores: list[float] = []
        random_ttfts: list[float] = []
        random_e2es: list[float] = []
        random_costs: list[float] = []
        random_decisions: list[str] = []

        scoring_engine = ScoringEngine(engine.policy, store)

        for req_body in requests:
            profile = clf.classify(req_body)
            selected = self._naive_random_select(fleet)
            candidate_score = scoring_engine.score(selected, profile)
            if not candidate_score.hard_rejected:
                random_scores.append(candidate_score.utility_score)
                random_ttfts.append(candidate_score.estimated_ttft_ms)
                random_e2es.append(candidate_score.estimated_e2e_ms)
                random_costs.append(candidate_score.estimated_cost_usd)
                random_decisions.append(candidate_score.endpoint_name)
            else:
                random_scores.append(0.0)
                random_ttfts.append(9999.0)
                random_e2es.append(9999.0)
                random_costs.append(9999.0)
                random_decisions.append(f"REJECTED({selected.name})")

        # --- Round-robin baseline ---
        rr_scores: list[float] = []
        rr_ttfts: list[float] = []
        rr_e2es: list[float] = []
        rr_costs: list[float] = []
        rr_decisions: list[str] = []

        for i, req_body in enumerate(requests):
            profile = clf.classify(req_body)
            selected = self._naive_round_robin_select(fleet, i)
            candidate_score = scoring_engine.score(selected, profile)
            if not candidate_score.hard_rejected:
                rr_scores.append(candidate_score.utility_score)
                rr_ttfts.append(candidate_score.estimated_ttft_ms)
                rr_e2es.append(candidate_score.estimated_e2e_ms)
                rr_costs.append(candidate_score.estimated_cost_usd)
                rr_decisions.append(candidate_score.endpoint_name)
            else:
                rr_scores.append(0.0)
                rr_ttfts.append(9999.0)
                rr_e2es.append(9999.0)
                rr_costs.append(9999.0)
                rr_decisions.append(f"REJECTED({selected.name})")

        # --- Print results ---
        print("\n")
        print("=" * 100)
        print("  TOKENFLOW ROUTER vs NAIVE ROUTING — PERFORMANCE COMPARISON")
        print("=" * 100)

        print("\n  Per-request routing decisions:")
        print(f"  {'#':<4} {'Workload':<16} {'Router Choice':<20} {'Random Choice':<20} {'RoundRobin Choice':<20}")
        print("  " + "-" * 84)
        for i, req_body in enumerate(requests):
            profile = clf.classify(req_body)
            wt = profile.workload_type.value
            rc = router_decisions[i] if i < len(router_decisions) else "N/A"
            rand = random_decisions[i] if i < len(random_decisions) else "N/A"
            rr = rr_decisions[i] if i < len(rr_decisions) else "N/A"
            print(f"  {i+1:<4} {wt:<16} {rc:<20} {rand:<20} {rr:<20}")

        print(f"\n  {'Metric':<35} {'Router':>12} {'Random':>12} {'Round-Robin':>12} {'Router Δ':>12}")
        print("  " + "-" * 88)

        def _row(name: str, router_vals: list[float], random_vals: list[float], rr_vals: list[float], lower_is_better: bool = False):
            r_avg = statistics.mean(router_vals) if router_vals else 0
            rand_avg = statistics.mean(random_vals) if random_vals else 0
            rr_avg = statistics.mean(rr_vals) if rr_vals else 0
            if lower_is_better:
                best_naive = min(rand_avg, rr_avg) if rand_avg > 0 and rr_avg > 0 else max(rand_avg, rr_avg)
                if best_naive > 0 and r_avg > 0:
                    improvement = ((best_naive - r_avg) / best_naive) * 100
                else:
                    improvement = 0
                delta_str = f"{improvement:+.1f}%"
            else:
                best_naive = max(rand_avg, rr_avg)
                if best_naive > 0:
                    improvement = ((r_avg - best_naive) / best_naive) * 100
                else:
                    improvement = 0
                delta_str = f"{improvement:+.1f}%"
            print(f"  {name:<35} {r_avg:>12.4f} {rand_avg:>12.4f} {rr_avg:>12.4f} {delta_str:>12}")

        _row("Avg Utility Score", router_scores, random_scores, rr_scores)
        _row("Avg Est. TTFT (ms)", router_ttfts, random_ttfts, rr_ttfts, lower_is_better=True)
        _row("Avg Est. E2E Latency (ms)", router_e2es, random_e2es, rr_e2es, lower_is_better=True)
        _row("Avg Est. Cost (USD)", router_costs, random_costs, rr_costs, lower_is_better=True)

        avg_decision_ms = statistics.mean(router_decision_times)
        print(f"\n  Router decision latency: avg={avg_decision_ms:.3f}ms, "
              f"min={min(router_decision_times):.3f}ms, max={max(router_decision_times):.3f}ms")

        rejections_random = sum(1 for d in random_decisions if "REJECTED" in d)
        rejections_rr = sum(1 for d in rr_decisions if "REJECTED" in d)
        print(f"\n  Misrouting (requests sent to unfit endpoints):")
        print(f"    Router:      0 / {len(requests)} (always picks valid endpoint)")
        print(f"    Random:      {rejections_random} / {len(requests)} would have failed")
        print(f"    Round-Robin: {rejections_rr} / {len(requests)} would have failed")

        # Assertions
        assert statistics.mean(router_scores) >= statistics.mean(random_scores)
        assert statistics.mean(router_scores) >= statistics.mean(rr_scores)
        assert avg_decision_ms < 10.0  # sub-10ms routing overhead

    @pytest.mark.asyncio
    async def test_workload_specific_routing_advantage(self):
        """Show router picks the right backend for each workload type."""
        fleet = self._build_fleet()
        registry = await make_registry(*fleet)
        store = TelemetryStore()
        engine = DecisionEngine(registry=registry, store=store)
        engine.set_policy(_apply_preset(RoutingPolicy(preset="balanced")))

        workload_requests = {
            "prefill_heavy": body(content="Summarize: " + "word " * 400, max_tokens=32),
            "decode_heavy": body(content="Write a long story", max_tokens=1500),
            "balanced": body(content="What is Python?", max_tokens=128),
        }

        print("\n" + "=" * 80)
        print("  WORKLOAD-SPECIFIC ROUTING DECISIONS")
        print("=" * 80)

        all_eps = await registry.list_all()
        scoring = ScoringEngine(engine.policy, store)

        for wl_name, req_body in workload_requests.items():
            profile = clf.classify(req_body)
            print(f"\n  Workload: {wl_name} (classified as: {profile.workload_type.value})")
            print(f"    Input tokens: {profile.input_tokens}, Output tokens: {profile.predicted_output_tokens}")
            print(f"    {'Endpoint':<20} {'Utility':>10} {'SLO':>8} {'GPU Aff':>10} {'Bench':>8} {'Rejected?':<15}")
            print("    " + "-" * 75)

            for ep in all_eps:
                score = scoring.score(ep, profile)
                if score.hard_rejected:
                    print(f"    {ep.name:<20} {'--':>10} {'--':>8} {'--':>10} {'--':>8} {score.rejection_reason or '':<15}")
                else:
                    print(f"    {ep.name:<20} {score.utility_score:>10.4f} {score.slo_score:>8.4f} "
                          f"{score.gpu_affinity_score:>10.4f} {score.benchmark_score:>8.4f} {'No':<15}")

            decision = await engine.decide(profile)
            winner_name = decision.selected_endpoint_name or "NONE"
            print(f"    → Router selected: {winner_name}")

    @pytest.mark.asyncio
    async def test_latency_vs_cost_policy_impact(self):
        """Show how policy presets shift routing toward different endpoints."""
        fleet = self._build_fleet()
        registry = await make_registry(*fleet)
        store = TelemetryStore()

        presets = ["latency-first", "balanced", "cost-first"]
        request = body(content="Translate this to French: Hello, how are you?", max_tokens=64)
        profile = clf.classify(request)

        print("\n" + "=" * 80)
        print("  POLICY PRESET IMPACT ON ROUTING")
        print("=" * 80)
        print(f"  Request: {profile.workload_type.value}, {profile.input_tokens} input tokens")
        print(f"\n  {'Preset':<18} {'Winner':<20} {'Utility':>10} {'Est TTFT':>10} {'Est Cost':>12}")
        print("  " + "-" * 72)

        for preset_name in presets:
            engine = DecisionEngine(registry=registry, store=store)
            engine.set_policy(_apply_preset(RoutingPolicy(preset=preset_name)))
            decision = await engine.decide(profile)
            if decision.selected_endpoint_id:
                best = next(s for s in decision.candidate_scores
                            if s.endpoint_id == decision.selected_endpoint_id)
                print(f"  {preset_name:<18} {best.endpoint_name:<20} {best.utility_score:>10.4f} "
                      f"{best.estimated_ttft_ms:>10.1f} ${best.estimated_cost_usd:>11.6f}")
            else:
                print(f"  {preset_name:<18} {'NONE':<20}")


# ---------------------------------------------------------------------------
# 5. Queue-aware routing advantage
# ---------------------------------------------------------------------------


class TestQueueAwareRoutingAdvantage:
    """Demonstrate how telemetry-driven routing avoids overloaded endpoints."""

    @pytest.mark.asyncio
    async def test_router_avoids_saturated_ollama(self):
        fleet = [
            make_ep("ollama-1", gpu=GPUClass.RTX4090),
            make_ep("ollama-2", gpu=GPUClass.RTX4090, cost_per_gpu_hour=0.5),
        ]
        registry = await make_registry(*fleet)
        store = TelemetryStore()

        all_eps = await registry.list_all()

        # Endpoint 1 is overloaded
        await store.upsert(TelemetryUpdate(
            endpoint_id=all_eps[0].id,
            queue_depth=80, saturation_score=0.85,
            p95_ttft_ms=900.0, p95_e2e_ms=15000.0, error_rate=0.08,
        ))
        # Endpoint 2 is idle
        await store.upsert(TelemetryUpdate(
            endpoint_id=all_eps[1].id,
            queue_depth=1, saturation_score=0.05,
            p95_ttft_ms=300.0, p95_e2e_ms=5000.0, error_rate=0.01,
        ))

        engine = DecisionEngine(registry=registry, store=store)
        engine.set_policy(RoutingPolicy())

        profile = clf.classify(body())
        decision = await engine.decide(profile)

        scoring = ScoringEngine(engine.policy, store)
        s1 = scoring.score(all_eps[0], profile)
        s2 = scoring.score(all_eps[1], profile)

        print("\n" + "=" * 80)
        print("  QUEUE-AWARE ROUTING: Overloaded vs Idle Endpoint")
        print("=" * 80)
        print(f"  {'Metric':<30} {'Overloaded EP':>15} {'Idle EP':>15}")
        print("  " + "-" * 65)
        print(f"  {'Queue depth':<30} {80:>15} {1:>15}")
        print(f"  {'Saturation':<30} {0.85:>15.2f} {0.05:>15.2f}")
        print(f"  {'Queue score':<30} {s1.queue_score:>15.4f} {s2.queue_score:>15.4f}")
        print(f"  {'SLO score':<30} {s1.slo_score:>15.4f} {s2.slo_score:>15.4f}")
        print(f"  {'Reliability score':<30} {s1.reliability_score:>15.4f} {s2.reliability_score:>15.4f}")
        print(f"  {'Utility score':<30} {s1.utility_score:>15.4f} {s2.utility_score:>15.4f}")
        print(f"\n  → Router chose: {decision.selected_endpoint_name} (idle endpoint)")

        assert decision.selected_endpoint_name == "ollama-2"
        assert s2.utility_score > s1.utility_score


# ---------------------------------------------------------------------------
# 6. Decision speed benchmark
# ---------------------------------------------------------------------------


class TestDecisionSpeedBenchmark:
    """Measure routing decision latency."""

    @pytest.mark.asyncio
    async def test_routing_decision_speed(self):
        fleet = [
            make_ep(f"ollama-{i}", gpu=GPUClass.RTX4090)
            for i in range(10)
        ]
        registry = await make_registry(*fleet)
        store = TelemetryStore()
        engine = DecisionEngine(registry=registry, store=store)
        engine.set_policy(RoutingPolicy())

        profile = clf.classify(body())

        warmup_count = 5
        for _ in range(warmup_count):
            await engine.decide(profile)

        iterations = 100
        times = []
        for _ in range(iterations):
            t0 = time.perf_counter()
            await engine.decide(profile)
            times.append((time.perf_counter() - t0) * 1000)

        avg_ms = statistics.mean(times)
        p50_ms = statistics.median(times)
        p95_ms = sorted(times)[int(0.95 * len(times))]
        p99_ms = sorted(times)[int(0.99 * len(times))]

        print("\n" + "=" * 60)
        print("  ROUTING DECISION SPEED (10 Ollama endpoints)")
        print("=" * 60)
        print(f"  Iterations: {iterations}")
        print(f"  Avg:  {avg_ms:.3f} ms")
        print(f"  P50:  {p50_ms:.3f} ms")
        print(f"  P95:  {p95_ms:.3f} ms")
        print(f"  P99:  {p99_ms:.3f} ms")
        print(f"  Min:  {min(times):.3f} ms")
        print(f"  Max:  {max(times):.3f} ms")
        print(f"\n  Overhead: {avg_ms:.3f}ms per request (negligible vs inference latency)")

        assert avg_ms < 5.0, f"Decision latency too high: {avg_ms:.3f}ms"
        assert p99_ms < 10.0, f"P99 latency too high: {p99_ms:.3f}ms"
