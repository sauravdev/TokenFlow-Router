"""Benchmark-aware backend scoring helpers.

These are routing-time heuristics, not lab-grade benchmark claims.
The router uses them as priors when fresh endpoint telemetry is missing or
when multiple endpoints have similar live performance.
"""

from __future__ import annotations

from dataclasses import dataclass

from tokenflow.models import BackendType, OptimizationTarget, WorkloadType


@dataclass(frozen=True)
class BackendBenchmark:
    interactive_ttft_ms: float
    decode_tps: float
    prefill_tps: float
    max_concurrency: int
    memory_efficiency: float
    cold_start_penalty_ms: float


@dataclass(frozen=True)
class BackendGuidance:
    strengths: str
    telemetry_source: str
    affinity: dict[WorkloadType, float]


# Conservative priors meant for relative ordering only.
_DEFAULTS: dict[BackendType, BackendBenchmark] = {
    BackendType.NIM: BackendBenchmark(
        interactive_ttft_ms=180.0,
        decode_tps=180.0,
        prefill_tps=9000.0,
        max_concurrency=24,
        memory_efficiency=0.72,
        cold_start_penalty_ms=1200.0,
    ),
    BackendType.VLLM: BackendBenchmark(
        interactive_ttft_ms=230.0,
        decode_tps=260.0,
        prefill_tps=6000.0,
        max_concurrency=64,
        memory_efficiency=0.92,
        cold_start_penalty_ms=900.0,
    ),
    BackendType.SGLANG: BackendBenchmark(
        interactive_ttft_ms=160.0,
        decode_tps=170.0,
        prefill_tps=11000.0,
        max_concurrency=32,
        memory_efficiency=0.80,
        cold_start_penalty_ms=1000.0,
    ),
    BackendType.DYNAMO: BackendBenchmark(
        interactive_ttft_ms=210.0,
        decode_tps=240.0,
        prefill_tps=9500.0,
        max_concurrency=72,
        memory_efficiency=0.88,
        cold_start_penalty_ms=1500.0,
    ),
    BackendType.OLLAMA: BackendBenchmark(
        interactive_ttft_ms=420.0,
        decode_tps=90.0,
        prefill_tps=2800.0,
        max_concurrency=8,
        memory_efficiency=0.55,
        cold_start_penalty_ms=400.0,
    ),
}

_BACKEND_GUIDANCE: dict[BackendType, BackendGuidance] = {
    BackendType.NIM: BackendGuidance(
        strengths="Reasoning, premium SLO",
        telemetry_source="/metrics (nim: prefix)",
        affinity={
            WorkloadType.REASONING: 1.00,
            WorkloadType.PREFILL_HEAVY: 0.90,
            WorkloadType.BALANCED: 0.80,
            WorkloadType.DECODE_HEAVY: 0.70,
        },
    ),
    BackendType.VLLM: BackendGuidance(
        strengths="Decode-heavy, high throughput",
        telemetry_source="/metrics (vllm: prefix)",
        affinity={
            WorkloadType.REASONING: 0.75,
            WorkloadType.PREFILL_HEAVY: 0.70,
            WorkloadType.BALANCED: 0.85,
            WorkloadType.DECODE_HEAVY: 1.00,
        },
    ),
    BackendType.SGLANG: BackendGuidance(
        strengths="Prefill-heavy, KV cache reuse",
        telemetry_source="/get_server_info",
        affinity={
            WorkloadType.REASONING: 0.70,
            WorkloadType.PREFILL_HEAVY: 1.00,
            WorkloadType.BALANCED: 0.85,
            WorkloadType.DECODE_HEAVY: 0.75,
        },
    ),
    BackendType.DYNAMO: BackendGuidance(
        strengths="Both prefill + decode, KV transfer",
        telemetry_source="/metrics (vllm: + dynamo: prefix)",
        affinity={
            WorkloadType.REASONING: 0.85,
            WorkloadType.PREFILL_HEAVY: 0.95,
            WorkloadType.BALANCED: 0.90,
            WorkloadType.DECODE_HEAVY: 0.95,
        },
    ),
    BackendType.OLLAMA: BackendGuidance(
        strengths="Edge/local deployments, low operational overhead",
        telemetry_source="health + lightweight capability probing",
        affinity={
            WorkloadType.REASONING: 0.55,
            WorkloadType.PREFILL_HEAVY: 0.65,
            WorkloadType.BALANCED: 0.72,
            WorkloadType.DECODE_HEAVY: 0.60,
        },
    ),
}


def get_backend_benchmark(backend: BackendType) -> BackendBenchmark:
    return _DEFAULTS[backend]


def get_backend_guidance(backend: BackendType) -> BackendGuidance:
    return _BACKEND_GUIDANCE[backend]


def backend_affinity(backend: BackendType, workload: WorkloadType) -> float:
    return get_backend_guidance(backend).affinity[workload]


def recommended_backend_for_workload(workload: WorkloadType) -> BackendType:
    ranking = sorted(
        _BACKEND_GUIDANCE.items(),
        key=lambda item: item[1].affinity[workload],
        reverse=True,
    )
    return ranking[0][0]


def priority_metric_for_workload(workload: WorkloadType) -> str:
    if workload == WorkloadType.PREFILL_HEAVY:
        return "TTFT"
    if workload == WorkloadType.DECODE_HEAVY:
        return "ITL"
    if workload == WorkloadType.REASONING:
        return "E2E reliability"
    return "E2E + cost"


def benchmark_score(
    backend: BackendType,
    workload: WorkloadType,
    optimization_target: OptimizationTarget,
) -> float:
    bench = get_backend_benchmark(backend)
    affinity = backend_affinity(backend, workload)
    cold_start_inverse = 1.0 - min(bench.cold_start_penalty_ms / 2000.0, 1.0)
    ttft_inverse = 1.0 - min(bench.interactive_ttft_ms / 500.0, 1.0)

    if workload == WorkloadType.REASONING:
        score = 0.45 * affinity + 0.25 * ttft_inverse + 0.20 * (
            bench.prefill_tps / 11000.0
        ) + 0.10 * cold_start_inverse
        return max(0.0, min(1.0, score))

    if optimization_target == OptimizationTarget.THROUGHPUT:
        if workload == WorkloadType.PREFILL_HEAVY:
            score = 0.45 * affinity + 0.35 * (bench.prefill_tps / 11000.0) + 0.20 * (
                bench.max_concurrency / 72.0
            )
        elif workload == WorkloadType.DECODE_HEAVY:
            score = 0.45 * affinity + 0.35 * (bench.decode_tps / 260.0) + 0.20 * (
                bench.max_concurrency / 72.0
            )
        else:
            score = 0.40 * affinity + 0.20 * (bench.prefill_tps / 11000.0) + 0.20 * (
                bench.decode_tps / 260.0
            ) + 0.10 * (bench.max_concurrency / 72.0) + 0.10 * bench.memory_efficiency
    else:
        if workload == WorkloadType.PREFILL_HEAVY:
            score = 0.40 * affinity + 0.35 * ttft_inverse + 0.25 * (
                bench.prefill_tps / 11000.0
            )
        elif workload == WorkloadType.DECODE_HEAVY:
            score = 0.40 * affinity + 0.30 * ttft_inverse + 0.30 * (
                bench.decode_tps / 260.0
            )
        else:
            score = 0.45 * affinity + 0.25 * ttft_inverse + 0.20 * (
                bench.prefill_tps / 11000.0 + bench.decode_tps / 260.0
            ) / 2.0 + 0.10 * bench.memory_efficiency

    return max(0.0, min(1.0, score))
