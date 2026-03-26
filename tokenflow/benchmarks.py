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


def get_backend_benchmark(backend: BackendType) -> BackendBenchmark:
    return _DEFAULTS[backend]


def benchmark_score(
    backend: BackendType,
    workload: WorkloadType,
    optimization_target: OptimizationTarget,
) -> float:
    bench = get_backend_benchmark(backend)

    if optimization_target == OptimizationTarget.THROUGHPUT:
        if workload == WorkloadType.PREFILL_HEAVY:
            score = 0.55 * (bench.prefill_tps / 11000.0) + 0.30 * (
                bench.max_concurrency / 72.0
            ) + 0.15 * bench.memory_efficiency
        elif workload == WorkloadType.DECODE_HEAVY:
            score = 0.60 * (bench.decode_tps / 260.0) + 0.25 * (
                bench.max_concurrency / 72.0
            ) + 0.15 * bench.memory_efficiency
        else:
            score = 0.45 * (bench.prefill_tps / 11000.0) + 0.30 * (
                bench.decode_tps / 260.0
            ) + 0.15 * (bench.max_concurrency / 72.0) + 0.10 * bench.memory_efficiency
    else:
        if workload == WorkloadType.PREFILL_HEAVY:
            score = 0.60 * (1.0 - min(bench.interactive_ttft_ms / 500.0, 1.0)) + 0.40 * (
                bench.prefill_tps / 11000.0
            )
        elif workload == WorkloadType.DECODE_HEAVY:
            score = 0.55 * (1.0 - min(bench.interactive_ttft_ms / 500.0, 1.0)) + 0.45 * (
                bench.decode_tps / 260.0
            )
        else:
            score = 0.65 * (1.0 - min(bench.interactive_ttft_ms / 500.0, 1.0)) + 0.35 * (
                bench.memory_efficiency
            )

    return max(0.0, min(1.0, score))
