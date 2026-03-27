from tokenflow.benchmarks import (
    backend_affinity,
    get_backend_guidance,
    priority_metric_for_workload,
    recommended_backend_for_workload,
)
from tokenflow.models import BackendType, WorkloadType


def test_backend_guidance_matches_documented_strengths():
    assert get_backend_guidance(BackendType.NIM).telemetry_source == "/metrics (nim: prefix)"
    assert get_backend_guidance(BackendType.VLLM).telemetry_source == "/metrics (vllm: prefix)"
    assert get_backend_guidance(BackendType.SGLANG).telemetry_source == "/get_server_info"
    assert get_backend_guidance(BackendType.DYNAMO).telemetry_source == "/metrics (vllm: + dynamo: prefix)"


def test_recommended_backend_by_workload():
    assert recommended_backend_for_workload(WorkloadType.PREFILL_HEAVY) == BackendType.SGLANG
    assert recommended_backend_for_workload(WorkloadType.DECODE_HEAVY) == BackendType.VLLM
    assert recommended_backend_for_workload(WorkloadType.REASONING) == BackendType.NIM
    assert recommended_backend_for_workload(WorkloadType.BALANCED) == BackendType.DYNAMO


def test_affinity_matrix_values():
    assert backend_affinity(BackendType.NIM, WorkloadType.REASONING) == 1.00
    assert backend_affinity(BackendType.NIM, WorkloadType.PREFILL_HEAVY) == 0.90
    assert backend_affinity(BackendType.VLLM, WorkloadType.DECODE_HEAVY) == 1.00
    assert backend_affinity(BackendType.SGLANG, WorkloadType.PREFILL_HEAVY) == 1.00
    assert backend_affinity(BackendType.DYNAMO, WorkloadType.BALANCED) == 0.90
    assert backend_affinity(BackendType.DYNAMO, WorkloadType.DECODE_HEAVY) == 0.95


def test_priority_metric_mapping():
    assert priority_metric_for_workload(WorkloadType.PREFILL_HEAVY) == "TTFT"
    assert priority_metric_for_workload(WorkloadType.DECODE_HEAVY) == "ITL"
    assert priority_metric_for_workload(WorkloadType.BALANCED) == "E2E + cost"
    assert priority_metric_for_workload(WorkloadType.REASONING) == "E2E reliability"
