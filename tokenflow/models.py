"""Core data models for TokenFlow Router."""

from __future__ import annotations

import uuid
from datetime import datetime
from enum import Enum
from typing import Any, Optional

from pydantic import BaseModel, Field, field_validator


# ---------------------------------------------------------------------------
# Enumerations
# ---------------------------------------------------------------------------


class GPUClass(str, Enum):
    # Blackwell generation
    B200 = "B200"               # NVIDIA Blackwell B200 — highest compute, 192GB HBM3e
    # Hopper generation
    H200 = "H200"               # H200 SXM5 — 141GB HBM3e, best memory bandwidth for decode
    H100 = "H100"               # H100 SXM5/PCIe — top-tier inference, best prefill
    # Ampere generation
    A100 = "A100"               # A100 80GB — strong all-rounder
    # Ada Lovelace / Ampere datacenter
    L40S = "L40S"               # L40S 48GB — strong standard lane
    L40 = "L40"                 # L40 48GB — previous generation Ada
    # Workstation / prosumer
    RTX_PRO_6000 = "RTX_PRO_6000"  # RTX Pro 6000 Blackwell — 96GB GDDR7, fits 70B models
    # Datacenter economy
    A10G = "A10G"               # A10G 24GB — economy datacenter
    L4 = "L4"                   # L4 24GB Ada — economy, low-power
    # Consumer / edge
    RTX4090 = "RTX4090"         # RTX 4090 24GB — prosumer, limited VRAM
    RTX_LAPTOP = "RTX_LAPTOP"   # Mobile RTX GPU 8–16GB — edge/small models only
    RTX3090 = "RTX3090"         # RTX 3090 24GB — older consumer
    # CPU fallback
    CPU = "CPU"                 # CPU-only inference — batch/offline tiny models only
    # Managed frontier APIs — no local GPU, billed per-token by the provider
    FRONTIER_API = "FRONTIER_API"  # OpenAI / Anthropic / OpenRouter / xAI managed endpoint
    UNKNOWN = "UNKNOWN"


class BackendType(str, Enum):
    """Inference serving backend used by an endpoint."""
    NIM = "nim"         # NVIDIA NIM (TensorRT-LLM / Triton) — fastest prefill, best for reasoning
    VLLM = "vllm"       # vLLM PagedAttention — excellent decode throughput, memory-efficient batching
    SGLANG = "sglang"   # SGLang RadixAttention — best for prefix-heavy/shared-prefix (RAG, long context)
    DYNAMO = "dynamo"   # NVIDIA Dynamo — distributed KV-aware routing across worker pools
    OLLAMA = "ollama"   # Ollama — lightweight local serving, good cold-start and edge deployments
    OPENAI = "openai"   # Frontier API endpoint — OpenAI / Anthropic / OpenRouter / xAI; per-token billing


class CostClass(str, Enum):
    PREMIUM = "premium"     # B200 / H200 / H100 / A100
    STANDARD = "standard"   # L40S / L40 / RTX_PRO_6000 / A10G
    ECONOMY = "economy"     # L4 / RTX4090 / RTX_LAPTOP / RTX3090 / CPU


class PriorityTier(str, Enum):
    PREMIUM = "premium"
    STANDARD = "standard"
    BATCH = "batch"
    OFFLINE = "offline"


class LatencyClass(str, Enum):
    INTERACTIVE = "interactive"   # p95 TTFT < 500ms, E2E < 5s
    STANDARD = "standard"         # p95 TTFT < 2s, E2E < 30s
    BATCH = "batch"               # best-effort, minutes OK
    OFFLINE = "offline"           # scheduled, async


class WorkloadType(str, Enum):
    PREFILL_HEAVY = "prefill_heavy"   # long input, short output
    DECODE_HEAVY = "decode_heavy"     # short input, long output
    BALANCED = "balanced"             # moderate both
    REASONING = "reasoning"           # complex multi-step


class OptimizationTarget(str, Enum):
    LATENCY = "latency"
    THROUGHPUT = "throughput"
    AUTO = "auto"


class TokenBand(str, Enum):
    TINY = "tiny"       # < 256
    SMALL = "small"     # 256 – 1K
    MEDIUM = "medium"   # 1K – 4K
    LARGE = "large"     # 4K – 16K
    XLARGE = "xlarge"   # > 16K


class EndpointHealth(str, Enum):
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"
    UNKNOWN = "unknown"


class RouteOutcome(str, Enum):
    SUCCESS = "success"
    FALLBACK_USED = "fallback_used"
    FAILED = "failed"
    REJECTED = "rejected"


# ---------------------------------------------------------------------------
# Endpoint
# ---------------------------------------------------------------------------


class EndpointProfile(BaseModel):
    """Registered inference endpoint with static metadata (NIM, vLLM, SGLang, or Dynamo)."""

    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    name: str = Field(..., description="Human-readable name, e.g. 'nim-h100-llama3'")
    nim_url: str = Field(..., description="Base URL of the inference endpoint")
    backend_type: BackendType = Field(default=BackendType.NIM, description="Serving backend: nim | vllm | sglang | dynamo")
    model_name: str = Field(..., description="Model served, e.g. 'meta/llama-3.1-8b-instruct'")
    model_family: str = Field(default="", description="Model family, e.g. 'llama3'")
    model_version: str = Field(default="latest")
    gpu_name: GPUClass = Field(default=GPUClass.UNKNOWN)
    gpu_count: int = Field(default=1, ge=1)
    region: str = Field(default="us-east-1")
    cost_class: CostClass = Field(default=CostClass.STANDARD)
    max_context_tokens: int = Field(default=8192)
    supports_streaming: bool = True
    supports_reasoning: bool = False
    tenant_tags: list[str] = Field(default_factory=list)
    capability_flags: dict[str, Any] = Field(default_factory=dict)
    # Cost config (USD per GPU-hour). For OPENAI-backed endpoints the
    # router converts per-token pricing into an effective hourly rate via
    # `capability_flags["price_per_1k_input_usd"]` and `..._output_usd`.
    cost_per_gpu_hour: float = Field(default=3.0)
    # Auth — used to inject Authorization: Bearer <api_key> on outbound
    # forwards. Only set for OPENAI / frontier-API endpoints. Excluded
    # from JSON responses so the key never leaks via /admin/endpoints.
    api_key: Optional[str] = Field(default=None, exclude=True, repr=False)
    # State
    health: EndpointHealth = Field(default=EndpointHealth.UNKNOWN)
    enabled: bool = True
    registered_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)

    @field_validator("nim_url")
    @classmethod
    def strip_trailing_slash(cls, v: str) -> str:
        return v.rstrip("/")

    @property
    def completions_url(self) -> str:
        return f"{self.nim_url}/v1/chat/completions"

    @property
    def models_url(self) -> str:
        return f"{self.nim_url}/v1/models"


class EndpointRegisterRequest(BaseModel):
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
    # OPENAI / frontier-API auth: stored on the endpoint, used by the
    # gateway proxy to inject `Authorization: Bearer <api_key>` on
    # outbound requests. Never serialised back in /admin/endpoints
    # responses (excluded on EndpointProfile).
    api_key: Optional[str] = None


# ---------------------------------------------------------------------------
# Telemetry
# ---------------------------------------------------------------------------


class EndpointTelemetry(BaseModel):
    """Live metrics snapshot for an endpoint."""

    endpoint_id: str
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    # Throughput
    rpm: float = 0.0           # requests per minute
    rph: float = 0.0           # requests per hour
    queue_depth: int = 0
    active_requests: int = 0
    tokens_per_second: float = 0.0
    # Latency (ms)
    p50_ttft_ms: float = 0.0
    p95_ttft_ms: float = 0.0
    p50_itl_ms: float = 0.0
    p95_itl_ms: float = 0.0
    p50_e2e_ms: float = 0.0
    p95_e2e_ms: float = 0.0
    # Reliability
    error_rate: float = 0.0    # 0.0–1.0
    saturation_score: float = 0.0  # 0.0–1.0

    @property
    def is_stale(self, threshold_s: int = 60) -> bool:
        from datetime import timezone

        now = datetime.now(timezone.utc)
        ts = self.timestamp
        if ts.tzinfo is None:
            from datetime import timezone as tz
            ts = ts.replace(tzinfo=tz.utc)
        return (now - ts).total_seconds() > threshold_s


class TelemetryUpdate(BaseModel):
    """Input for manual telemetry push (e.g. from NIM sidecar)."""

    endpoint_id: str
    rpm: float | None = None
    rph: float | None = None
    queue_depth: int | None = None
    active_requests: int | None = None
    tokens_per_second: float | None = None
    p50_ttft_ms: float | None = None
    p95_ttft_ms: float | None = None
    p50_itl_ms: float | None = None
    p95_itl_ms: float | None = None
    p50_e2e_ms: float | None = None
    p95_e2e_ms: float | None = None
    error_rate: float | None = None
    saturation_score: float | None = None


# ---------------------------------------------------------------------------
# Request classification
# ---------------------------------------------------------------------------


class RequestProfile(BaseModel):
    """Enriched request profile after classification."""

    request_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    tenant_id: str = "default"
    app_id: str = "default"
    model_requested: str
    input_tokens: int = 0
    predicted_output_tokens: int = 256  # estimated if not provided
    priority_tier: PriorityTier = PriorityTier.STANDARD
    latency_class: LatencyClass = LatencyClass.STANDARD
    optimization_target: OptimizationTarget = OptimizationTarget.AUTO
    budget_sensitivity: float = 0.5  # 0 = ignore cost, 1 = cost critical
    streaming: bool = False
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    # Derived (set by classifier)
    workload_type: WorkloadType = WorkloadType.BALANCED
    input_token_band: TokenBand = TokenBand.SMALL
    output_token_band: TokenBand = TokenBand.SMALL
    prefill_ratio: float = 1.0
    decode_ratio: float = 1.0
    burst_class: str = "normal"  # normal | burst | spike
    inferred_model_family: str = "unknown"
    inferred_model_size_b: float | None = None
    isl_tokens: int = 0
    osl_tokens: int = 0
    total_tokens: int = 0
    # Raw request (for forwarding)
    raw_body: dict[str, Any] = Field(default_factory=dict)


# ---------------------------------------------------------------------------
# Route decision
# ---------------------------------------------------------------------------


class CandidateScore(BaseModel):
    endpoint_id: str
    endpoint_name: str
    utility_score: float
    slo_score: float
    cost_score: float
    queue_score: float
    gpu_affinity_score: float
    benchmark_score: float
    model_fit_score: float
    reliability_score: float
    estimated_ttft_ms: float
    estimated_itl_ms: float
    estimated_e2e_ms: float
    estimated_cost_usd: float
    hard_rejected: bool = False
    rejection_reason: str | None = None


class RouteDecision(BaseModel):
    """Full routing decision with explainability."""

    request_id: str
    selected_endpoint_id: str | None
    selected_endpoint_name: str | None
    candidate_scores: list[CandidateScore] = Field(default_factory=list)
    hard_rejections: list[dict[str, str]] = Field(default_factory=list)
    policy_id: str = "default"
    estimated_cost_usd: float = 0.0
    predicted_ttft_ms: float = 0.0
    predicted_itl_ms: float = 0.0
    predicted_e2e_ms: float = 0.0
    # Actual (filled in post-response)
    actual_ttft_ms: float | None = None
    actual_e2e_ms: float | None = None
    outcome: RouteOutcome = RouteOutcome.SUCCESS
    fallback_used: bool = False
    fallback_count: int = 0
    decision_latency_ms: float = 0.0
    timestamp: datetime = Field(default_factory=datetime.utcnow)


# ---------------------------------------------------------------------------
# Policy
# ---------------------------------------------------------------------------


class PolicyRule(BaseModel):
    """A single routing policy rule."""

    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    name: str
    description: str = ""
    priority: int = 100  # lower = evaluated first
    conditions: dict[str, Any] = Field(default_factory=dict)
    actions: dict[str, Any] = Field(default_factory=dict)
    enabled: bool = True


class TenantPolicy(BaseModel):
    """Per-tenant routing configuration."""

    tenant_id: str
    allowed_gpu_classes: list[GPUClass] = Field(default_factory=list)
    allowed_regions: list[str] = Field(default_factory=list)
    blocked_endpoints: list[str] = Field(default_factory=list)
    max_rpm: float | None = None
    max_rph: float | None = None
    budget_usd_per_hour: float | None = None
    priority_tier_override: PriorityTier | None = None
    cost_weight_override: float | None = None  # 0–1


class RoutingPolicy(BaseModel):
    """Complete routing policy set."""

    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    name: str = "default"
    description: str = ""
    # Global weights for utility function
    slo_weight: float = 0.30
    cost_weight: float = 0.20
    queue_weight: float = 0.15
    gpu_affinity_weight: float = 0.15
    model_fit_weight: float = 0.10
    reliability_weight: float = 0.10
    # SLO targets (ms)
    slo_ttft_ms: float = 500.0
    slo_itl_ms: float = 50.0
    slo_e2e_ms: float = 5000.0
    # Hard limits
    max_queue_depth: int = 100
    min_health_score: float = 0.5
    max_error_rate: float = 0.1
    # Rules
    rules: list[PolicyRule] = Field(default_factory=list)
    tenant_policies: dict[str, TenantPolicy] = Field(default_factory=dict)
    # Presets override weights
    preset: str = "balanced"  # latency-first | balanced | cost-first


# ---------------------------------------------------------------------------
# API response wrappers
# ---------------------------------------------------------------------------


class HealthResponse(BaseModel):
    status: str = "ok"
    version: str = "0.1.0"
    endpoints_registered: int = 0
    endpoints_healthy: int = 0


class ExplainResponse(BaseModel):
    request_id: str
    decision: RouteDecision
    request_profile: RequestProfile
    policy_name: str
