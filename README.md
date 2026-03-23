# TokenFlow Router

> **Route every token to the right GPU lane.**

TokenFlow Router is an open-source, request-aware policy router that sits in front of multiple inference backends (NIM, vLLM, SGLang, Dynamo) and decides — per request — which model endpoint, GPU pool, and service tier should serve it.

```
┌──────────────────────────────────────────────────────────────┐
│                        Your Applications                     │
│          (any OpenAI-compatible client or SDK)               │
└──────────────────────────┬───────────────────────────────────┘
                           │  POST /v1/chat/completions
                           ▼
┌──────────────────────────────────────────────────────────────┐
│                    TokenFlow Router                          │
│                                                              │
│  ┌─────────────┐  ┌──────────────┐  ┌────────────────────┐  │
│  │  Classifier │  │Policy Engine │  │  Scoring Engine    │  │
│  │  (token     │  │  (tenant     │  │  Utility(e) =      │  │
│  │   shape,    │  │   rules,     │  │  SLO + Cost +      │  │
│  │   workload  │  │   budget,    │  │  Queue + GPU +     │  │
│  │   type)     │  │   RPM caps)  │  │  Backend Affinity  │  │
│  └─────────────┘  └──────────────┘  └────────────────────┘  │
└──────┬───────────────┬──────────────────┬────────────────────┘
       │               │                  │
       ▼               ▼                  ▼
┌─────────────┐ ┌─────────────┐   ┌──────────────────────────┐
│  NIM (B200) │ │ vLLM (H200) │   │   SGLang / Dynamo        │
│  Reasoning  │ │Decode-heavy │   │   Prefill / Disaggregated│
│  Premium    │ │  Standard   │   │   Economy / Research     │
└─────────────┘ └─────────────┘   └──────────────────────────┘
```

## What it does

TokenFlow Router is **not** an inference engine or model server. Its job is to answer one question for every incoming request:

> *Given this request shape, this current traffic, this model target, and this GPU fleet state — where should the request go right now?*

**It optimises for:**
- `TTFT` (time to first token) for interactive/prefill-heavy requests
- `ITL` (inter-token latency) for streaming/decode-heavy requests
- `E2E latency` for batch and balanced workloads
- `Cost per request` across heterogeneous GPU pools
- Business policy: per-tenant budgets, SLO tiers, GPU affinity rules

**Supported backends:**

| Backend | Strengths | Telemetry source |
|---|---|---|
| **NIM** (TensorRT-LLM) | Reasoning, premium SLO | `/metrics` (`nim:` prefix) |
| **vLLM** (PagedAttention) | Decode-heavy, high throughput | `/metrics` (`vllm:` prefix) |
| **SGLang** (RadixAttention) | Prefill-heavy, KV cache reuse | `/get_server_info` |
| **Dynamo** (disaggregated) | Both prefill + decode, KV transfer | `/metrics` (`vllm:` + `dynamo:` prefix) |

---

## Architecture

### Request lifecycle

```
1. Request arrives at POST /v1/chat/completions
2. Classifier enriches the request:
   - counts input tokens (heuristic: ~4 chars/token)
   - estimates output tokens (from max_tokens or model hint)
   - classifies workload: prefill_heavy / decode_heavy / balanced / reasoning
   - assigns token bands: tiny / small / medium / large / xlarge
   - sets latency class: interactive / standard / batch / offline
3. Policy engine applies tenant rules:
   - RPM throttling → demote to batch
   - Budget caps → maximise cost savings
   - Priority overrides
   - DSL rule matching
4. Hard constraints filter incompatible endpoints:
   - CPU endpoints: only for BATCH / OFFLINE workloads
   - RTX_LAPTOP: rejected if total tokens > 4096
5. Decision engine scores every candidate endpoint:
   Utility(e) = w_slo * SLOScore(e)
              + w_cost * CostScore(e)
              + w_queue * QueueScore(e)
              + w_gpu * GPUAffinityScore(e)   ← GPU tier × backend affinity × KV-warm bonus
              + w_model * ModelFitScore(e)
              + w_reliability * ReliabilityScore(e)
6. Best-scoring endpoint is selected
7. Request is proxied to the winning endpoint
8. TTFT and E2E latency are measured and recorded
9. Routing decision is stored for /explain API
```

### Workload classification

| Workload | Signal | Priority metric | Best backend |
|---|---|---|---|
| `prefill_heavy` | input/output > 3 | TTFT | SGLang (RadixAttention KV reuse) |
| `decode_heavy` | output/input > 3 | ITL | vLLM (PagedAttention) |
| `balanced` | moderate both | E2E + cost | Dynamo or vLLM |
| `reasoning` | model name hint | E2E reliability | NIM (TensorRT-LLM) |

### GPU tier hierarchy

| Tier | GPU class | VRAM | Typical use |
|---|---|---|---|
| 8 | **B200** | 192 GB HBM3e | Highest compute, Blackwell |
| 7 | **H200** | 141 GB HBM3e | Best memory bandwidth for decode |
| 6 | H100 | 80 GB HBM3 | General premium workloads |
| 5 | A100 | 80/40 GB HBM2e | Established premium pool |
| 4 | L40S | 48 GB GDDR6 | Standard inference |
| 3 | L40, **RTX Pro 6000** | 48 GB / 96 GB GDDR7 | 70B models on GDDR7 |
| 2 | A10G, L4, RTX 4090 | 24–24 GB | Economy inference |
| 1 | **RTX Laptop**, RTX 3090 | 8–16 GB | Edge / small models (≤4096 tokens) |
| 0 | **CPU** | — | Offline / batch tiny models only |

### Backend affinity scoring

Each backend gets a workload-type affinity multiplier applied to the GPU tier score:

| Backend | reasoning | prefill_heavy | balanced | decode_heavy |
|---|---|---|---|---|
| NIM | **1.00** | 0.90 | 0.80 | 0.70 |
| vLLM | 0.75 | 0.70 | 0.85 | **1.00** |
| SGLang | 0.70 | **1.00** | 0.85 | 0.75 |
| Dynamo | 0.85 | **0.95** | 0.90 | **0.95** |

**KV-cache warm bonus:** SGLang `cache_hit_rate` and Dynamo `kv_hit_rate` provide up to +0.15 bonus on `GPUAffinityScore` for prefill-heavy requests.

---

## Quickstart

### Docker Compose

```bash
git clone https://github.com/sauravdev/TokenFlow-Router
cd TokenFlow-Router

# Start router + mock endpoints
docker-compose up -d

# Register a NIM endpoint
curl -X POST http://localhost:8080/admin/endpoints \
  -H "Content-Type: application/json" \
  -d '{
    "name": "nim-b200-llama3-70b",
    "nim_url": "http://your-nim-host:8000",
    "model_name": "meta/llama-3.1-70b-instruct",
    "gpu_name": "B200",
    "backend_type": "nim",
    "cost_class": "premium",
    "cost_per_gpu_hour": 12.0,
    "max_context_tokens": 131072,
    "supports_reasoning": true
  }'

# Register a vLLM endpoint
curl -X POST http://localhost:8080/admin/endpoints \
  -H "Content-Type: application/json" \
  -d '{
    "name": "vllm-h200-llama3-70b",
    "nim_url": "http://your-vllm-host:8000",
    "model_name": "meta/llama-3.1-70b-instruct",
    "gpu_name": "H200",
    "backend_type": "vllm",
    "cost_class": "premium",
    "cost_per_gpu_hour": 10.0,
    "max_context_tokens": 131072
  }'

# Register an SGLang endpoint
curl -X POST http://localhost:8080/admin/endpoints \
  -H "Content-Type: application/json" \
  -d '{
    "name": "sglang-h100-llama3-8b",
    "nim_url": "http://your-sglang-host:30000",
    "model_name": "meta/llama-3.1-8b-instruct",
    "gpu_name": "H100",
    "backend_type": "sglang",
    "cost_class": "standard",
    "cost_per_gpu_hour": 6.0,
    "max_context_tokens": 32768
  }'

# Send an inference request (OpenAI-compatible)
curl -X POST http://localhost:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -H "x-tenant-id: my-team" \
  -H "x-priority-tier: standard" \
  -d '{
    "model": "meta/llama-3.1-70b-instruct",
    "messages": [{"role": "user", "content": "Hello"}],
    "max_tokens": 256
  }'
```

### Python (from source)

```bash
pip install -e ".[dev]"

# Start with a policy file
tokenflow serve --policy-file examples/configs/policy.yaml

# Or directly
uvicorn tokenflow.main:app --host 0.0.0.0 --port 8080
```

---

## CLI

```bash
# Start the server
tokenflow serve --port 8080 --policy-file examples/configs/policy.yaml

# Register an endpoint (any backend)
tokenflow register \
  --name vllm-h200 \
  --url http://vllm-host:8000 \
  --model meta/llama-3.1-8b-instruct \
  --gpu H200 \
  --backend vllm \
  --cost-class premium

# List endpoints
tokenflow list

# Switch routing preset
tokenflow policy preset latency-first   # or: balanced, cost-first

# Explain a routing decision
tokenflow explain <request-id>

# Run a simulation (no real endpoints needed)
tokenflow simulate --preset balanced --requests 200
```

---

## Routing headers

Attach these headers to influence routing per-request:

| Header | Type | Default | Description |
|---|---|---|---|
| `x-tenant-id` | string | `default` | Tenant identifier for policy lookup |
| `x-app-id` | string | `default` | Application identifier |
| `x-priority-tier` | string | `standard` | `premium` / `standard` / `batch` / `offline` |
| `x-budget-sensitivity` | float 0–1 | `0.5` | 0 = ignore cost, 1 = cost critical |

---

## Admin API

All admin endpoints are under `/admin/...`.

### Endpoints

| Method | Path | Description |
|---|---|---|
| `POST` | `/admin/endpoints` | Register an endpoint (NIM/vLLM/SGLang/Dynamo) |
| `GET` | `/admin/endpoints` | List all endpoints |
| `GET` | `/admin/endpoints/{id}` | Get one endpoint |
| `DELETE` | `/admin/endpoints/{id}` | Delete endpoint |
| `PUT` | `/admin/endpoints/{id}/enable` | Enable endpoint |
| `PUT` | `/admin/endpoints/{id}/disable` | Disable endpoint |
| `POST` | `/admin/telemetry` | Push telemetry update |
| `GET` | `/admin/telemetry/{id}` | Get current telemetry |
| `GET` | `/admin/telemetry/{id}/history` | Get telemetry history |
| `GET` | `/admin/policy` | Get active policy |
| `POST` | `/admin/policy` | Replace active policy |
| `POST` | `/admin/policy/preset` | Switch preset |
| `GET` | `/admin/routes/explain/{id}` | Explain a routing decision |
| `GET` | `/admin/routes/recent` | Last N routing decisions |
| `GET` | `/admin/metrics` | Prometheus metrics |

### Explain API

Every routing decision is stored and retrievable:

```bash
curl http://localhost:8080/admin/routes/explain/<request-id>
```

Response includes:
- Selected endpoint and why
- All candidate scores (SLO, cost, queue, GPU affinity, backend affinity, reliability)
- Hard-rejected endpoints and rejection reasons
- KV-cache warm bonus applied (SGLang / Dynamo)
- Predicted vs actual TTFT/E2E
- Policy rules applied

---

## Policy configuration

Policy is loaded from YAML at startup (or hot-swapped via API):

```yaml
name: production-balanced
preset: balanced   # latency-first | balanced | cost-first

# Utility function weights
slo_weight: 0.30
cost_weight: 0.20
queue_weight: 0.15
gpu_affinity_weight: 0.15
model_fit_weight: 0.10
reliability_weight: 0.10

# SLO targets (ms)
slo_ttft_ms: 500
slo_itl_ms: 50
slo_e2e_ms: 5000

# DSL rules
rules:
  - name: reasoning-to-nim
    priority: 5
    conditions:
      workload_type: "reasoning"
    actions:
      preferred_backend: "nim"

  - name: prefill-to-sglang
    priority: 6
    conditions:
      workload_type: "prefill_heavy"
    actions:
      preferred_backend: "sglang"

  - name: cpu-offline-only
    priority: 1
    conditions:
      gpu_class: "CPU"
    actions:
      allowed_priority_tiers: ["batch", "offline"]

  - name: throttle-burst
    priority: 20
    conditions:
      burst_class: "burst"
      priority_tier: "standard"
    actions:
      set_priority: "batch"

# Per-tenant policies
tenant_policies:
  my-enterprise-tenant:
    allowed_gpu_classes: [B200, H200, H100, A100]
    max_rpm: 500
    budget_usd_per_hour: 50.0
    priority_tier_override: premium

  edge-inference:
    allowed_gpu_classes: [RTX_LAPTOP, RTX4090, L4]
    max_rpm: 100
    budget_usd_per_hour: 2.0

  research-lab:
    allowed_gpu_classes: [B200, H200, H100, A100, RTX_PRO_6000]
    max_rpm: 200
    budget_usd_per_hour: 30.0
```

### Presets

| Preset | SLO weight | Cost weight | SLO TTFT target |
|---|---|---|---|
| `latency-first` | 0.50 | 0.05 | 300ms |
| `balanced` | 0.30 | 0.20 | 500ms |
| `cost-first` | 0.15 | 0.45 | 2000ms |

---

## Telemetry

TokenFlow Router collects per-backend metrics automatically:

### vLLM (`/metrics` — `vllm:` prefix)
- `num_requests_waiting` — queue depth
- `gpu_cache_usage_perc` — KV cache utilisation
- `p50/p95 TTFT`, `ITL`, `E2E` — from Prometheus histograms

### SGLang (`/get_server_info`)
- `cache_hit_rate` — RadixAttention KV reuse ratio (stored in `capability_flags`)
- `num_running_reqs`, `num_waiting_reqs`
- `avg_prefill_throughput`, `avg_decode_throughput`
- `token_usage`

### Dynamo (`/metrics` — `vllm:` + `dynamo:` prefix)
- `dynamo:prefill_worker_queue_depth`
- `dynamo:decode_worker_queue_depth`
- `dynamo:kv_hit_rate` — KV transfer reuse ratio
- `dynamo:kv_cache_transfer_bandwidth_bytes_per_sec`

### NIM (`/metrics` — `nim:` prefix)
- Standard latency histograms
- Queue depth and cache utilisation

All metrics use **EMA smoothing** (alpha=0.3) to prevent routing instability from momentary spikes.

---

## Prometheus metrics

Available at `GET /admin/metrics`:

```
tokenflow_route_decisions_total{outcome, endpoint_name, workload_type, priority_tier, backend_type}
tokenflow_route_decision_latency_ms (histogram)
tokenflow_upstream_request_latency_ms{endpoint_name, backend_type} (histogram)
tokenflow_upstream_ttft_ms{endpoint_name, backend_type} (histogram)
tokenflow_estimated_cost_usd_total{tenant_id, endpoint_name}
tokenflow_fallback_total{reason}
tokenflow_active_requests{endpoint_name}
tokenflow_endpoint_health{endpoint_id, endpoint_name, backend_type}
tokenflow_kv_cache_hit_rate{endpoint_name, backend_type}
```

---

## Simulator

Test routing decisions without real endpoints:

```bash
# CLI
tokenflow simulate --preset latency-first --requests 500 --model meta/llama-3.1-70b-instruct

# Python
from simulator.engine import run_simulation, standard_fleet, make_request_body
from tokenflow.models import RoutingPolicy

requests = [make_request_body("meta/llama-3.1-8b-instruct", input_tokens=500, output_tokens=256)
            for _ in range(100)]
result = await run_simulation(requests, RoutingPolicy(preset="balanced"))
print(result.slo_attainment_rate)
print(result.endpoint_distribution)
print(result.backend_distribution)
```

---

## Positioning

| Layer | Responsibility | Who owns it |
|---|---|---|
| Model serving (TensorRT-LLM) | Optimised inference runtime, API, lifecycle | **NIM** |
| Model serving (PagedAttention) | High-throughput decode, flexible model support | **vLLM** |
| Model serving (RadixAttention) | Prefill-optimised KV reuse | **SGLang** |
| Worker-level disaggregated routing | KV cache overlap, prefill/decode worker selection | **Dynamo** |
| Cross-endpoint policy routing | Request economics, SLOs, GPU lanes, business rules | **TokenFlow Router** |

---

## Roadmap

| Phase | Features |
|---|---|
| V1 (current) | Endpoint registry, telemetry, NIM + vLLM + SGLang + Dynamo adapters, backend-aware scoring, GPU tier hierarchy (B200→CPU), KV-warm bonus, policy DSL, OpenAI gateway, explain API, Prometheus metrics, CLI, simulator |
| V2 | Embeddings/rerank routing, multimodal, regional routing, Dynamo hint injection, shadow mode, canary routing |
| V3 | Learned latency estimators, traffic forecasting, adaptive reservation, RL-assisted policy tuning |

---

## Development

```bash
# Install with dev dependencies
pip install -e ".[dev]"

# Run tests
pytest tests/

# Run unit tests only
pytest tests/unit/

# Lint
ruff check tokenflow/

# Type check
mypy tokenflow/
```

---

## License

Apache 2.0 — see [LICENSE](LICENSE).

---

*Built for the Agentic AI era. Route every token to the right GPU lane.*
