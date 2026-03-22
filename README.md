# TokenFlow Router

> **Route every token to the right GPU lane.**

TokenFlow Router is an open-source, request-aware policy router that sits in front of multiple NVIDIA NIM deployments and decides — per request — which model endpoint, GPU pool, and service tier should serve it.

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
│  │   type)     │  │   RPM caps)  │  │  Model + Reliability│  │
│  └─────────────┘  └──────────────┘  └────────────────────┘  │
└──────────┬───────────────┬───────────────────┬───────────────┘
           │               │                   │
           ▼               ▼                   ▼
    ┌─────────────┐ ┌─────────────┐   ┌─────────────┐
    │  NIM (H100) │ │  NIM (L40S) │   │   NIM (L4)  │
    │  Premium    │ │  Standard   │   │   Economy   │
    │  Reasoning  │ │  General    │   │   Batch     │
    └─────────────┘ └─────────────┘   └─────────────┘
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

**It is complementary to:**
- **NIM** — which handles model serving, lifecycle, and optimised inference runtimes
- **Dynamo** — which handles worker-level KV-aware routing within a deployment

TokenFlow Router sits *above* both.

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
4. Decision engine scores every candidate endpoint:
   Utility(e) = w_slo * SLOScore(e)
              + w_cost * CostScore(e)
              + w_queue * QueueScore(e)
              + w_gpu * GPUAffinityScore(e)
              + w_model * ModelFitScore(e)
              + w_reliability * ReliabilityScore(e)
5. Hard constraints filter out incompatible endpoints
6. Best-scoring endpoint is selected
7. Request is proxied to NIM endpoint
8. TTFT and E2E latency are measured and recorded
9. Routing decision is stored for /explain API
```

### Workload classification

| Workload | Signal | Priority metric | Best GPU lane |
|---|---|---|---|
| `prefill_heavy` | input/output > 3 | TTFT | Strongest prefill GPU |
| `decode_heavy` | output/input > 3 | ITL | Decode-efficient pool |
| `balanced` | moderate both | E2E + cost | Mid-range GPU |
| `reasoning` | model name hint | E2E reliability | Premium GPU |

---

## Quickstart

### Docker Compose

```bash
git clone https://github.com/sauravdev/TokenFlow-Router
cd TokenFlow-Router

# Start router + mock NIM endpoints
docker-compose up -d

# Register your NIM endpoints
curl -X POST http://localhost:8080/admin/endpoints \
  -H "Content-Type: application/json" \
  -d '{
    "name": "nim-h100-llama3-70b",
    "nim_url": "http://your-nim-host:8000",
    "model_name": "meta/llama-3.1-70b-instruct",
    "gpu_name": "H100",
    "cost_class": "premium",
    "cost_per_gpu_hour": 8.0,
    "max_context_tokens": 32768,
    "supports_reasoning": true
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

# Register an endpoint
tokenflow register \
  --name nim-h100 \
  --url http://nim-host:8000 \
  --model meta/llama-3.1-8b-instruct \
  --gpu H100 \
  --cost-class premium

# List endpoints
tokenflow list

# Switch routing preset
tokenflow policy preset latency-first   # or: balanced, cost-first

# Explain a routing decision
tokenflow explain <request-id>

# Run a simulation (no real NIM needed)
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
| `POST` | `/admin/endpoints` | Register a NIM endpoint |
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
- All candidate scores (SLO, cost, queue, GPU affinity, reliability)
- Hard-rejected endpoints and rejection reasons
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
  - name: premium-on-h100
    priority: 10
    conditions:
      priority_tier: "premium"
    actions:
      set_budget_sensitivity: 0.0

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
    allowed_gpu_classes: [H100, A100]
    max_rpm: 500
    budget_usd_per_hour: 50.0
    priority_tier_override: premium
```

### Presets

| Preset | SLO weight | Cost weight | SLO TTFT target |
|---|---|---|---|
| `latency-first` | 0.50 | 0.05 | 300ms |
| `balanced` | 0.30 | 0.20 | 500ms |
| `cost-first` | 0.15 | 0.45 | 2000ms |

---

## Telemetry

TokenFlow Router collects:
- `p50/p95 TTFT` — time to first token
- `p50/p95 ITL` — inter-token latency
- `p50/p95 E2E` — end-to-end latency
- `queue_depth` and `active_requests`
- `tokens_per_second`
- `error_rate` and `saturation_score`

These can be **pushed** (from NIM sidecars) via `POST /admin/telemetry` or **pulled** (scraped from NIM's Prometheus `/metrics` endpoint) by the background collector.

Metrics use **EMA smoothing** (configurable alpha) to prevent routing instability from momentary spikes.

---

## Prometheus metrics

Available at `GET /admin/metrics`:

```
tokenflow_route_decisions_total{outcome, endpoint_name, workload_type, priority_tier}
tokenflow_route_decision_latency_ms (histogram)
tokenflow_upstream_request_latency_ms{endpoint_name} (histogram)
tokenflow_upstream_ttft_ms{endpoint_name} (histogram)
tokenflow_estimated_cost_usd_total{tenant_id, endpoint_name}
tokenflow_fallback_total{reason}
tokenflow_active_requests{endpoint_name}
tokenflow_endpoint_health{endpoint_id, endpoint_name}
```

---

## Simulator

Test routing decisions without real NIM endpoints:

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
```

---

## Positioning

| Layer | Responsibility | Who owns it |
|---|---|---|
| Model serving | Optimised inference runtime, API, lifecycle | **NIM** |
| Worker-level KV routing | KV cache overlap, decode worker selection | **Dynamo** |
| Cross-endpoint policy routing | Request economics, SLOs, GPU lanes, business rules | **TokenFlow Router** |

---

## Roadmap

| Phase | Features |
|---|---|
| V1 (current) | Endpoint registry, telemetry, scoring engine, policy DSL, OpenAI gateway, explain API, Prometheus metrics, CLI, simulator |
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
