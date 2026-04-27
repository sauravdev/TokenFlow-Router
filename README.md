# TokenFlow Router

> **Route every token to the right GPU lane.**

TokenFlow Router is an open-source, request-aware policy router that sits in front of multiple inference backends (NIM, vLLM, SGLang, Dynamo, Ollama, and frontier APIs like OpenAI/Anthropic/OpenRouter) and decides — per request — which model endpoint, GPU pool, and service tier should serve it.

## TL;DR

If you run more than one inference stack or GPU tier, TokenFlow Router gives you a single OpenAI-compatible endpoint that can:

- inspect the request shape (**model**, **ISL**, **OSL**, streaming, workload type)
- inspect the fleet shape (**backend**, **GPU class**, **queue**, **health**, **cost**)
- apply business policy (**tenant**, **priority**, **budget**, **SLO**)
- choose the best backend for either **latency** or **throughput**

In plain English: this is the layer that decides whether a request should go to **NIM on H100**, **vLLM on H200**, **SGLang on L40S**, **Dynamo**, or **Ollama** — and explains why.

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

> *Given this request shape, this current traffic, this model target, the request's ISL/OSL, and this GPU fleet state — where should the request go right now?*

**It optimises for:**
- `TTFT` (time to first token) for interactive/prefill-heavy requests
- `ITL` (inter-token latency) for streaming/decode-heavy requests
- `E2E latency` for batch and balanced workloads
- `Cost per request` across heterogeneous GPU pools
- Business policy: per-tenant budgets, SLO tiers, GPU affinity rules

## When to use TokenFlow Router

Use it when you have any of these problems:

- the same model exists on multiple backends and you want the router to pick the best one dynamically
- some traffic cares about **lowest latency**, while other traffic cares about **highest throughput**
- you have a mixed fleet (B200/H200/H100/L40S/L4/consumer GPUs/CPU) and want requests routed intelligently
- you need per-tenant budgets, priorities, or GPU access rules
- you want a single OpenAI-compatible endpoint instead of exposing multiple serving stacks directly

If you only run one backend on one machine, you probably do **not** need this project.

**Supported backends:**

| Backend | Strengths | Telemetry source |
|---|---|---|
| **NIM** (TensorRT-LLM) | Reasoning, premium SLO | `/metrics` (`nim:` prefix) |
| **vLLM** (PagedAttention) | Decode-heavy, high throughput | `/metrics` (`vllm:` prefix) |
| **SGLang** (RadixAttention) | Prefill-heavy, KV cache reuse | `/get_server_info` |
| **Dynamo** (disaggregated) | Both prefill + decode, KV transfer | `/metrics` (`vllm:` + `dynamo:` prefix) |
| **Ollama** | Edge/local deployments, low operational overhead | health + lightweight capability probing |

This guidance is centralized in `tokenflow/benchmarks.py` so the router, dormant-template activator, tests, and docs all use the same backend-strength matrix instead of drifting separately.

---

## Architecture

### Core concepts in plain English

- **HW** = the hardware lane behind an endpoint: GPU class, GPU count, backend type
- **LLM model** = the requested model name, plus inferred family/size when possible
- **ISL** = input sequence length (prompt size in tokens)
- **OSL** = output sequence length (expected generation size in tokens)
- **Latency mode** = optimize for snappy first token / interactive feel
- **Throughput mode** = optimize for sustained output and fleet efficiency

A rough intuition:
- long prompt, short answer → **prefill-heavy**
- short prompt, long answer → **decode-heavy**
- reasoning model → prefer more reliable premium lanes

### Example routing decisions

| Request shape | Likely best fit | Why |
|---|---|---|
| 70B reasoning model, medium ISL, streaming, `latency` | NIM on H100/H200/B200 | better premium interactive lane, strong headroom |
| 8B or 70B, short ISL, very long OSL, `throughput` | vLLM or Dynamo on memory-rich GPUs | strong decode throughput and batching |
| long-context RAG prompt, shorter answer | SGLang | strong prefill / prefix reuse behavior |
| small local model, cheap offline job | Ollama / economy GPU / CPU lane | lower ops overhead, cheaper lane |

### Request lifecycle

```
1. Request arrives at POST /v1/chat/completions
2. Classifier enriches the request:
   - counts input tokens and captures **ISL** (input sequence length)
   - estimates output tokens and captures **OSL** (output sequence length)
   - infers the requested **LLM family** and rough **model size** from the model name
   - classifies workload: prefill_heavy / decode_heavy / balanced / reasoning
   - assigns token bands: tiny / small / medium / large / xlarge
   - sets latency class: interactive / standard / batch / offline
   - resolves user routing intent: `latency` or `throughput` (via `routing.optimize_for` or `X-Optimization-Target`)
3. Router evaluates endpoint **hardware fit** before scoring:
   - checks GPU class / VRAM headroom against inferred model size + ISL/OSL working set
   - rejects lanes that are unlikely to fit the requested model/context efficiently
4. Policy engine applies tenant rules:
   - RPM throttling → demote to batch
   - Budget caps → maximise cost savings
   - Priority overrides
   - DSL rule matching
5. Hard constraints filter incompatible endpoints:
   - CPU endpoints: only for BATCH / OFFLINE workloads
   - RTX_LAPTOP: rejected if total tokens > 4096
6. Decision engine scores every candidate endpoint:
   Utility(e) = w_slo * SLOScore(e)
              + w_cost * CostScore(e)
              + w_queue * QueueScore(e)
              + w_gpu * (AffinityScore(e) + BenchmarkScore(e)) / 2
              + w_model * ModelFitScore(e)
              + w_reliability * ReliabilityScore(e)
   - `latency` intent favours lower TTFT / lower cold-start risk
   - `throughput` intent favours higher decode/prefill throughput, concurrency, and memory efficiency
7. Best-scoring endpoint is selected
8. Request is proxied to the winning endpoint
9. TTFT and E2E latency are measured and recorded
10. Routing decision is stored for /explain API
```

### Workload classification

| Workload | Signal | Priority metric | Best backend |
|---|---|---|---|
| `prefill_heavy` | input/output > 3 | TTFT | SGLang (RadixAttention KV reuse) |
| `decode_heavy` | output/input > 3 | ITL | vLLM (PagedAttention) |
| `balanced` | moderate both | E2E + cost | Dynamo or vLLM |
| `reasoning` | model name hint | E2E reliability | NIM (TensorRT-LLM) |

### Request understanding before backend selection

Before choosing a backend, the router builds a request profile from four core signals:

- **HW**: the registered endpoint hardware (`gpu_name`, `gpu_count`, backend type)
- **LLM model**: exact model name plus inferred family/size (for example `llama`, `qwen`, `70b`, `8b`)
- **ISL**: input sequence length in tokens
- **OSL**: output sequence length in tokens

Those signals are then combined with the user's routing intent:

- **optimize for latency** → favor faster TTFT, stronger hardware headroom, and low cold-start risk
- **optimize for throughput** → favor higher sustained decode/prefill throughput, concurrency, and memory-efficient backends

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

### One-click install (recommended)

For the fastest path from clone to running router — no YAML editing —
use the bundled installer + interactive wizard:

```bash
git clone https://github.com/sauravdev/TokenFlow-Router
cd TokenFlow-Router
./scripts/install.sh
```

The installer detects your environment (Docker, Kubernetes, GPUs,
cloud CLIs), then `tokenflow init` walks you through:

  1. picking a deployment target (Docker / Kubernetes / bare-metal)
  2. picking a routing policy preset (latency-first / balanced / cost-first)
  3. registering one or more backends (NIM / vLLM / SGLang / Dynamo / Ollama / OpenAI-compatible frontier APIs)
     via interactive prompts
  4. (optional) enabling dormant-backend auto-spin-up
  5. (optional) wiring a spot / preemptible adapter for cloud capacity

It writes `examples/configs/policy.yaml`, `.env`, and a runnable
`.tokenflow/register_endpoints.sh`. State is persisted in
`.tokenflow/onboarding.json` so you can resume:

```bash
./scripts/install.sh --resume
```

If you skipped `--apply`, bring everything up with:

```bash
docker compose up -d
.tokenflow/register_endpoints.sh
curl http://localhost:8080/health
```

### Kubernetes (EKS, AKS, GKE, k3s)

```bash
helm install tokenflow deploy/k8s/helm/tokenflow-router \
  --namespace tokenflow --create-namespace \
  --set-file policy.content=examples/configs/policy.yaml
```

The chart includes a `Deployment`, `Service`, `ConfigMap` for the
policy, optional `HorizontalPodAutoscaler` and `ServiceMonitor`, and a
ServiceAccount with annotation slots for EKS IRSA / AKS workload
identity. See `deploy/k8s/helm/tokenflow-router/values.yaml` for tuning
knobs and cluster-specific examples.

### Spot / preemptible scaling

`examples/autoscale/spot_adapter.py` plugs into the existing capacity
controller and supports AWS Spot, Azure Spot VMs, and GCP Spot VMs.
Enable in the wizard, or set `capacityController.enabled=true` in the
Helm values.

### Docker Compose (manual)

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

# Register a Dynamo endpoint
curl -X POST http://localhost:8080/admin/endpoints \
  -H "Content-Type: application/json" \
  -d '{
    "name": "dynamo-h100-llama3-70b",
    "nim_url": "http://your-dynamo-router:8000",
    "model_name": "meta/llama-3.1-70b-instruct",
    "gpu_name": "H100",
    "backend_type": "dynamo",
    "cost_class": "premium",
    "cost_per_gpu_hour": 9.0,
    "max_context_tokens": 131072,
    "capability_flags": {
      "disaggregated": true,
      "kv_transfer": true
    }
  }'

# Send an inference request (OpenAI-compatible)
curl -X POST http://localhost:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -H "x-tenant-id: my-team" \
  -H "x-priority-tier: standard" \
  -H "x-optimization-target: latency" \
  -d '{
    "model": "meta/llama-3.1-70b-instruct",
    "messages": [{"role": "user", "content": "Hello"}],
    "max_tokens": 256,
    "routing": {"optimize_for": "latency"}
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

### Live demo with two real vLLM backends (`examples/production_demo/`)

For an end-to-end walkthrough against *real* model servers — not mock
endpoints — see `examples/production_demo/`. It contains a single
production-scenario benchmark that puts TokenFlow head-to-head against
intent-based routing on a multi-tenant SaaS LLM platform:

- 3 tenants (free / standard / enterprise) with different budget caps
  and GPU allowlists
- 5 workload shapes (chat, reasoning, long-context, summarization,
  decode-heavy) — sampled deterministically from a fixed seed
- 2 real vLLM backends (Qwen2.5-3B economy at $2.50/GPU-hr,
  Qwen2.5-7B premium at $8/GPU-hr)
- Live policy swap mid-run (balanced → cost-first via `/admin/policy/preset`)

Five things the benchmark exercises in a single ~10-minute run:

1. **Multi-cost-tier optimization** — utility-weighted cost score
2. **Hard constraints over inferred signals** — long-context (>4k
   tokens) is structurally rejected from the small-ctx backend
3. **Per-tenant policy enforcement** — budget caps, GPU allowlists, RPM limits
4. **Live policy swap** — preset change without restart, traffic mix
   shifts immediately
5. **Apples-to-apples** — both arms see the same fleet, same workload,
   same seed, same duration; only the routing brain differs

Minimal reproduction once you have two vLLM containers running (3B on
port 8001, 7B on 8002) and the router on 8080:

```bash
bash examples/production_demo/setup.sh         # register backends + load policy
python3 examples/production_demo/benchmark.py \
  --router  http://localhost:8080 \
  --fast    http://localhost:8001 \
  --quality http://localhost:8002 \
  --n 600 --rate 2 --concurrency 8
python3 examples/production_demo/chart.py      # render headline + per-tenant charts
```

See `examples/production_demo/README.md` for the full methodology, the
multi-tenant policy in `configs/policy.yaml`, and the JSON output schema.

---

## CLI

```bash
# Interactive setup wizard (first run) — no YAML required
tokenflow init                    # walks you through environment, backends, policy
tokenflow init --apply            # also brings the router up via docker compose
tokenflow init --resume           # resume a previous interactive session

# Start the server
tokenflow serve --port 8080 --policy-file examples/configs/policy.yaml

# Register an endpoint (any backend) — or rely on the wizard's
# auto-generated .tokenflow/register_endpoints.sh
tokenflow register \
  --name vllm-h200 \
  --url http://vllm-host:8000 \
  --model meta/llama-3.1-8b-instruct \
  --gpu H200 \
  --backend vllm \
  --cost-class premium

# List endpoints
tokenflow list

# Switch routing preset (live, no restart)
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
| `x-optimization-target` | string | `auto` | `latency` / `throughput` / `auto` |
| `x-budget-sensitivity` | float 0–1 | `0.5` | 0 = ignore cost, 1 = cost critical |

---

## Response metadata

Successful non-streaming responses include a `_tokenflow` block with:
- selected backend and GPU
- resolved optimization target
- request shape summary (`llm_model`, `model_family`, `model_size_b`, `isl_tokens`, `osl_tokens`, `total_tokens`)
- `capacity_plan`, including:
  - `active_backend`
  - `turn_down_candidates`
- a short end-user benefit explanation

The gateway also exposes lightweight response headers for external callers:
- `X-TokenFlow-Active-Backend`
- `X-TokenFlow-Active-Endpoint`
- `X-TokenFlow-Turn-Down-Candidates`

---

## Dormant backend templates and single-owner model placement

If you do **not** want to keep every backend live all the time, use `/admin/profiles` instead of `/admin/endpoints`.

Profile templates let you register **available but dormant** backends. TokenFlow can then:
- activate the best matching backend template when a request arrives
- prefer a single backend for a model/workload instead of activating everything
- deactivate idle templates after a TTL so duplicate model copies do not stay live unnecessarily
- keep a built-in **hysteresis buffer** between activation and deactivation so real-time traffic is less likely to flap endpoints up/down

### Example: keep only the needed backend active

```bash
curl -X POST http://localhost:8080/admin/profiles \
  -H "Content-Type: application/json" \
  -d '{
    "name": "dynamo-h100-llama3-70b-template",
    "nim_url": "http://your-dynamo-router:8000",
    "backend_type": "dynamo",
    "model_name": "meta/llama-3.1-70b-instruct",
    "model_family": "llama",
    "gpu_name": "H100",
    "cost_class": "premium",
    "activation_model_names": ["meta/llama-3.1-70b-instruct", "llama"],
    "workload_affinity": ["balanced", "decode_heavy"],
    "exclusive_model_residency": true,
    "idle_ttl_seconds": 900,
    "min_live_seconds": 180,
    "deactivation_buffer_seconds": 120
  }'
```

With `exclusive_model_residency=true`, activating this template can deactivate sibling templates for the same model family so TokenFlow does not keep multiple live copies by default.

To avoid deactivation impacting real-time traffic, templates now support a built-in buffer:
- `idle_ttl_seconds` — how long the endpoint must be idle before deactivation is even considered
- `min_live_seconds` — minimum dwell time after activation before shutdown is allowed
- `deactivation_buffer_seconds` — extra quiet period added on top of idle TTL before shutdown

Deactivation also refuses to proceed if telemetry still shows active requests or queued work.

### Sample external controller

A sample controller is included at:
- `examples/autoscale/tokenflow_capacity_controller.py`
- `examples/autoscale/endpoint_actions.example.json`

It polls `/admin/profiles`, treats each template's `activated` flag as desired state, and maps that into engine-specific start/stop hooks.

The sample now includes:
- cooldown windows to avoid flapping
- retry / backoff for failed starts
- stop protection via multiple inactive polls before shutdown
- adapters for:
  - raw commands
  - `systemd`
  - Docker
  - Kubernetes (`kubectl scale`)
  - **AWS Spot** / **Azure Spot VMs** / **GCP Spot VMs**
    (see `examples/autoscale/spot_adapter.py`)

Example:

```bash
python examples/autoscale/tokenflow_capacity_controller.py \
  --tokenflow-url http://localhost:8080 \
  --config examples/autoscale/endpoint_actions.example.json \
  --dry-run
```

This sample also lines up with the response headers exposed to external callers:
- `X-TokenFlow-Active-Backend`
- `X-TokenFlow-Active-Endpoint`
- `X-TokenFlow-Turn-Down-Candidates`

### Endpoint warmup grace

When the router auto-activates a dormant profile, the new endpoint's
container can take 30–60 seconds to boot. During that window, telemetry
probes will fail (connection refused). To prevent the endpoint from
being prematurely flipped to `UNHEALTHY`, the telemetry collector
applies a **warmup grace period** (default 120s, configurable via
`TOKENFLOW_ENDPOINT_WARMUP_GRACE_S`):

- A failed probe within `endpoint_warmup_grace_s` of `registered_at`
  is logged but does **not** mark the endpoint unhealthy.
- The endpoint stays `UNKNOWN` (still routable) until the container
  becomes reachable.
- After the grace window expires, the existing `UNHEALTHY` behavior
  resumes for genuine outages.

The dormant-activation → spin-up → route flow is exercised by the
production scenario in `examples/production_demo/`.

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
| `POST` | `/admin/profiles` | Register a dormant backend template |
| `GET` | `/admin/profiles` | List backend templates |
| `GET` | `/admin/profiles/{id}` | Get one backend template |
| `POST` | `/admin/profiles/{id}/activate` | Manually activate a template |
| `POST` | `/admin/profiles/{id}/deactivate` | Deactivate a live template |
| `POST` | `/admin/profiles/reconcile` | Run idle deactivation immediately |
| `DELETE` | `/admin/profiles/{id}` | Delete a backend template |
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

## Benchmarks

`examples/production_demo/` contains a single production-scenario
benchmark that puts TokenFlow head-to-head against intent-based routing
on a multi-tenant SaaS LLM workload. Same fleet, same workload, same
seed — only the routing brain differs. See
`examples/production_demo/README.md` for full methodology.

**Live result on 2× H200, 600 requests / arm, ~10 min:**

| Metric                       | Intent-based | **TokenFlow** | Δ        |
| ---------------------------- | -----------: | ------------: | -------: |
| Success rate                 |       100.0% |         93.2% | −6.8 pp  |
| p50 latency                  |        423 ms|        275 ms | −35%     |
| p95 latency                  |      1,827 ms|      1,175 ms | −36%     |
| **Total cost (600 req)**     |      $0.816  |   **$0.210**  | **−74%** |
| Cost per 1k tokens           |     $0.0112  |     $0.0031   | −72%     |

Per-tenant cost reduction: free −82%, standard −67%, enterprise −73%.
The 6.8 pp success drop is concentrated in free-tier long-context — a
deliberate cost-first batch-policy refusal (the router declines to
spend premium GPU dollars on a free-tier request that its 4k-context
economy lane can't fit) — not a bug. Intent-based has no concept of
tenants and silently spends 5× more keeping all requests succeeding.

The benchmark exercises six architectural advantages of TokenFlow over
intent-based routing in a single ~10-minute run:

1. **Multi-signal blended utility** — TokenFlow combines context-fit, queue depth, cost, GPU affinity, model fit, reliability into a weighted score. Intent-based routing reads only the prompt.
2. **Hard constraints over inferred signals** — context-fit, tenant allowlists, health thresholds are binary filters that don't depend on workload-type inference. Misclassified requests can still land on a viable backend.
3. **Per-tenant policy enforcement** — budget caps, RPM limits, GPU allowlists, priority overrides — read from headers, applied at scoring time. Intent classifiers don't see tenants.
4. **Multi-cost-tier optimization** — economy backend at $2.50/GPU-hr vs premium at $8/GPU-hr; weighted cost score routes only what *needs* premium to premium.
5. **Live policy swap** — swap latency-first / balanced / cost-first at runtime, no restart. The benchmark POSTs `/admin/policy/preset` mid-run and reports pre/post-swap traffic shift.
6. **Full decision trace** — `GET /admin/routes/explain/{request_id}` returns candidate scores, hard rejections, and which policy rules fired.

Per-request routing overhead: **0.13–0.17 ms** (`_tokenflow.decision_ms`
in any response).

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

## Implementation Status

This section reflects the current maturity of the codebase so contributors and users can set accurate expectations.

### Implemented and production-ready

| Area | Details |
|---|---|
| Endpoint registry | Register, list, enable/disable, delete endpoints via REST API |
| Multi-backend support | NIM, vLLM, SGLang, Dynamo, Ollama, **OpenAI / frontier APIs** — with per-backend telemetry adapters |
| Frontier API routing | OpenAI-compatible adapter (works with OpenAI, Anthropic, OpenRouter, xAI, Together, Fireworks). Per-endpoint API key stored on registration, injected as `Authorization: Bearer` on outbound forwards, never serialised back via `/admin/endpoints` |
| Scoring engine | 6-component weighted utility function (SLO, cost, queue, GPU affinity, model fit, reliability) |
| GPU tier hierarchy | B200 → H200 → H100 → A100 → L40S → ... → CPU |
| Backend affinity | Per-backend multipliers per workload type (prefill-heavy, decode-heavy, balanced, reasoning) |
| KV-cache warm bonus | SGLang cache_hit_rate + Dynamo kv_hit_rate used to reward warm prefix caches |
| Hard constraint filtering | CPU-only-for-batch, RTX_LAPTOP token budget, queue depth ceiling, error rate ceiling |
| Policy DSL | Rules with conditions + actions, tenant policies, RPM/budget throttling |
| 3 routing presets | `latency-first`, `balanced`, `cost-first` — hot-swappable via API |
| Fallback chain | Automatic re-routing on upstream failure, excludes failed endpoints |
| OpenAI-compatible gateway | POST /v1/chat/completions — streaming + non-streaming, SSE |
| Token counting | tiktoken (cl100k_base) when available, 4-chars/token heuristic fallback |
| Telemetry collection | Background EMA-smoothed polling every 10s, per-backend adapters |
| Stale telemetry handling | Falls back to GPU-tier heuristics after 60s stale threshold |
| Observability | Prometheus metrics, structured JSON logs, per-request explain API |
| Workload report | GET /admin/report — tokens/requests/cost per backend + workload breakdown |
| Admin API auth | Optional `TOKENFLOW_ADMIN_API_KEY` enforced on all `/admin/*` routes |
| CORS hardening | Configurable `TOKENFLOW_ALLOWED_ORIGINS` (default `*` for local dev) |
| Dynamic backend profiles | Dormant backend templates with `workload_affinity`, synchronous first-use activation, single-owner model residency, and idle deactivation |
| Endpoint warmup grace | Newly-activated dormant endpoints stay routable as `UNKNOWN` during container boot; configurable `TOKENFLOW_ENDPOINT_WARMUP_GRACE_S` (default 120s) |
| Interactive onboarding | `tokenflow init` Rich-based wizard generates policy.yaml, .env, and register_endpoints.sh — no hand-edited YAML required |
| One-click installer | `scripts/install.sh` detects environment, sets up venv, runs the wizard, optionally brings up Docker Compose |
| Helm chart | `deploy/k8s/helm/tokenflow-router/` with Deployment, Service, ConfigMap, optional HPA + ServiceMonitor — production-shaped for EKS / AKS / GKE / k3s |
| Spot capacity adapters | `examples/autoscale/spot_adapter.py` — AWS Spot, Azure Spot VMs, GCP Spot VMs adapters with start/stop/preemption-check, plug into the existing capacity controller |

### Partially implemented / known gaps

| Area | Status |
|---|---|
| Token estimation accuracy | tiktoken cl100k_base is ~1–2% accurate for most models; not byte-perfect for every model family |
| Registry persistence | In-memory only — endpoints lost on restart, no disk/DB persistence |
| Streaming TTFT measurement | Measures TTFT from first SSE chunk, which may include proxy overhead |
| Policy conflict resolution | Tenant policy + DSL rules can produce conflicting overrides; last-rule-wins semantics |
| Spot adapter validation | AWS / Azure / GCP adapters are interface + reference implementations; vendor CLI flags should be validated against current docs before production use (capacity-pool exhaustion, IMDSv2, hibernation modes have edge cases) |
| Onboarding non-interactive | `tokenflow init` is interactive only — no `--non-interactive` flag yet for unattended CI/CD scenarios |

### Planned / roadmap

| Phase | Features |
|---|---|
| V2 | Embeddings/rerank routing, multimodal support, regional routing, Dynamo hint injection, shadow mode, canary routing, registry persistence |
| V3 | Learned latency estimators, traffic forecasting, adaptive reservation, RL-assisted policy tuning |

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
