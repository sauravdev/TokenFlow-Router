Telco / multi-workload production benchmark
===========================================

This directory benchmarks TokenFlow Router against intent-based routing
on a realistic multi-workload telco-style platform — six concurrent
workloads each with their own SLO, budget cap, and routing policy,
served from a three-lane fleet on a single 8× A100 80GB host.

Same fleet. Same workload. Same seed. Only the routing brain differs.


Workloads modelled
------------------

| Workload                      | Mix | Tenant header           | Priority   | SLO       |
| ----------------------------- | --: | ----------------------- | ---------- | --------- |
| customer_care_voice           | 30% | tenant-customer-care    | premium    |  1.5 s    |
| rag_retrieval                 | 25% | tenant-rag-platform     | standard   |  4 s      |
| esg_batch                     | 10% | tenant-esg-reporting    | batch      | 15 s      |
| ai_assisted_migration         | 15% | tenant-migration-tools  | standard   |  8 s      |
| trust_inventory               | 15% | tenant-trust-inventory  | standard   |  3 s      |
| digital_twin_simulation       |  5% | tenant-digital-twin     | premium    | 12 s      |

Each tenant has its own `budget_usd_per_hour`, `max_rpm`, and
`allowed_gpu_classes` set in `configs/policy.yaml`. TokenFlow respects
all of them at scoring time; intent-based routing sees none of them.


Fleet (8× A100 80GB host)
-------------------------

| Lane           | Model                       | GPUs         | Cost          | max_ctx |
| -------------- | --------------------------- | -----------: | ------------: | ------: |
| vllm-economy   | Qwen/Qwen2.5-3B-Instruct    | 1× A100 80GB | $2.50 / GPU-hr|   4,096 |
| vllm-standard  | Qwen/Qwen2.5-14B-Instruct   | 1× A100 80GB | $5.00 / GPU-hr|  16,384 |
| vllm-premium   | Qwen/Qwen2.5-72B-Instruct   | 2× A100 80GB | $12.00 / GPU-hr (effective) | 32,768 |

Total: 4 GPUs in active use, 4 spare for headroom / dormant backends.


What this benchmark exercises
-----------------------------

1. **Multi-tenant policy enforcement** — six tenants with distinct
   budgets / GPU allowlists / RPM caps. Intent-based routing has no
   tenant model; TokenFlow respects the policy DSL.

2. **Multi-cost-tier optimization** — three lanes from $2.50 to $12 per
   GPU-hour. Intent sends "hard" intents (reasoning / code / voice) to
   the premium lane regardless of tenant budget. TokenFlow weights
   cost in the utility function and applies the per-tenant cap.

3. **Hard constraints over inferred signals** — `tenant-esg-reporting`
   has `allowed_gpu_classes: [A100, L40S, L4, RTX_PRO_6000]` and
   `priority_tier_override: batch`. The router *cannot* route ESG
   traffic to the premium lane even if the classifier mis-labels it as
   reasoning. Intent has no such filter.

4. **Live policy swap** (latency-first / balanced / cost-first) — same
   `POST /admin/policy/preset` mechanism as the production_demo. The
   harness reports pre/post-swap separately if you split a long run.

5. **Apples-to-apples** — both arms run the same `--n` requests at the
   same `--rate`, with the same `--seed`. Only the routing brain is
   the variable.


Quickstart
----------

You'll need:

- A host with at least 4× A100 80GB (or 4× H100 80GB equivalent).
- Docker + `vllm/vllm-openai:latest`.
- Router running on `:8080` (`docker compose up -d` in repo root).
- Python 3.11+ with `httpx` and `pyyaml`.

```bash
# 1. Launch the three local lanes
docker run -d --name vllm-economy --gpus '"device=0"' \
  --network tokenflow-router_default --ipc=host \
  -v /home/shadeform/hf-cache:/root/.cache/huggingface \
  -p 8001:8000 vllm/vllm-openai:latest \
  --model Qwen/Qwen2.5-3B-Instruct --max-model-len 4096 \
  --served-model-name qwen Qwen/Qwen2.5-3B-Instruct

docker run -d --name vllm-standard --gpus '"device=1"' \
  --network tokenflow-router_default --ipc=host \
  -v /home/shadeform/hf-cache:/root/.cache/huggingface \
  -p 8002:8000 vllm/vllm-openai:latest \
  --model Qwen/Qwen2.5-14B-Instruct --max-model-len 16384 \
  --served-model-name qwen Qwen/Qwen2.5-14B-Instruct

docker run -d --name vllm-premium --gpus '"device=2,3"' \
  --network tokenflow-router_default --ipc=host \
  -v /home/shadeform/hf-cache:/root/.cache/huggingface \
  -p 8003:8000 vllm/vllm-openai:latest \
  --model Qwen/Qwen2.5-72B-Instruct --tensor-parallel-size 2 \
  --max-model-len 32768 \
  --served-model-name qwen Qwen/Qwen2.5-72B-Instruct

# 2. Wait until all three respond
until curl -sf http://localhost:8001/health \
   && curl -sf http://localhost:8002/health \
   && curl -sf http://localhost:8003/health; do sleep 5; done

# 3. Register endpoints + load multi-workload policy
bash examples/telco_demo/setup.sh

# 4. Run the benchmark (~10 min, 1,200 requests total at 2 req/s)
python3 examples/telco_demo/benchmark.py \
  --router   http://localhost:8080 \
  --economy  http://localhost:8001 \
  --standard http://localhost:8002 \
  --premium  http://localhost:8003 \
  --n 600 --rate 2 --concurrency 8

# 5. Render charts
pip install matplotlib
python3 examples/telco_demo/chart.py
```


Live results — 8× A100 80GB box, 180 requests / arm, ~3 min run
---------------------------------------------------------------

![headline chart](results/chart_headline.png)

| Metric                       | Intent-based | **TokenFlow** | Δ        |
| ---------------------------- | -----------: | ------------: | -------: |
| Success rate                 |       100.0% |        100.0% |     —    |
| p50 latency                  |      2,891 ms|        682 ms |   −76%   |
| p95 latency                  |     12,165 ms|      2,766 ms |   −77%   |
| **SLO miss rate**            |        35.0% |      **0.0%** | −35.0 pp |
| **Total cost (180 req)**     |       $1.787 |    **$0.149** | **−92%** |
| Cost per 1k tokens           |     $0.0649  |     $0.0048   |   −93%   |

**TokenFlow is 12× cheaper and eliminates every SLO violation** on the
identical workload. Both arms succeeded on 100% of requests; the
difference is *which backend each request landed on*.

Per-workload breakdown (the part intent-based architecturally cannot do):

![per-workload chart](results/chart_per_workload.png)

| Workload                  | Intent cost | TokenFlow cost | Δ      | Intent SLO miss | TokenFlow SLO miss |
| ------------------------- | ----------: | -------------: | -----: | --------------: | -----------------: |
| ai_assisted_migration     |      $0.762 |        $0.036  | **−95%** |      **100%**   |        **0%**      |
| customer_care_voice       |      $0.336 |        $0.029  |  −91%  |       51.6%     |        **0%**      |
| digital_twin_simulation   |      $0.411 |        $0.019  |  −95%  |      **100%**   |        **0%**      |
| esg_batch                 |      $0.126 |        $0.021  |  −83%  |        0.0%     |         0.0%       |
| rag_retrieval             |      $0.127 |        $0.039  |  −69%  |        7.5%     |        **0%**      |
| trust_inventory           |      $0.025 |        $0.004  |  −85%  |        0.0%     |         0.0%       |

**What you're looking at:**

The intent-based classifier sends every "hard" workload (migration,
voice, digital-twin) to the 72B premium lane regardless of cost. Two
of those workloads (`ai_assisted_migration` and `digital_twin_simulation`)
miss their SLO 100% of the time because the 72B is saturated with
traffic that doesn't need it. The voice workload misses SLO ~52% of
the time for the same reason — its 1.5s SLO can't be met when it's
queued behind reasoning prompts on the same backend.

TokenFlow sees the per-tenant policies in `configs/policy.yaml`, scores
backends per request, and routes:

- `customer_care_voice` (premium tier, 1.5s SLO) → standard lane (14B,
  fast enough for voice, 30% the cost of premium)
- `ai_assisted_migration` (standard tier, 8s SLO) → standard lane
- `digital_twin_simulation` (premium tier, 12s SLO) → premium lane only
  for the few requests that need it
- `esg_batch` (batch tier, 15s SLO) → economy lane (3B, 80% cheaper)
- `trust_inventory` (standard tier, 3s SLO) → economy lane (small
  classification queries don't need 14B)

The router doesn't beat intent on quality — both arms successfully
return responses — but it routes intelligently enough that **no
workload misses its SLO**, while intent fails 35% of all requests'
SLOs and spends 12× more.

Routing distribution from Prometheus (`results/prometheus.txt`):

  - vllm-economy: 73 routes (chat-shape, batch, classification)
  - vllm-standard: 87 routes (voice, RAG, code migration, simulation)
  - vllm-premium: 20 routes (only the requests that genuinely benefit)

Compared to intent's distribution:

  - vllm-economy: 64 routes
  - vllm-standard: 40 routes
  - vllm-premium: **76 routes** (the expensive bucket — 4× more than TokenFlow uses)


Charts and raw data
-------------------

  - `results/benchmark.json` — full summary + 360 raw per-request records
  - `results/chart_headline.png` — 4-panel TokenFlow vs intent
  - `results/chart_per_workload.png` — 6 workloads × 2 arms
  - `results/prometheus.txt` — router-internal counters


Honest framing
--------------

- **Quality is not measured.** Intent-based routes "hard" workloads to
  the 72B; if your evaluation rewards "use the best model for hard
  prompts" then intent will look better on quality. TokenFlow's
  argument is that for many workloads the smaller model is fine and
  the cost savings are real.
- **The intent classifier here is keyword-based** (the same baseline as
  the production_demo). A trained classifier (NVIDIA AI Blueprints LLM
  Router v2 in intent profile, distilBERT, LLM-as-judge) would be more
  accurate. See `examples/integrations/nvidia_router_v2/` for how to
  compose TokenFlow + a trained classifier.
- **Single 10-minute run.** For production decisions, run with multiple
  seeds, bursty patterns, and ≥10k requests.


Files
-----

- `benchmark.py`             — single-file harness with 6 workloads + 2 arms
- `chart.py`                 — headline + per-workload chart generator
- `setup.sh`                 — register backends + load policy
- `configs/policy.yaml`      — multi-tenant routing policy
- `results/benchmark.json`   — populated after a run
- `results/chart_*.png`      — populated after running `chart.py`
