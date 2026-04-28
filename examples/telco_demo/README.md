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


What's reported
---------------

After the run, `results/benchmark.json` has both the summary tables and
the full raw per-request records:

  - `totals` — by arm
  - `by_workload` — per-(arm, workload), so you can see e.g. how
    `customer_care_voice` p95 differs between intent and TokenFlow
  - `raw` — every individual request with its endpoint_used, latency,
    cost, success, intent label (for arm A), tenant header, etc.

Charts written to `results/`:

  - `chart_headline.png` — 4-panel (success / p95 / total cost / cost-per-1k-tok)
  - `chart_per_workload.png` — 6 workloads × 2 arms across cost / p95 /
    success rate

Run for real and the actual numbers replace the placeholders here.


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
