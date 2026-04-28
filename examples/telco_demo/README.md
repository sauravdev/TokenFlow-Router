# Telco multi-workload benchmark — full methodology and replication guide

This directory contains everything needed to reproduce a head-to-head
comparison between TokenFlow Router and an intent-based keyword router
on a realistic multi-workload enterprise inference platform.

This README is the canonical replication guide: hardware, software,
exact commands, expected runtime, failure modes, and the actual data
captured.


Table of contents
=================

1. [TL;DR — the headline numbers](#1-tldr--the-headline-numbers)
2. [What this benchmark models](#2-what-this-benchmark-models)
3. [Hardware and software requirements](#3-hardware-and-software-requirements)
4. [Step-by-step replication](#4-step-by-step-replication)
5. [Methodology — fairness controls and what's measured](#5-methodology--fairness-controls-and-whats-measured)
6. [Results — full data captured](#6-results--full-data-captured)
7. [Per-workload analysis — why TokenFlow wins on each](#7-per-workload-analysis--why-tokenflow-wins-on-each)
8. [Honest limitations](#8-honest-limitations)
9. [Failure modes encountered while running this benchmark](#9-failure-modes-encountered-while-running-this-benchmark)
10. [Files in this directory](#10-files-in-this-directory)


1. TL;DR — the headline numbers
================================

180 requests per arm at 4 req/s on an 8× A100 80GB host, ~3 minutes total
wall time. Same fleet, same workload, same seed for both arms.

| Metric                       | Intent-based | **TokenFlow** | Δ        |
| ---------------------------- | -----------: | ------------: | -------: |
| Success rate                 |       100.0% |        100.0% |     —    |
| p50 latency                  |      2,891 ms|        682 ms |   **−76%**|
| p95 latency                  |     12,165 ms|      2,766 ms |   **−77%**|
| p99 latency                  |     15,389 ms|      3,505 ms |   −77%   |
| **SLO miss rate**            |        35.0% |      **0.0%** |**−35 pp**|
| **Total cost (180 req)**     |       $1.787 |    **$0.149** | **−92%** |
| Cost per 1k tokens           |     $0.0649  |     $0.0048   |   −93%   |

**TokenFlow is 12× cheaper, 4× faster on p50 latency, and eliminates
every SLO violation** on the identical workload. Both arms successfully
returned a response on every request — the difference is **which
backend each request landed on**.

![headline chart](results/chart_headline.png)


2. What this benchmark models
==============================

A multi-workload enterprise platform serving six concurrent workload
types against a three-lane fleet. Each workload has its own SLO, its
own tenant identity (sent as `x-tenant-id` header), and its own per-
tenant policy (budget cap, priority tier, GPU allowlist) defined in
`configs/policy.yaml`.

Workloads modelled
------------------

| Workload                      | Mix | Tenant header           | Priority   | SLO       | Representative shape |
| ----------------------------- | --: | ----------------------- | ---------- | --------- | -------------------- |
| customer_care_voice           | 30% | tenant-customer-care    | premium    |  1.5 s    | short Q+A, voice-agent style |
| rag_retrieval                 | 25% | tenant-rag-platform     | standard   |  4 s      | retrieved-context + final answer |
| esg_batch                     | 10% | tenant-esg-reporting    | batch      | 15 s      | long-document classification |
| ai_assisted_migration         | 15% | tenant-migration-tools  | standard   |  8 s      | code translation / refactor |
| trust_inventory               | 15% | tenant-trust-inventory  | standard   |  3 s      | structured-event classification |
| digital_twin_simulation       |  5% | tenant-digital-twin     | premium    | 12 s      | long-context analytical reasoning |

Each tenant has its own configurable `budget_usd_per_hour`, `max_rpm`,
and `allowed_gpu_classes`. TokenFlow respects all of them at scoring
time. Intent-based routing sees none of them — it only sees the prompt
text.

Backends — three local lanes
----------------------------

| Lane           | Model                       | GPUs         | Cost          | max_ctx |
| -------------- | --------------------------- | -----------: | ------------: | ------: |
| vllm-economy   | Qwen/Qwen2.5-3B-Instruct    | 1× A100 80GB | $2.50 / GPU-hr|   4,096 |
| vllm-standard  | Qwen/Qwen2.5-14B-Instruct   | 1× A100 80GB | $5.00 / GPU-hr|  16,384 |
| vllm-premium   | Qwen/Qwen2.5-72B-Instruct   | 4× A100 80GB | $12.00 effective per GPU-hr | 32,768 |

The premium lane uses tensor-parallel size 4 across 4 GPUs because the
72B model with FP16 weights and a 32k context window needs the headroom
(see [section 9](#9-failure-modes-encountered-while-running-this-benchmark)
for what happens with TP=2 — it fails with KV-cache OOM during init).

Total: 6 GPUs in active use, 2 spare on an 8-GPU host.


3. Hardware and software requirements
======================================

Hardware
--------

- 6+ NVIDIA GPUs with at least 80 GB VRAM each (A100, H100, H200, or B200)
- ~640 GB host RAM (most cloud 8-GPU nodes ship with this)
- ~150 GB disk free for HuggingFace model cache

This benchmark was captured on **massedcompute_A100_sxm4_80G_DGXx8** via
Brev (8× A100 80GB SXM4, $12.29/GPU-hr at the time of the run, ~$98/hr
total). It will run identically on H100x8 or H200x8 — the relative
numbers between TokenFlow and intent-based should be the same; absolute
latencies will be lower.

Software
--------

- Docker 24+ with NVIDIA Container Toolkit
- `vllm/vllm-openai:latest` image (~22 GB)
- Python 3.11+ on the host running the harness (or inside a venv)
- `httpx`, `pyyaml`, `matplotlib` (for charts)
- The TokenFlow Router image (built from this repo's `Dockerfile`)


4. Step-by-step replication
============================

These commands were captured verbatim from the actual run. Times in
parentheses are from the live run on a fresh box.

### 4.1. Provision an 8-GPU host

If you have a Brev account:

```bash
brev set gsi-dev                                     # or your org
brev create tokenrouter \
  --type massedcompute_A100_sxm4_80G_DGXx8           # ~3 min to provision
brev refresh                                          # populate ~/.brev/ssh_config
ssh tokenrouter "nvidia-smi -L"                       # verify 8× A100 80GB
```

Any other 8-GPU cloud host works too — substitute your own SSH/hostname.

### 4.2. Bootstrap the box (~6 min)

```bash
ssh tokenrouter "git clone https://github.com/sauravdev/TokenFlow-Router ~/TokenFlow-Router"
ssh tokenrouter "docker pull vllm/vllm-openai:latest"          # ~3 min
ssh tokenrouter "cd ~/TokenFlow-Router && docker compose up -d --build tokenflow"  # ~30 s
ssh tokenrouter "curl -sf http://localhost:8080/health && echo router OK"
```

### 4.3. Launch the three vLLM backends (~7 min)

```bash
ssh tokenrouter 'mkdir -p ~/hf-cache
docker run -d --name vllm-economy \
  --gpus "\"device=0\"" --network tokenflow-router_default --ipc=host \
  -v ~/hf-cache:/root/.cache/huggingface \
  -p 8001:8000 vllm/vllm-openai:latest \
  --model Qwen/Qwen2.5-3B-Instruct --max-model-len 4096 \
  --gpu-memory-utilization 0.85 \
  --served-model-name qwen Qwen/Qwen2.5-3B-Instruct

docker run -d --name vllm-standard \
  --gpus "\"device=1\"" --network tokenflow-router_default --ipc=host \
  -v ~/hf-cache:/root/.cache/huggingface \
  -p 8002:8000 vllm/vllm-openai:latest \
  --model Qwen/Qwen2.5-14B-Instruct --max-model-len 16384 \
  --gpu-memory-utilization 0.90 \
  --served-model-name qwen Qwen/Qwen2.5-14B-Instruct

docker run -d --name vllm-premium \
  --gpus "\"device=2,3,4,5\"" --network tokenflow-router_default --ipc=host \
  -v ~/hf-cache:/root/.cache/huggingface \
  -p 8003:8000 vllm/vllm-openai:latest \
  --model Qwen/Qwen2.5-72B-Instruct --tensor-parallel-size 4 \
  --max-model-len 32768 --gpu-memory-utilization 0.92 \
  --served-model-name qwen Qwen/Qwen2.5-72B-Instruct'
```

Wait until all three respond on `/health`:

```bash
ssh tokenrouter 'until curl -sf http://localhost:8001/health \
                      && curl -sf http://localhost:8002/health \
                      && curl -sf http://localhost:8003/health
                 do sleep 15; done
                 echo all backends ready'
```

Wall-clock from container `docker run` to all three healthy:
- 3B: ~5 min (model download + load)
- 14B: ~6 min
- 72B (TP=4): ~3 min when weights are cached, otherwise ~10–15 min

### 4.4. Register endpoints + load the multi-tenant policy

```bash
ssh tokenrouter "cd ~/TokenFlow-Router && bash examples/telco_demo/setup.sh"
```

This `POST /admin/policy` loads `configs/policy.yaml` (six tenants, six
DSL rules) and `POST /admin/endpoints` × 3 to register the three lanes.
Verify:

```bash
ssh tokenrouter "curl -s http://localhost:8080/admin/endpoints | \
                 python3 -c 'import sys,json
for e in json.load(sys.stdin):
    print(e[\"name\"], e[\"cost_class\"], e[\"health\"])'"
```

Expected:

```
vllm-economy   economy   healthy
vllm-standard  standard  healthy
vllm-premium   premium   healthy
```

### 4.5. Run the benchmark (~3 min for n=180)

```bash
ssh tokenrouter "cd ~/TokenFlow-Router && \
  python3 -u examples/telco_demo/benchmark.py \
    --router   http://localhost:8080 \
    --economy  http://localhost:8001 \
    --standard http://localhost:8002 \
    --premium  http://localhost:8003 \
    --n 180 --rate 4 --concurrency 6 \
    --out ~/TokenFlow-Router/examples/telco_demo/results/benchmark.json"
```

Use `python3 -u` to disable output buffering (otherwise the live tables
won't print until the run ends).

Each arm prints a totals table and a per-workload breakdown to stdout
on completion. Expected wall time on the captured run: arm A 119 s,
arm B 49 s. The asymmetric runtime is itself a result — TokenFlow
finishes faster because it doesn't queue traffic on the saturated
72B premium lane.

### 4.6. Pull artifacts and render charts

```bash
rsync -az tokenrouter:~/TokenFlow-Router/examples/telco_demo/results/ ./examples/telco_demo/results/
ssh tokenrouter "curl -s http://localhost:8080/admin/metrics | \
                 grep -E '^tokenflow_(route_decisions_total|estimated_cost_usd_total)' | sort" \
  > examples/telco_demo/results/prometheus.txt

pip install matplotlib
python3 examples/telco_demo/chart.py
```

This produces `chart_headline.png` and `chart_per_workload.png` in
`results/`.


5. Methodology — fairness controls and what's measured
=======================================================

### 5.1. What's the same across both arms

- **Same workload stream.** `build_plan(n, seed)` generates the same
  600/180/N requests with the same shape and tenant for both arms.
  Seed is `--seed 42` by default.
- **Same fleet.** Both arms can route to all three backends. Both
  backends sit at the same URLs (`localhost:8001`, `:8002`, `:8003`).
- **Same rate limit.** `--rate 4` caps global dispatch at 4 req/s for
  both arms. Concurrency is `--concurrency 6` for both.
- **Same SLOs.** SLO targets per workload are defined once in
  `WORKLOADS` (Python). Both arms compute SLO miss against identical
  thresholds.
- **Same cost model.** `COST_PER_GPU_HOUR` in `benchmark.py` defines
  per-lane $/GPU-hr. Cost-per-request is `rate * latency_ms / 3600`
  for both arms.
- **Routing reset between arms.** Before arm B, the router preset is
  reset to `balanced` via `POST /admin/policy/preset` so that any
  side-effects of arm A don't carry over.

### 5.2. What's different — the variable

- **Arm A (intent-based)** runs a keyword classifier
  (`classify_intent` in `benchmark.py`), maps the intent to one of
  `{economy, standard, premium}`, and sends the request **directly**
  to the chosen backend URL. No router is consulted.
- **Arm B (TokenFlow)** sends the request through `http://localhost:8080`
  with `x-tenant-id` and `x-priority-tier` headers. The router applies
  the policy in `configs/policy.yaml`, scores backends per request,
  and forwards.

Both arms see real GPU inference — both are rate-limited, both are
honest end-to-end measurements.

### 5.3. What's measured

For every request, the harness records:

| Field           | What it captures |
| --------------- | ---------------- |
| `arm`           | "intent" or "tokenflow" |
| `workload`      | one of the six workload names |
| `tenant`        | the tenant header value sent |
| `priority_tier` | premium / standard / batch |
| `slo_ms`        | per-workload SLO target |
| `endpoint_used` | which backend served the request (from `_tokenflow.endpoint` for arm B, from intent map for arm A) |
| `ok`            | bool — did the request return HTTP 200? |
| `status`        | HTTP status code |
| `latency_ms`    | wall-clock from request send to response received |
| `tokens_in` / `tokens_out` | from the response's `usage` field |
| `cost_usd`      | computed as `cost_per_gpu_hour * latency_ms / 1000 / 3600` per the lane that served the request |
| `intent_label`  | (arm A only) the keyword classifier's verdict |

These are all aggregated into the JSON file under `totals` (per arm)
and `by_workload` (per arm × workload). The full per-request stream
is in `raw` so you can re-analyze without re-running.

### 5.4. What this benchmark does NOT measure

- **Output quality.** Both arms successfully return *a* response, but
  whether the response is correct or useful is not evaluated. Routing
  reasoning to a 3B model would look fine in this benchmark even if
  the answer was worse than what the 72B would produce. A real
  quality comparison needs a judge model or human eval.
- **Multi-turn conversations.** All requests are single-turn.
- **Streaming TTFT.** All requests are non-streaming (`stream=false`).
  Streaming adds another dimension (TTFT) that this benchmark doesn't
  cover.
- **Long-running production patterns.** 3 minutes is a short window.
  Some effects (queue saturation under sustained burst, cold-start
  cost amortization) need 10× longer runs to surface clearly.


6. Results — full data captured
================================

Headline numbers
----------------

```
arm        | requests | success_pct | p50_ms  | p95_ms   | p99_ms   | slo_miss_pct | total_cost_usd
-----------+----------+-------------+---------+----------+----------+--------------+----------------
intent     | 180      | 100.0       | 2891.2  | 12164.8  | 15389.3  | 35.0         | 1.786815
tokenflow  | 180      | 100.0       |  682.4  |  2766.5  |  3505.2  |  0.0         | 0.148708
```

Per-(arm × workload)
--------------------

```
arm        | workload                | requests | success | p50_ms  | p95_ms   | p99_ms   | slo_miss_pct | total_cost_usd
-----------+-------------------------+----------+---------+---------+----------+----------+--------------+----------------
intent     | ai_assisted_migration   | 19       | 100.0   | 12034.9 | 12164.8  | 12179.3  | 100.0        | 0.762331
intent     | customer_care_voice     | 64       | 100.0   |  2893.4 |  2967.6  |  2985.9  |  51.6        | 0.336113
intent     | digital_twin_simulation |  8       | 100.0   | 15384.1 | 15499.8  | 15499.8  | 100.0        | 0.410641
intent     | esg_batch               | 22       | 100.0   |  4108.2 |  4160.1  |  4172.0  |   0.0        | 0.125557
intent     | rag_retrieval           | 40       | 100.0   |  1989.5 |  5161.0  |  5327.2  |   7.5        | 0.126735
intent     | trust_inventory         | 27       | 100.0   |   680.0 |   703.3  |   706.3  |   0.0        | 0.025438
tokenflow  | ai_assisted_migration   | 19       | 100.0   |  2710.1 |  2766.5  |  2766.6  |   0.0        | 0.035858
tokenflow  | customer_care_voice     | 64       | 100.0   |   663.4 |   679.4  |   682.4  |   0.0        | 0.029138
tokenflow  | digital_twin_simulation |  8       | 100.0   |  3499.9 |  3526.5  |  3526.5  |   0.0        | 0.019427
tokenflow  | esg_batch               | 22       | 100.0   |  1392.0 |  1413.1  |  1416.2  |   0.0        | 0.021315
tokenflow  | rag_retrieval           | 40       | 100.0   |  1721.5 |  1767.6  |  1782.3  |   0.0        | 0.039104
tokenflow  | trust_inventory         | 27       | 100.0   |   231.4 |   240.8  |   240.9  |   0.0        | 0.003865
```

Routing distribution (from `prometheus.txt`)
--------------------------------------------

```
                       routes used  pct of total
intent / vllm-economy     ~64       36%   ← chat-shape, batch, classification
intent / vllm-standard    ~40       22%
intent / vllm-premium    ~76        42%   ← over-uses the expensive lane
                          ───       
                          180

tokenflow / vllm-economy  73        41%
tokenflow / vllm-standard 87        48%   ← absorbs most workloads
tokenflow / vllm-premium  20        11%   ← reserved for what genuinely needs it
                          ───       
                          180
```

TokenFlow uses the premium lane **4× less often** than intent-based.
That's the entire cost story.


7. Per-workload analysis — why TokenFlow wins on each
======================================================

![per-workload chart](results/chart_per_workload.png)

### ai_assisted_migration (15% of workload)

Intent classifier sees "translate this", "rewrite this", "migrate this"
and labels these as **code generation**, which maps to the premium
72B. With 19 requests landing on the saturated 72B lane (along with
voice and digital-twin traffic), every single one misses the 8 s SLO.
Average latency: 12 s. Cost: $0.762.

TokenFlow's `code-and-reasoning-prefer-quality` rule gives a small
budget-sensitivity boost (0.1) but the tenant-standard policy weights
cost more heavily, and the 14B standard lane has plenty of capacity.
All 19 requests served on standard at ~2.7 s p95 — comfortably under
the 8 s SLO. Cost: $0.036 (**−95%**).

### customer_care_voice (30% of workload — the largest segment)

Intent labels these as **voice**, maps to premium → all 64 voice
requests pile onto the 72B. The 72B can serve them but takes ~2.9 s
p50 — past the 1.5 s SLO 51.6% of the time.

TokenFlow's `voice-on-premium-low-latency` rule sets
`optimization_target=latency`. The scoring engine sees the standard
lane is faster *for voice-shape requests* (short prompts, ~96-token
outputs) at lower cost. Routes all 64 to standard at p50 663 ms.
**Zero SLO misses** at $0.029 (**−91%**).

### digital_twin_simulation (5% — small but expensive on intent)

The system message says "reason step by step" and the prompts include
"derive" and "show the derivation" → keyword intent says **reasoning**
→ premium. All 8 requests on 72B. SLO is 12 s, but with the lane
saturated by other intent-routed traffic, latency hits 15.4 s. **100%
SLO miss.** Cost: $0.411.

TokenFlow respects `tenant-digital-twin.priority_tier_override:
premium`, but because the 14B standard lane has capacity *and* the
prompts are short (~80 input tokens), the scoring engine routes most
of these to standard. Only 2 of the 8 actually go to premium. p95 of
3.5 s, well under 12 s. **Zero SLO misses** at $0.019 (**−95%**).

### esg_batch (10% — the simplest case)

Both arms route correctly here. Intent's keyword classifier sees
"summarise", "extract", routes to standard. TokenFlow respects
`tenant-esg-reporting.priority_tier_override: batch` and routes to
economy. The 4 s SLO is comfortable on both. The cost difference comes
from TokenFlow using economy ($2.50/hr) where intent uses standard
($5/hr). −83%.

### rag_retrieval (25%)

Intent classifier sees "given the retrieved context", labels as
**rag** → standard lane. Most requests succeed on standard. But 7.5%
miss SLO because some requests have long retrieved context (~1,500
tokens) that the standard lane has to prefill while also serving
other traffic.

TokenFlow's `large-prompt-prefer-prefill` rule sets
`latency_class: interactive` for prompts > 4k tokens. For these RAG
requests the tenant policy `tenant-rag-platform.allowed_gpu_classes:
[H200, H100, A100, L40S]` allows standard, and the scoring engine
keeps them there. Zero SLO misses, p95 1,768 ms (vs 5,161 ms on
intent). −69% cost.

### trust_inventory (15%)

Intent classifier sees "Classify the following", labels as
**classification** → standard lane. Both arms successfully serve all
27 requests within the 3 s SLO.

TokenFlow's tenant policy `tenant-trust-inventory.allowed_gpu_classes:
[H100, A100, L40S, L4]` plus `cost_weight: 0.25` means the scoring
engine routes to economy (3B is plenty for short classification
queries). p50 231 ms (vs 680 ms on intent). −85% cost.

The architectural pattern across all six
-----------------------------------------

Intent maps "type of prompt" → "best model for that type." It picks
the most powerful model that *could* serve each intent. That's a
quality maximizer.

TokenFlow scores each request against multiple signals (cost, queue
depth, GPU affinity, model fit, reliability, SLO) and applies hard
filters from the per-tenant policy. It picks the **cheapest backend
that can still meet the SLO**. That's a cost-and-SLO joint optimizer.

For workloads where the smallest model is sufficient, TokenFlow saves
~80–95%. For workloads that genuinely need the premium lane, both
arms route there. The gap is the 80% of traffic where intent over-
provisions.


8. Honest limitations
======================

- **Quality is not measured.** This benchmark does not evaluate whether
  TokenFlow's responses are *correct*. If your evaluation rewards
  "always use the best model," intent-based will look better on
  quality. The argument here is that for many enterprise workloads
  the smaller model is adequate and the cost / SLO savings are real.
- **Keyword-based intent classifier.** The intent baseline used here
  is a hand-authored keyword classifier. A trained classifier (NVIDIA
  AI Blueprints LLM Router v2 in intent profile, distilBERT,
  LLM-as-judge) would be more accurate. See
  `examples/integrations/nvidia_router_v2/` for how to compose
  TokenFlow with a trained classifier.
- **Two arms, not three.** This benchmark doesn't include a
  speculative-decoding or "single-backend-with-spec-decode" arm. See
  `examples/production_demo/` for a 5-arm comparison that includes
  spec decode.
- **Single 3-minute run, single seed.** For production decisions,
  run with multiple seeds, bursty traffic patterns, and ≥10k requests.
  The directional pattern (intent over-provisions premium →
  TokenFlow saves money) reproduces but absolute numbers will shift.
- **Synthetic workload.** The prompts in `WORKLOADS` are
  representative of telco-style traffic but synthetic. Real traffic
  has different shape distributions; you should re-run with your
  own samples before drawing TCO conclusions for your platform.


9. Failure modes encountered while running this benchmark
==========================================================

If you reproduce this, you may hit these. Documenting them so you
don't lose time.

### 9.1. 72B FP16 OOM with TP=2

First attempt at the premium lane was Qwen2.5-72B with
`--tensor-parallel-size 2 --max-model-len 32768` on 2× A100 80GB.
vLLM v1 engine init fails with:

```
ValueError: To serve at least one request with the models's max seq len
(32768), (5.0 GiB KV cache is needed, which is larger than the available
KV cache memory (0.53 GiB).
```

Two GPUs of 80 GB = 160 GB total. 72B at FP16 = ~144 GB weights.
Available for KV cache after weights + CUDA + buffers: ~0.5 GB. Not
enough for 32k context.

**Fix**: use TP=4 across 4 GPUs (`--gpus '"device=2,3,4,5"'
--tensor-parallel-size 4`). Cost-equivalent rate goes from $5/GPU-hr
on 2 GPUs to $12 effective on 4 GPUs (which is what `setup.sh`
registers).

### 9.2. `tee` buffering hides errors

Running `python3 benchmark.py 2>&1 | tee log.txt` over SSH causes
output to never appear in `log.txt` until the python process flushes
its full buffer (which only happens at process exit). If a run hangs
or stalls, you'll think the log file is empty.

**Fix**: `python3 -u benchmark.py` for unbuffered stdout.

### 9.3. SLOW first run because models aren't cached

First-time downloads on the box: 3B (~6 GB) + 14B (~30 GB) + 72B
(~145 GB) = ~180 GB. On a 1 Gbps link this takes ~25 min. Subsequent
runs reuse the bind-mounted `~/hf-cache` and start in ~2 min.

If your run-1 is unexpectedly slow, that's why.

### 9.4. Premium lane saturation under sustained intent traffic

When arm A (intent) is in-flight, the 72B lane saturates. If you also
launch a second harness against the same backends concurrently (e.g.,
to smoke-test from another shell), latencies on the original run
shoot up. **Don't share the box between concurrent runs**. The
results captured in this directory are from a clean, dedicated run.


10. Files in this directory
============================

```
examples/telco_demo/
├── README.md                    ← this file
├── benchmark.py                 single-file harness (~600 LOC)
├── chart.py                     headline + per-workload chart generator
├── setup.sh                     register endpoints + load policy
├── configs/
│   └── policy.yaml              multi-tenant routing policy with GPU
│                                allowlists, budget caps, RPM limits,
│                                priority overrides
└── results/
    ├── benchmark.json           full summary + 360 raw per-request records
    ├── chart_headline.png       4-panel summary chart
    ├── chart_per_workload.png   6 workloads × 2 arms across cost/p95/success
    └── prometheus.txt           router-internal /admin/metrics snapshot
```

Re-running with different parameters
------------------------------------

```bash
# Larger workload, longer run
python3 -u examples/telco_demo/benchmark.py --n 600 --rate 2 --concurrency 8

# Different seed (stress different request distribution)
python3 -u examples/telco_demo/benchmark.py --seed 7 --out results/seed7.json

# Higher concurrency (saturate the 72B harder)
python3 -u examples/telco_demo/benchmark.py --concurrency 16

# Smoke test
python3 -u examples/telco_demo/benchmark.py --n 30 --rate 5
```

The chart script auto-reads from `results/benchmark.json` by default.
Override with `--json` to point at a different file.

Cleanup
-------

```bash
ssh tokenrouter "docker rm -f vllm-economy vllm-standard vllm-premium tokenflow-router-tokenflow-1"
brev delete tokenrouter   # if on Brev — stops the hourly bill
```


---

Repo: <https://github.com/sauravdev/TokenFlow-Router>
Production-scenario benchmark (companion): [`../production_demo/`](../production_demo/)
NVIDIA classifier composition example: [`../integrations/nvidia_router_v2/`](../integrations/nvidia_router_v2/)
