# Cold-start dormant scenario — premium lane on demand

A common production pattern: keep the cheap lane (vllm-economy / Qwen 3B
on 1× A100) hot at all times, and let the premium lane (Qwen 72B / 2× A100)
be **dormant** between bursts of premium traffic. Most-of-the-time you
pay for the economy lane only; the moment a premium request lands, the
router warms the premium lane and routes the request to it without
violating the SLO.

This is one of the few capabilities TokenFlow has that intent-based
routers do not: **the router decides not just which backend, but
whether to wake one up.**

## Architecture

```
┌──────────────┐    ┌────────────────────────────────────────────┐
│ premium req  │───▶│ TokenFlow Router                           │
└──────────────┘    │                                            │
                    │  1. classifier → workload_type=reasoning   │
                    │  2. policy: priority_tier=premium          │
                    │  3. score:  vllm-premium is dormant        │
                    │             → wake it (health=warming)     │
                    │  4. wait endpoint_warmup_grace_s          │
                    │  5. route to vllm-premium when healthy     │
                    └────────────────────────────────────────────┘
                                          │
                              ┌───────────▼───────────┐
                              │ vllm-premium (Qwen72B)│
                              │ status: dormant→warm  │
                              │ wake-up time: ~96 s   │
                              │ (measured, not guess) │
                              └───────────────────────┘
```

The `endpoint_warmup_grace_s` setting (in `tokenflow/config.py`)
controls how long the router will wait for a freshly woken endpoint to
become healthy before considering the routing decision a failure. The
cold-start grace window is *per-endpoint* and starts when the endpoint
transitions from `dormant` to `warming`.

## Measured cold-start times (on this fleet)

`docker stop` → `docker start` → first `/health` 200, with model
weights warm on local NVMe (i.e., the typical production case where
the container restarts but doesn't re-download from HuggingFace):

| Lane          | Model               | GPUs           | Weights | **Start → healthy** | First request |
| ------------- | ------------------- | -------------- | ------: | ------------------: | ------------: |
| vllm-economy  | Qwen2.5-3B-Instruct | 1× A100        |    6 GB |             **50 s** |       780 ms  |
| vllm-standard | Qwen2.5-14B-Instruct| 1× A100        |   28 GB |             **46 s** |       214 ms  |
| vllm-premium  | Qwen2.5-72B-Instruct| 4× A100 (TP=4) |  144 GB |             **96 s** |       906 ms  |

What dominates each window:

- **~25–30 s baseline** — vLLM engine init, Python module imports,
  CUDA driver context creation. Pays this even for a 1 GB model.
- **+5–10 s** — model-shard load from disk → GPU memory.
- **+~30 s for TP=4** — multi-GPU init: NCCL handshake, per-GPU CUDA
  contexts, all-gather warmup, weight sharding across 4 ranks.
- **+5–10 s** — first-request kernel selection / warmup (cuBLAS,
  flash-attn, etc.).

The 14B is *slightly faster* than the 3B because the weights load is
not the bottleneck — both are dominated by the engine-init constant.
The 72B's extra ~50 s is almost entirely TP-init overhead and the 144
GB read across 4 PCIe paths.

These are **warm-disk** numbers. For a **cold-disk** start (fresh
HuggingFace download), add ~3–5 min for the 3B, ~10–15 min for the
14B, and ~25–60 min for the 72B depending on network. Prod systems
should ensure model caches are pre-pulled or use shared-cache mounts.

## Why dormant lanes matter

For real telco-shape traffic the premium / digital-twin / voice
workloads can be 5–10% of total request volume but consume 40–60% of
GPU spend if the premium lane runs continuously. Putting that lane to
sleep when there's no premium traffic and waking it on first request:

- Drops idle GPU spend from `$48/GPU-hr × 24h = $1,152/day` (4× A100
  premium) to roughly `$192/day` (assuming ~4 hours of actual premium
  traffic). That's **−83%** infrastructure spend on the premium lane.
- Adds a one-shot **~96 s** cold-start latency penalty for the *first*
  premium request after dormancy (measured above). Subsequent requests
  within the warm-window hit a hot endpoint at ~900 ms.
- Requires the router to (a) detect dormancy, (b) trigger wake-up via
  the capacity controller, (c) hold the request until healthy.

All three are implemented in TokenFlow today
(`profile_manager.ensure_capacity_for_request`,
`endpoint_warmup_grace_s`, `BackendType.dormant` health state).

The 96 s cold-start is **the wrong tool for interactive traffic** — a
voice agent with a 1.5 s SLO can't tolerate that. Cold-start lanes are
appropriate for:

- **Batch / async traffic** where the SLO is minutes (esg_batch,
  digital_twin_simulation in this demo).
- **Bursty premium traffic** that the router can route around for the
  first 100 s. The voice/RAG lanes stay hot; only the
  reasoning-specific lane sleeps.
- **Pre-warming on schedule** — set `endpoint_warmup_grace_s` to a few
  hundred seconds and the router will hold requests gracefully, but in
  prod you typically combine this with a scheduled `docker start`
  ahead of expected traffic windows.

For interactive workloads with sub-second SLOs, keep the lane hot or
use a **warm-pool** strategy: 1 economy replica always on, +1 hot
replica preemptively scaled at peak hours. The `dormant` state is for
cost-vs-latency at coarse time granularity, not a substitute for
proper capacity planning.

## Replicating the scenario

1. Mark `vllm-premium` as dormant via the admin API:

   ```bash
   curl -X PATCH http://localhost:8080/admin/endpoints/vllm-premium \
     -H 'Content-Type: application/json' \
     -d '{"health": "dormant"}'
   ```

2. Send a premium-tier reasoning request and time it:

   ```bash
   time curl -X POST http://localhost:8080/v1/chat/completions \
     -H 'x-tenant-id: tenant-digital-twin' \
     -H 'x-priority-tier: premium' \
     -d '{"model":"qwen","messages":[{"role":"user","content":"Reason step by step about ..."}]}'
   ```

3. Observe the router holding the request, then routing to a now-warm
   `vllm-premium` once the endpoint reports healthy. The first-request
   latency includes the wake-up time; the second request lands on a hot
   endpoint and hits the normal premium-lane latency.

## Caveats

The wake-up code path assumes the dormant backend is actually
restartable from a stopped-but-not-deleted state — typical for
container orchestrators (`docker start <name>`) or Kubernetes pods
scaled to zero with HPA + KEDA. The router does not pull container
images on demand; it expects the orchestrator's `start` action to
return within `endpoint_warmup_grace_s`.

For provider-managed lanes (`backend_type=openai`), there is no
dormant state — the API is always-on, billed per-token, and TokenFlow
treats it as a plain healthy endpoint.
