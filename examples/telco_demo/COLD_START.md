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
                              │ wake-up time: ~25 s   │
                              └───────────────────────┘
```

The `endpoint_warmup_grace_s` setting (in `tokenflow/config.py`)
controls how long the router will wait for a freshly woken endpoint to
become healthy before considering the routing decision a failure. The
cold-start grace window is *per-endpoint* and starts when the endpoint
transitions from `dormant` to `warming`.

## Why dormant lanes matter

For real telco-shape traffic the premium / digital-twin / voice
workloads can be 5–10% of total request volume but consume 40–60% of
GPU spend if the premium lane runs continuously. Putting that lane to
sleep when there's no premium traffic and waking it on first request:

- Drops idle GPU spend from `12 USD/hr × 24h = $288/day` to roughly
  `$48/day` (assuming ~4 hours of actual premium traffic).
- Adds a one-shot ~25 s cold-start latency penalty for the *first*
  premium request after dormancy. Subsequent requests within the
  warm-window hit a hot endpoint.
- Requires the router to (a) detect dormancy, (b) trigger wake-up via
  the capacity controller, (c) hold the request until healthy.

All three are implemented in TokenFlow today
(`profile_manager.ensure_capacity_for_request`,
`endpoint_warmup_grace_s`, `BackendType.dormant` health state).

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
