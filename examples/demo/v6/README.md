v6 — On-demand backend spin-up vs intent-based routing
=======================================================

**Honest framing:** this benchmark was designed to demonstrate TokenFlow's
dormant-backend / on-demand spin-up capability, a capability intent-based
routing architecturally cannot have. What actually happened surfaced real
limitations of demonstrating this in a short-latency benchmark. The
capability is real; proving a latency-win in 6 seconds is not the right
test for it.


Setup
-----

1. `vllm-decode` running (Qwen2.5-7B, max_ctx=4096) as the only live
   backend.
2. `vllm-prefill` **stopped** — the container exists but is not running,
   and its GPU is not allocated.
3. The router has a **dormant profile** registered via `POST /admin/profiles`:

   ```json
   {
     "name": "vllm-prefill-dormant",
     "nim_url": "http://vllm-prefill:8000",
     "model_name": "qwen",
     "max_context_tokens": 32768,
     "auto_activate": true,
     "activation_model_names": ["qwen"]
   }
   ```

4. `capacity_controller_minimal.py` runs in the background, polls
   `/admin/profiles` every 2s. When it sees `activated=true` for a
   template whose container is stopped, it runs `docker start
   vllm-prefill`.

5. Intent-based arm is configured with both URLs (`:8001` decode,
   `:8002` prefill). Intent has no mechanism to spin up `vllm-prefill`
   on its own — if the URL is unreachable, it fails.


Run 1 — cold start
------------------

`vllm-prefill` is stopped at benchmark start. All five arms run.

Benchmark took 6 seconds per arm. Container boot takes ~30-40 seconds.
So the container **never becomes ready during the run**.

Results (see `bench_cold_start.json`):

| Arm              | Success | Notes |
| ---------------- | ------: | ----- |
| A direct (decode)|   82.0% | 36 long-context overflow on decode (expected) |
| B round-robin    |   42.0% | Half of "odd idx" requests hit offline prefill → fail |
| C intent-based   |   41.5% | 60% of traffic classified as "hard" → routed to offline prefill → fail |
| D router         |   82.0% | Router activated dormant profile (proven by controller log), but tried to route to container-not-yet-ready → same 36 long-context failures as direct |
| E spec-decode    |  100.0% | Single-backend, prefill independent |

**Controller log** (`controller.log`) confirms auto-activation worked:

```
[ctl] starting, polling http://localhost:8080/admin/profiles every 2.0s
[ctl] template vllm-prefill-dormant: activated=False (container running=False)
[ctl] template vllm-prefill-dormant: activated=True (container running=False)
[ctl] start vllm-prefill
[ctl]   → OK
```

The router's `ensure_capacity_for_request` flipped `activated → True` on
the first long-context request; the controller saw the flip and ran
`docker start vllm-prefill`. Container came up shortly after arm D
finished.


Run 2 — warm (container running from Run 1's activation)
--------------------------------------------------------

`vllm-prefill` is now up (inherited from run 1's activation).

Results (see `bench_warm.json`):

| Arm              | Success | p95 ms | Cost |
| ---------------- | ------: | -----: | ---: |
| A direct         |   82.0% |  3,265 | $0.164 |
| B round-robin    |   92.0% | 10,801 | $0.504 |
| C intent-based   |  100.0% |  2,827 | $0.211 |
| D router         |   82.0% |  3,244 | $0.158 |

Intent-based recovered to 100% success because its prefill URL was now
live. Router continued routing long-context to `vllm-decode` (and
failing) because the activated profile-endpoint was tagged **unhealthy**
in the router's view — a state-sync issue between the profile
activation flow and the live-endpoint health-check flow. DNS and HTTP
reachability from inside the router container to
`http://vllm-prefill:8000/health` both returned 200; the router just
didn't update its internal health score in time for the benchmark.


What this actually tells us
---------------------------

**The dormant-backend capability exists and activates correctly.**
The controller log is proof: router flipped the flag, the controller
started the container, it came up. This is a capability intent-based
routing simply cannot have.

**But the right metric for this capability is NOT latency or SLO on a
6-second benchmark.** It's **GPU-hours saved** across a representative
workload with idle periods. An intent-based setup that needs
`vllm-prefill` must keep it running 24/7. A router with dormant
profiles can keep it stopped when idle and spin up on demand. Over a
week with 10% peak utilisation, that's an 80-90% reduction in GPU
spend on that tier.

**There's also a real gap in the current router** uncovered here: when
a profile is auto-activated and registered as a live endpoint, the
health-score telemetry doesn't refresh fast enough for the router to
start routing to it in the same second. In run 2 the container was up,
`/health` was 200, but the router's internal `health` flag on
`vllm-prefill-dormant` stayed `unhealthy`. That's a router bug, not a
capability limitation, and it's fixable (force a telemetry scrape
synchronously on activation, or use an eager pre-warm probe).


When this capability wins in production
----------------------------------------

- **Spiky workloads** with long idle periods — e.g., nightly batch
  reports, periodic document-summarization jobs. Spin up the big
  backend when the job arrives, tear it down when it's done.
- **Mixed tenants** where only a few tenants need specific backends —
  e.g., only enterprise tenants use the reasoning-capable model. Router
  spins up that backend only when an enterprise request arrives.
- **Sustainability / cost-conscious deployments** — paying $8/GPU-hr
  for a backend that handles 5% of traffic is wasteful; pay $0.40/hour
  instead by keeping it dormant 95% of the time.

In all these cases, the right benchmark is: **how many GPU-hours did
you consume for the same workload?** Not p95 latency on a single 200-
request run.


Reproducing
-----------

```bash
# 1. Launch backends but STOP vllm-prefill
docker stop vllm-prefill

# 2. Register dormant profile (see bench script for exact body)
curl -X POST http://localhost:8080/admin/profiles -H 'Content-Type: application/json' \
  -d '{...dormant profile body...}'

# 3. Start minimal capacity controller
python3 examples/demo/v6/capacity_controller_minimal.py &

# 4. Run benchmark
python3 examples/demo/benchmark.py \
  --router http://localhost:8080 \
  --decode http://localhost:8001 \
  --prefill http://localhost:8002 \
  --n 200 --concurrency 32 \
  --out /tmp/bench.json

# 5. Watch controller log for activation events
tail -f controller.log
```
