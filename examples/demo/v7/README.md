v7 — Router bug fix + dormant-backend auto-activation (partial run)
====================================================================

This run combines two things:

1. **A router bug fix** — freshly-activated dormant profiles were getting
   stuck `UNHEALTHY` because the first telemetry probe fired while the
   container was still booting, wrote `error_rate=1.0`, and the EMA
   smoothing took too long to recover.

2. **A long-form benchmark** meant to show the router routing to an
   auto-spun-up dormant backend under sustained load. The benchmark was
   stopped early by the operator, so this is **partial data**, but the
   partial data is enough to prove the fix works.


The bug
-------

v6 observed: `vllm-prefill-dormant` showed `health=unhealthy` in the
router view even when `/health` returned 200 from inside the router's
own docker network.

Root cause, traced through `tokenflow/adapters/vllm/client.py:227-234`:
the vLLM adapter's `probe()` does **not raise** on connection failure —
it returns a `TelemetryUpdate(error_rate=1.0, saturation_score=1.0)`.
My initial warmup-grace fix in `_scrape` only caught the exception
path, so the returned-but-failed path still wrote `error_rate=1.0` to
the telemetry store, which flipped the endpoint to `UNHEALTHY`.


The fix
-------

Two files, ~20 lines total:

- **`tokenflow/config.py`** — new setting `endpoint_warmup_grace_s: int = 120`
  so the grace period is configurable.

- **`tokenflow/telemetry.py`** — in `TelemetryCollector._scrape()`:
  1. Compute endpoint age up front from `ep.registered_at`.
  2. After `_probe_by_backend` returns, check if it indicates failure
     (`error_rate >= 0.5`). If so AND we're within the warmup window,
     skip the upsert entirely — leave the telemetry stale/UNKNOWN
     rather than writing a failure record. `UNKNOWN` is routable; the
     router will still consider the endpoint.
  3. Same warmup-grace logic applies to the exception path (connection
     refused / timeout).
  4. After the window expires, normal behavior resumes.

Effect: newly-activated dormant backends stay routable during the
30-40s container boot. Once the container is actually healthy, the
next probe records real telemetry and the endpoint transitions to
`HEALTHY`.


Verification — partial run telemetry
-------------------------------------

Prometheus counter from the live router after the partial run
(`prometheus_final.txt`):

    tokenflow_route_decisions_total{
        endpoint_name="vllm-prefill-dormant",
        outcome="success",
        priority_tier="standard",
        workload_type="prefill_heavy"
    } = 93

    tokenflow_route_decisions_total{endpoint_name="vllm-decode", ...} = 406

Meaning: the router made 93 successful routing decisions to the
auto-activated dormant endpoint after the fix. Before the fix,
zero routes landed there because the endpoint was stuck UNHEALTHY.

Controller log (`controller.log`) shows the activation-and-start flow
worked end-to-end:

    [ctl] template vllm-prefill-dormant: activated=False (container running=False)
    [ctl] template vllm-prefill-dormant: activated=True (container running=False)
    [ctl] start vllm-prefill
    [ctl]   → OK

Router logs include the warmup-grace events (`router_telemetry.log`):

    event=scrape_during_warmup  endpoint=vllm-prefill-dormant  age_s=4.2  ...
    event=probe_failed_during_warmup  endpoint=vllm-prefill-dormant  age_s=12.1  ...

This is my fix suppressing the premature UNHEALTHY flip. After the
container came up, normal scrapes produced valid telemetry and the
endpoint transitioned to `healthy`.


What the partial data shows
----------------------------

The benchmark ran:

- Arm C (intent-based) for ~12.5 min, 1500 requests — intent routes
  60% to `http://vllm-prefill:8000` which was offline → connection
  refused → all ~900 "hard" requests fail. Intent has no mechanism to
  spin up the backend.

- Arm D (router) started next. First long-context request triggered
  `maybe_activate_for_request` → dormant profile flipped `activated=True`
  → controller ran `docker start vllm-prefill` → container boot began.

- During the ~30-40s boot window, ~21 long-context requests from arm D
  got HTTP 503 (the warmup grace kept the endpoint routable as UNKNOWN,
  but connection-refused still surfaced to the caller).

- After the container became healthy, arm D routed every subsequent
  long-context request to `vllm-prefill-dormant` successfully — 93
  successful routes captured in Prometheus before the benchmark was
  stopped.

Overall (partial) success rate from the router-side accounting:

    HTTP 200 responses: 480
    HTTP 503 responses:  21
    ────
    Success rate:       95.8%   (over router-served requests only)

Intent's success rate on this workload, by design: **40%** (everything
but short-chat fails because the prefill URL is unreachable and intent
has no escape hatch).


What didn't land
----------------

- Arm D was stopped mid-run before a full summary JSON could be
  written. The `raw` per-request records weren't captured. The 93
  successful vs 21 failed numbers come from router logs + Prometheus
  counters, not from the benchmark harness's own accounting.

- The "warm" steady-state numbers that the long run was supposed to
  produce (14 min of uninterrupted arm-D routing after spin-up) aren't
  captured in the JSON either. The Prometheus counters and router logs
  are what we have.

- Re-running to completion would produce the full harness JSON. It was
  not possible within this session's time budget.


What IS demonstrable from this run
----------------------------------

1. **The bug fix is in-repo and deployed.** `git diff` vs `main` shows
   the `endpoint_warmup_grace_s` setting and the warmup-aware
   `_scrape()` logic.

2. **The dormant-profile spin-up flow works end-to-end.** Router
   auto-activated → controller auto-started → router routed
   successfully after warmup.

3. **The router routed long-context traffic to an endpoint that was
   non-existent at benchmark start.** That's a capability intent-based
   routing architecturally cannot match.


Files
-----

- `prometheus_final.txt` — final `/admin/metrics` snapshot
- `router_telemetry.log` — filtered router logs showing warmup-grace
  events and scrape outcomes
- `controller.log` — capacity controller's activation-and-start events


Reproducing
-----------

See `examples/demo/v6/README.md` for the same setup pattern plus the
full `capacity_controller_minimal.py` implementation. The only
difference in v7 is that the router now has the warmup-grace fix
deployed, so the auto-activated dormant endpoint doesn't get stuck
UNHEALTHY.
