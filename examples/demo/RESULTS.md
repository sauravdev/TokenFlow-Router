TokenFlow Router — live benchmark results
==========================================

Fleet under test
----------------

Two real vLLM backends on an 8×H100 80GB box:

| Lane          | Model                     | GPU | cost_class | max_ctx |
| ------------- | ------------------------- | --- | ---------- | ------- |
| vllm-fast     | Qwen/Qwen2.5-3B-Instruct  | 0   | economy    | 4096    |
| vllm-quality  | Qwen/Qwen2.5-7B-Instruct  | 1   | premium    | 16384   |

Both are registered with the TokenFlow router (see `deploy_backends.sh`).
The router runs the `production-balanced` policy from
`examples/configs/policy.yaml`.

Workload
--------

`benchmark.py` generates a mix with a fixed seed:

| Shape          | Weight | Output tokens | Notes                     |
| -------------- | -----: | ------------: | ------------------------- |
| short_chat     |    45% |            32 | one-liner Q&A             |
| reasoning      |    20% |           300 | step-by-step (premium)    |
| prefill_heavy  |    20% |            80 | summarisation of ~2k tok  |
| decode_heavy   |    15% |           500 | essays, stories           |

Each request stream is identical across arms (seed=42).

Arms
----

- **A direct** — every request → `vllm-fast` directly. No router, no
  round-robin, just one backend.
- **B round-robin** — alternate between `vllm-fast` and `vllm-quality` per
  request. This is what a dumb multi-backend load balancer does.
- **C router** — route every request through TokenFlow with headers
  indicating tenant and priority tier; router picks the endpoint.


Results — concurrency 8 (n=150)
-------------------------------

Low concurrency, both backends lightly loaded:

| Arm             |  RPS | p50 ms | p95 ms | p99 ms | SLO miss % | $ total | $/1k tok |
| --------------- | ---: | -----: | -----: | -----: | ---------: | ------: | -------: |
| A direct        | 11.14 |  287 |   2165 |   2226 |        0.0 | 0.068   | 0.0031   |
| B round-robin   |  9.38 |  302 |   3236 |   3264 |        0.0 | 0.187   | 0.0085   |
| **C router**    |  9.42 |  322 | **2371** | **2580** |      0.0 | **0.149** | **0.0068** |

Router-vs-round-robin wins at low concurrency:

- **p95 latency: -27%** (2371 ms vs 3236 ms)
- **p99 latency: -21%** (2580 ms vs 3264 ms)
- **cost: -20%** ($0.149 vs $0.187 total)

Why? Because round-robin sends half of the *short chat* and *prefill-heavy*
requests to the slower 7B model unnecessarily, stretching the tail. The
router sends those to the fast lane and reserves the 7B for
reasoning/premium workloads only.

At concurrency 8 the single-backend direct arm (A) is actually the
cheapest — for a workload this small, one vllm-fast can handle everything
and saves money. That's the *wrong* comparison though: Arm A gives up all
quality differentiation (no 7B reasoning path), and it has no headroom for
the next point.


Results — concurrency 32 (n=200)
--------------------------------

Realistic multi-tenant load. This is where routing strategy matters.

| Arm             |  RPS  | p50 ms | p95 ms | p99 ms | SLO miss % | $ total | $/1k tok |
| --------------- | ----: | -----: | -----: | -----: | ---------: | ------: | -------: |
| A direct        | 27.81 |    469 |   3038 |   3113 |        0.0 | 0.131   | 0.0041   |
| B round-robin   | 12.53 |    491 |  9438  | 11741  | **11.0**   | 0.877   | 0.0274   |
| **C router**    | **28.49** | **437** | **2530** | **2723** | 0.0 | 0.235 | 0.0074 |

Router-vs-round-robin wins at realistic concurrency:

- **throughput: +127% (2.27×)** — 28.49 vs 12.53 RPS
- **p95 latency: -73%** — 2530 ms vs 9438 ms
- **p99 latency: -77%** — 2723 ms vs 11741 ms
- **SLO miss rate: 0% vs 11%** — round-robin drops 22/200 requests outside SLO
- **cost: -73%** — $0.235 vs $0.877 for the same 200 requests


Why round-robin collapses at concurrency 32
-------------------------------------------

Round-robin sends 100 out of 200 requests to the 7B model (vllm-quality)
because that's what round-robin does. The 7B model is 2–3× slower per token
than the 3B, so half of the workload queues up against a slower backend.
Under concurrency 32 that queue never drains — tail latency explodes and
11% of requests miss their SLO.

Round-robin also pays full premium-tier cost ($8/GPU-hr) for requests that
didn't need it, so total spend is roughly 4× what the router spends for
the same workload.


Why the router wins
-------------------

Prometheus metrics straight from the live router after the runs:

    tokenflow_route_decisions_total{endpoint="vllm-fast",     priority="standard", workload="balanced"}        119
    tokenflow_route_decisions_total{endpoint="vllm-fast",     priority="standard", workload="prefill_heavy"}    57
    tokenflow_route_decisions_total{endpoint="vllm-fast",     priority="standard", workload="decode_heavy"}     36
    tokenflow_route_decisions_total{endpoint="vllm-fast",     priority="batch",    workload="balanced"}         38
    tokenflow_route_decisions_total{endpoint="vllm-fast",     priority="batch",    workload="decode_heavy"}      8
    tokenflow_route_decisions_total{endpoint="vllm-fast",     priority="batch",    workload="prefill_heavy"}     6
    tokenflow_route_decisions_total{endpoint="vllm-quality",  priority="premium",  workload="decode_heavy"}     49

- Every premium / reasoning request (49/49) went to vllm-quality.
- Every standard and batch request (264/264) went to vllm-fast.
- No request with a larger model went through the 7B unnecessarily.

Decision latency itself is tiny: per-request routing takes **0.13–0.17 ms**
(see `_tokenflow.decision_ms` in any response body). Routing overhead is
well below the variance of the upstream call.


Limitations and caveats
-----------------------

1. **Quality is not measured.** The benchmark only measures latency, cost,
   and success. It does not measure output quality, which is the reason
   you'd route reasoning requests to a bigger model in the first place.
   A fair quality comparison needs a judge model or human eval; that is
   out of scope for this harness.

2. **Two backends only.** The router's benefits grow with fleet
   heterogeneity (NIM on B200, vLLM on H200, SGLang on L40S, Dynamo, CPU
   fallbacks, etc.). With two similar backends the upside is capped at
   ~2× throughput, which is roughly what we observe.

3. **Short run.** Numbers are from 150–200-request runs. For production
   comparisons, run ≥10k requests and include bursty traffic patterns.

4. **Single box.** Both backends share the same 8-GPU machine, so network
   latency is essentially zero. Cross-region routing would add tens of ms
   per hop.

5. **The first benchmark run showed higher p95/p99 on Arm A (8876/10053
   ms with 3.3% SLO miss) due to cold-start compilation of the vLLM
   engine. The table above is from the warm run. Always warm your
   backends before comparing.**


Reproducing
-----------

On the remote box (or anywhere with SSH tunnel to ports 8080/8001/8002):

    bash examples/demo/deploy_backends.sh           # register endpoints
    bash examples/demo/01_short_chat.sh             # sanity check
    python3 examples/demo/benchmark.py \
      --router http://localhost:8080 \
      --fast   http://localhost:8001 \
      --quality http://localhost:8002 \
      --n 200 --concurrency 32

Raw results for the runs above are saved as:

- `benchmark_low_conc.json`   (n=150, concurrency=8)
- `benchmark_high_conc.json`  (n=200, concurrency=32)
