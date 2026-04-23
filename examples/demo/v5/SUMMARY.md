v5 — Noise stress test (intent vs TokenFlow, inferred-signal robustness)
========================================================================

Both strategies use inferred signals:

- **Intent-based** uses a keyword (or ML) classifier over the prompt text.
- **TokenFlow** uses `workload_type`, which is derived from
  `input_tokens` (measured) and `predicted_output_tokens` (taken from
  `max_tokens` in the request — **can be wrong** if the client sends a
  misleading ceiling).

v5 applies controlled noise to both signals and compares how each
strategy degrades.


Noise knobs
-----------

```bash
python3 examples/demo/benchmark.py \
  --intent-noise   0.15 \   # randomise 15% of classifier outputs
  --workload-noise 0.15 \   # randomise 15% of max_tokens values
  ...
```

- **intent-noise** only affects arm C (intent-based). The classifier's
  output is replaced with a random label 15% of the time. Simulates an
  ML classifier with ~15% mis-classification rate (distilBERT / LLM-as-judge
  in production typically lands in this range).
- **workload-noise** only affects arm D (TokenFlow). A fraction of requests
  are sent with a `max_tokens` value inconsistent with the actual expected
  output, so TokenFlow's `predicted_output_tokens → workload_type` inference
  misfires.

Both noises are deterministic (seeded) so reruns produce identical sequences.


Results — c=32, n=200, Qwen2.5-7B on both backends
---------------------------------------------------

Focus on arms C (intent) and D (router). Arms A/B/E unaffected by noise.

| Config                            | Intent success | Intent p95 | Intent $ | Router success | Router p95 | Router $ |
| --------------------------------- | -------------: | ---------: | -------: | -------------: | ---------: | -------: |
| clean (0, 0)                      |         100.0% |   2,798 ms |   0.208  |         100.0% |   2,212 ms |   0.194  |
| intent-noise only (0.15, 0)       |          99.0% |   2,739 ms |   0.204  |         100.0% |   2,227 ms |   0.194  |
| workload-noise only (0, 0.15)     |         100.0% |   2,830 ms |   0.198  |         100.0% |   2,963 ms |   0.214  |
| both noisy (0.15, 0.15)           |          99.0% |   2,560 ms |   0.193  |         100.0% |   2,840 ms |   0.211  |


How each strategy degrades
--------------------------

**Intent-based:**

- **Fails ~1% of requests** under 15% noise (2 of 200 in this run).
  Why only 1%? Most mis-classifications are *harmless* — sending a chat
  request to the prefill lane still works, it's just sub-optimal.
  The 1% of failures are the dangerous ones: a `long_context` request
  mis-labeled as `chat` routes to the decode lane → context overflow →
  hard 400 error. That's 15% × 20% long-context = ~3% expected failures,
  but some mis-labels happen to still route the long-context request to
  a context-capable lane by luck.
- p95 and cost **barely move** under noise (classifier errors mostly
  cancel out across the noise distribution).

**TokenFlow:**

- **Never drops a request** under workload noise. Hard constraints
  (`max_context_tokens`, tenant allowlists, health threshold) don't use
  `workload_type` at all, so a wrong workload-type label can't cause
  context overflow or tenant violations.
- **p95 latency rises from 2,212 → 2,963 ms (+34%)** under workload
  noise — because a mis-inferred workload type pushes requests to a
  sub-optimal lane (e.g., a short-chat labeled as decode-heavy goes to
  the prefill lane, which has higher queue latency for small requests).
- **Cost rises from $0.194 → $0.214 (+10%)** because workload noise
  pushes more traffic to the premium lane.


Key insight — different failure modes
--------------------------------------

|                    | Intent-based                   | TokenFlow                        |
| ------------------ | ------------------------------ | -------------------------------- |
| Under noise, fails | Yes — drops requests           | No — always completes            |
| Under noise, slower| Minimally                      | Yes — p95 up to +34%             |
| Under noise, costlier | Minimally                   | Yes — cost up to +10%            |
| Failure visibility | Client sees HTTP 400/503       | Client sees slower response      |
| Recovery           | Requires classifier retrain    | Self-heals as queues drain       |

The failure modes are **qualitatively different**:

- Intent-based fails **hard** (drops requests) because the classifier is
  a deterministic function of the prompt and there's nothing between the
  classifier's output and the routing decision.
- TokenFlow fails **soft** (slower / more expensive) because
  `workload_type` is one signal of seven feeding into a weighted utility
  score, and hard constraints don't use it at all. The utility
  blending dilutes any single wrong input.

In a production SLO context, soft degradation is almost always
preferable: users tolerate a 10% latency increase, but 1% dropped
requests pages someone.

That said — TokenFlow **is** less efficient under workload noise than
in the clean case, and the cost increase is real. The solution in
production is to cross-validate `predicted_output_tokens` against
historical actual-output distributions for the same tenant/endpoint,
rather than trusting `max_tokens` blindly. That's on the router roadmap.


Raw files
---------

- `bench_i0.0_w0.0.json`   — clean baseline
- `bench_i0.15_w0.0.json`  — 15% intent noise, 0 workload noise
- `bench_i0.0_w0.15.json`  — 0 intent noise, 15% workload noise
- `bench_i0.15_w0.15.json` — both noises at 15%

See `../benchmark_chart_v5.png` for the summary visualisation.
