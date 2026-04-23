TokenFlow Router — live demo with real vLLM backends
=====================================================

This directory contains scenario scripts and a benchmark harness that exercise
the router against two *real* vLLM backends running on a single 8xH100 box.

Fleet under test
----------------

The router is fronting two heterogeneous lanes. Both run on H100 silicon, but
are registered with the router as distinct logical tiers so the policy engine
can differentiate them:

| Lane          | Model                     | GPU slot | cost_class | Purpose                                 |
| ------------- | ------------------------- | -------- | ---------- | --------------------------------------- |
| vllm-fast     | Qwen/Qwen2.5-3B-Instruct  | GPU 0    | economy    | decode-friendly short-context fast lane |
| vllm-quality  | Qwen/Qwen2.5-7B-Instruct  | GPU 1    | premium    | longer-context reasoning-capable lane   |

The policy in `examples/configs/policy.yaml` (`production-balanced`) has 7 DSL
rules that trigger on priority tier, workload type, prompt size, and output
size. The scenario scripts below each exercise one of those rule paths.

Running the scenarios
---------------------

The scripts assume the router is reachable at `http://localhost:8080`. If you
ran the deploy on a remote box, set up a tunnel first:

```
ssh -L 8080:localhost:8080 saurav-projects   # keep this shell open
```

Then in another terminal:

```
./examples/demo/01_short_chat.sh
./examples/demo/02_reasoning.sh
./examples/demo/03_prefill_heavy.sh
./examples/demo/04_premium_tenant.sh
./examples/demo/05_explain.sh
./examples/demo/06_policy_swap.sh
```

Each script prints the response body plus the decision from
`/admin/routes/explain/{request_id}`, which shows:

- which endpoint was selected and the full utility score
- which rules matched
- what alternatives were considered and why they lost

Benchmark
---------

`benchmark.py` drives 150 requests through three arms and produces a
side-by-side table:

- **Arm A (direct):** every request hits `vllm-fast` directly — what you get
  if you bypass the router entirely.
- **Arm B (round-robin):** alternate between `vllm-fast` and `vllm-quality`
  per request — what you get from a dumb load balancer.
- **Arm C (router):** route through TokenFlow with the `production-balanced`
  policy.

Run it with:

```
python examples/demo/benchmark.py --router http://localhost:8080 \
  --fast http://localhost:8001 --quality http://localhost:8002 --n 150
```
