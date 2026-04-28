TokenFlow Router + NVIDIA AI Blueprints LLM Router (v2) — composition
=====================================================================

The two projects have different design centers and **compose well**.
NVIDIA's v2 is a classifier-as-a-service that returns model-recommendation
labels per prompt. TokenFlow is a proxy + policy layer that makes the
actual routing decision based on multiple signals (cost, queue depth,
context-fit, tenant policy, …).

This directory shows how to wire them together so each does what it's
good at:

```
                client
                  │
                  ▼
       ┌──────────────────────────────┐
       │ TokenFlow Router (proxy +    │
       │ policy + scoring engine)     │
       │                              │
       │   request profile builder ◀──┼──── ① calls NVIDIA classifier
       │            │                 │      (intent / recommended model)
       │            ▼                 │
       │   scoring engine — folds in  │
       │   intent as one signal       │
       │   alongside cost / queue /   │
       │   context-fit / tenant rule  │
       │                              │
       │            ▼                 │
       │   selected backend ───────── │ ────▶ NIM / vLLM / SGLang / Dynamo
       │                              │       Ollama / OpenAI-frontier
       └──────────────────────────────┘
```

Why this composition makes sense
--------------------------------

| Layer                                     | Best done by                  |
| ----------------------------------------- | ----------------------------- |
| Classify "what kind of prompt is this?"   | Trained classifier (NVIDIA's) |
| Decide which backend gets it              | Multi-signal scoring (TokenFlow) |
| Enforce tenant budget / RPM / GPU rules   | Policy engine (TokenFlow)     |
| Forward inference traffic                 | Proxy (TokenFlow)             |
| Spin up dormant backends on demand        | Capacity controller (TokenFlow) |
| Observability & per-request decision trace | Prometheus + /admin/routes/explain (TokenFlow) |
| Multimodal (image + text) intent          | NVIDIA's CLIP+NN classifier   |

**TokenFlow handles enforcement; NVIDIA's classifier becomes one input.**
A classifier mistake doesn't cause a hard failure — it shifts the
soft-scoring components, but hard constraints (context-fit, tenant
allowlist) still catch the worst cases. That's the noise-robustness
property covered in the production benchmark.


How it's wired
--------------

TokenFlow ships an `ExternalClassifierClient` that any classifier
service speaking the standard request/response shape can plug into:

```python
from tokenflow.integrations.external_classifier import (
    ExternalClassifierClient, install_into_state,
)

client = ExternalClassifierClient(
    classifier_url="http://nvidia-router-v2:5000",
    timeout_s=0.5,        # tight — never blocks the hot path
)

install_into_state(app.state, client)
```

After `install_into_state`, every request that hits the gateway gets
classified first by NVIDIA's router; the canonical intent label
(`reasoning` / `chat` / `summarization` / `generation` / etc.) is
folded into TokenFlow's `RequestProfile.workload_type`. The scoring
engine and policy DSL keep working unchanged — they just see a more
accurate workload-type label than the keyword baseline.

If the classifier times out, errors, or returns an intent TokenFlow
doesn't recognise, the local heuristic kicks in. The classifier is
**hot-path-safe** — never a SPOF.


Wire format
-----------

`ExternalClassifierClient` POSTs:

```json
{
  "messages": [
    {"role": "system", "content": "..."},
    {"role": "user",   "content": "..."}
  ],
  "model": "qwen",
  "metadata": {"tenant_id": "...", "priority_tier": "..."}
}
```

…to `{classifier_url}/recommendation` and expects:

```json
{
  "intent": "reasoning",                    // or chat / summarization / ...
  "recommended_model": "gpt-5-chat",        // optional
  "confidence": 0.83,                        // optional
  "model_scores": {"gpt-5-chat": 0.83, ...} // optional
}
```

NVIDIA's v2 router exposes this shape on its `nat_sfc_router`
recommendation endpoint. Other classifiers (distilBERT services,
LLM-as-judge, etc.) can adopt the same wire format.

Mapping table from NVIDIA v2 intent labels to TokenFlow's
`WorkloadType` enum is in
`tokenflow/integrations/external_classifier.py:_INTENT_MAP`.


Example deploy (Docker Compose)
-------------------------------

`docker-compose.yml` in this directory brings up:

  1. NVIDIA AI Blueprints LLM Router v2 (intent profile)
  2. TokenFlow Router with `external_classifier` wired to it
  3. Two vLLM backends for inference

Run:

```bash
cd examples/integrations/nvidia_router_v2
docker compose up -d
curl http://localhost:8080/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"qwen","messages":[{"role":"user","content":"derive Bayes rule from first principles"}],"max_tokens":256}'

# Inspect the routing decision
curl http://localhost:8080/admin/routes/explain/<request_id> | python3 -m json.tool
```

The `_tokenflow.request_shape.workload_type` field in the response
will show `reasoning` (from NVIDIA's classifier) instead of the
keyword baseline's `balanced`. The policy's
`reasoning-on-premium-gpu` rule fires; the request lands on the 7B
backend; the explain trace shows both the classifier's confidence
and the scoring engine's utility breakdown.


When to use this composition vs either one alone
------------------------------------------------

**Use both when:**
- You need accurate intent classification (NVIDIA's CLIP+NN trained
  on production traffic beats keyword baselines)
- AND you need tenant policy / cost optimization / live policy swap /
  on-demand spin-up / Prometheus observability (TokenFlow's layer)
- AND/OR multimodal (image+text) routing (NVIDIA can classify image
  content; TokenFlow uses the resulting label as a routing signal)

**Use TokenFlow alone when:**
- Your routing decision is dominated by fleet state (queue depth,
  cost tiers, context-fit) rather than prompt semantics
- A keyword classifier or hand-authored rules are accurate enough for
  your traffic mix (e.g., the production_demo benchmark in this repo
  shows the keyword baseline routing 100% of long-context correctly
  via hard ctx-fit constraints)
- You don't have a separate classifier service deployed

**Use NVIDIA v2 alone when:**
- You're building from scratch and want a recommendation API
- You handle enforcement (tenant budgets, proxying, observability) in
  a different layer (your existing API gateway / service mesh)
- You're explicitly multimodal-first and don't need fleet-aware
  routing

See `../../production_demo/COMPARISON.md` in this repo for a full
feature head-to-head matrix.
