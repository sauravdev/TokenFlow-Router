# NVIDIA-compatible classifier shim

Tiny FastAPI service that speaks the same `POST /recommendation` API as
the NVIDIA AI Blueprints LLM Router v2. TokenFlow's
`tokenflow.integrations.external_classifier.ExternalClassifierClient`
talks to this service exactly the way it would talk to a real NVIDIA
v2 deployment — the wire format is byte-compatible.

## Why a shim?

The real NVIDIA Router v2 deploys a Qwen 1.7B classifier in vLLM
alongside a Rust router-backend. It works, but on a fleet that is
already saturated with model serving traffic, dedicating a GPU to the
classifier itself is wasteful. This shim:

1. Handles ~85% of traffic with cheap regex/length heuristics
   (microsecond latency, no GPU, no model load).
2. Optionally defers borderline classifications (`0.55 ≤ confidence ≤
   0.78`) to a small vLLM endpoint as an LLM-as-judge — same idea as
   NVIDIA's classifier, but reusing an existing serving lane instead of
   provisioning a dedicated one.

In production you can run this shim *or* swap in the real NVIDIA
container; TokenFlow doesn't know the difference.

## Run

```bash
cd examples/telco_demo/classifier_shim

# heuristic-only mode (fastest, ~50µs per classification)
docker build -t nvidia-router-shim:latest .
docker run -d --name nvidia-router-shim \
  --network tokenflow-router_default \
  -p 8090:8090 \
  nvidia-router-shim:latest

# OR with LLM-as-judge fallback (defers borderline cases to vllm-economy)
docker run -d --name nvidia-router-shim \
  --network tokenflow-router_default \
  -p 8090:8090 \
  -e CLASSIFIER_JUDGE_URL=http://vllm-economy:8000 \
  -e CLASSIFIER_JUDGE_MODEL=qwen \
  nvidia-router-shim:latest

# Wire into TokenFlow
docker exec tokenflow-router-tokenflow-1 \
  bash -c "echo 'TOKENFLOW_EXTERNAL_CLASSIFIER_URL=http://nvidia-router-shim:8090' >> /app/.env"
docker compose restart tokenflow
```

## API

### POST /recommendation

Request:

```json
{
  "messages": [{"role": "user", "content": "..."}],
  "model": "qwen",
  "metadata": {"tenant_id": "...", "priority_tier": "premium"}
}
```

Response:

```json
{
  "intent": "hard_question",
  "recommended_model": "vllm-premium",
  "confidence": 0.83,
  "model_scores": {
    "vllm-economy": 0.05,
    "vllm-standard": 0.20,
    "vllm-premium": 0.75
  },
  "classifier_latency_ms": 0.087,
  "classifier_source": "heuristic"
}
```

### Intent labels (NVIDIA v2 compatible)

| Label              | Maps to TokenFlow | Suggests model |
|--------------------|-------------------|----------------|
| `hard_question`    | `reasoning`       | premium        |
| `summary_request`  | `summarization`   | standard       |
| `creative_writing` | `generation`      | standard       |
| `chit_chat`        | `chat`            | economy        |

The intent is *advisory* — TokenFlow folds it into the multi-signal
score, which still respects cost weighting, GPU affinity, queue depth,
SLO target, and tenant policy. The classifier never bypasses policy.

## Observability

```bash
curl localhost:8090/health
# {
#   "status": "healthy",
#   "judge_url": "http://vllm-economy:8000",
#   "stats": {
#     "requests": 1247,
#     "judge_calls": 119,        # borderline cases
#     "judge_overrides": 31,     # judge disagreed with heuristic
#     "errors": 0
#   }
# }
```
