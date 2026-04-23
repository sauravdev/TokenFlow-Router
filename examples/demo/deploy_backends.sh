#!/usr/bin/env bash
# Register vllm-fast and vllm-quality with the running router.
# The router reaches them by container name via the shared docker network.
# Prerequisite: both vLLM containers are healthy (/health returns 200) and
# attached to the tokenflow-router_default docker network.

set -euo pipefail

ROUTER="${ROUTER:-http://localhost:8080}"

echo "==> Registering vllm-decode (Qwen2.5-7B, short-ctx decode-tuned)"
curl -s -X POST "$ROUTER/admin/endpoints" \
  -H "Content-Type: application/json" \
  -d '{
    "name": "vllm-decode",
    "nim_url": "http://vllm-decode:8000",
    "backend_type": "vllm",
    "model_name": "qwen",
    "gpu_name": "H100",
    "cost_class": "standard",
    "cost_per_gpu_hour": 4.0,
    "max_context_tokens": 4096,
    "supports_reasoning": true,
    "capability_flags": {"tuning": "decode", "max_num_seqs": 32}
  }' | python3 -m json.tool

echo
echo "==> Registering vllm-prefill (Qwen2.5-7B, long-ctx prefill-tuned)"
curl -s -X POST "$ROUTER/admin/endpoints" \
  -H "Content-Type: application/json" \
  -d '{
    "name": "vllm-prefill",
    "nim_url": "http://vllm-prefill:8000",
    "backend_type": "vllm",
    "model_name": "qwen",
    "gpu_name": "H100",
    "cost_class": "standard",
    "cost_per_gpu_hour": 4.0,
    "max_context_tokens": 32768,
    "supports_reasoning": true,
    "capability_flags": {"tuning": "prefill", "chunked_prefill": true, "max_num_batched_tokens": 8192}
  }' | python3 -m json.tool

echo
echo "==> Registered endpoints:"
curl -s "$ROUTER/admin/endpoints" | python3 -m json.tool
