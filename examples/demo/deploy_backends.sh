#!/usr/bin/env bash
# Register vllm-fast and vllm-quality with the running router.
# The router reaches them by container name via the shared docker network.
# Prerequisite: both vLLM containers are healthy (/health returns 200) and
# attached to the tokenflow-router_default docker network.

set -euo pipefail

ROUTER="${ROUTER:-http://localhost:8080}"

echo "==> Registering vllm-fast (economy / Qwen2.5-3B)"
curl -s -X POST "$ROUTER/admin/endpoints" \
  -H "Content-Type: application/json" \
  -d '{
    "name": "vllm-fast",
    "nim_url": "http://vllm-fast:8000",
    "backend_type": "vllm",
    "model_name": "qwen",
    "gpu_name": "H100",
    "cost_class": "economy",
    "cost_per_gpu_hour": 2.5,
    "max_context_tokens": 4096,
    "supports_reasoning": false
  }' | python3 -m json.tool

echo
echo "==> Registering vllm-quality (premium / Qwen2.5-7B)"
curl -s -X POST "$ROUTER/admin/endpoints" \
  -H "Content-Type: application/json" \
  -d '{
    "name": "vllm-quality",
    "nim_url": "http://vllm-quality:8000",
    "backend_type": "vllm",
    "model_name": "qwen",
    "gpu_name": "H100",
    "cost_class": "premium",
    "cost_per_gpu_hour": 8.0,
    "max_context_tokens": 16384,
    "supports_reasoning": true
  }' | python3 -m json.tool

echo
echo "==> Registered endpoints:"
curl -s "$ROUTER/admin/endpoints" | python3 -m json.tool
