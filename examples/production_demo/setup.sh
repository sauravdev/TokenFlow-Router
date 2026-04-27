#!/usr/bin/env bash
# Production-demo setup. Registers two backends with the router using the
# multi-tenant policy from configs/policy.yaml.

set -euo pipefail

ROUTER="${ROUTER:-http://localhost:8080}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "==> Loading multi-tenant policy"
POLICY=$(cat "$SCRIPT_DIR/configs/policy.yaml")
curl -s -X POST "$ROUTER/admin/policy" \
  -H "Content-Type: application/json" \
  -d "$(python3 -c "
import json, sys, yaml
print(json.dumps(yaml.safe_load(sys.stdin.read())))" <<< "$POLICY")" \
  | python3 -m json.tool | head -3

echo
echo "==> Registering vllm-fast (economy / Qwen2.5-3B / \$2.50 GPU-hr)"
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
  }' | python3 -m json.tool | head -3

echo
echo "==> Registering vllm-quality (premium / Qwen2.5-7B / \$8.00 GPU-hr)"
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
    "max_context_tokens": 32768,
    "supports_reasoning": true
  }' | python3 -m json.tool | head -3

echo
echo "==> Registered endpoints"
curl -s "$ROUTER/admin/endpoints" | python3 -c '
import sys, json
for e in json.load(sys.stdin):
    print(f"  {e[\"name\"]:14s}  {e[\"cost_class\"]:9s}  ${e[\"cost_per_gpu_hour\"]}/hr  ctx={e[\"max_context_tokens\"]}")
'
