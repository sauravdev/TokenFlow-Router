#!/usr/bin/env bash
# Telco-shape demo — register three lanes with the router using the
# multi-workload policy.

set -euo pipefail

ROUTER="${ROUTER:-http://localhost:8080}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "==> Loading multi-workload policy"
python3 - "$SCRIPT_DIR/configs/policy.yaml" <<'EOF'
import json, sys, urllib.request, yaml
path = sys.argv[1]
policy = yaml.safe_load(open(path))
req = urllib.request.Request(
    "http://localhost:8080/admin/policy",
    data=json.dumps(policy).encode(),
    headers={"Content-Type": "application/json"},
    method="POST",
)
try:
    r = urllib.request.urlopen(req, timeout=5).read().decode()
    print("  policy loaded:", json.loads(r).get("name", "?"))
except Exception as e:
    print(f"  policy load error: {e}")
EOF

echo
echo "==> Registering vllm-economy (Qwen2.5-3B / 1x A100 / \$2.50 GPU-hr)"
curl -s -X POST "$ROUTER/admin/endpoints" \
  -H "Content-Type: application/json" \
  -d '{
    "name": "vllm-economy",
    "nim_url": "http://vllm-economy:8000",
    "backend_type": "vllm",
    "model_name": "qwen",
    "gpu_name": "A100",
    "cost_class": "economy",
    "cost_per_gpu_hour": 2.5,
    "max_context_tokens": 4096,
    "supports_reasoning": false
  }' | python3 -m json.tool | head -3

echo
echo "==> Registering vllm-standard (Qwen2.5-14B / 1x A100 / \$5 GPU-hr)"
curl -s -X POST "$ROUTER/admin/endpoints" \
  -H "Content-Type: application/json" \
  -d '{
    "name": "vllm-standard",
    "nim_url": "http://vllm-standard:8000",
    "backend_type": "vllm",
    "model_name": "qwen",
    "gpu_name": "A100",
    "cost_class": "standard",
    "cost_per_gpu_hour": 5.0,
    "max_context_tokens": 16384,
    "supports_reasoning": true
  }' | python3 -m json.tool | head -3

echo
echo "==> Registering vllm-premium (Qwen2.5-72B / 2x A100 / \$12 GPU-hr)"
curl -s -X POST "$ROUTER/admin/endpoints" \
  -H "Content-Type: application/json" \
  -d '{
    "name": "vllm-premium",
    "nim_url": "http://vllm-premium:8000",
    "backend_type": "vllm",
    "model_name": "qwen",
    "gpu_name": "A100",
    "gpu_count": 2,
    "cost_class": "premium",
    "cost_per_gpu_hour": 12.0,
    "max_context_tokens": 32768,
    "supports_reasoning": true
  }' | python3 -m json.tool | head -3

echo
echo "==> Registered endpoints"
curl -s "$ROUTER/admin/endpoints" | python3 -c '
import sys, json
for e in json.load(sys.stdin):
    print(f"  {e[\"name\"]:14s}  {e[\"cost_class\"]:9s}  ${e[\"cost_per_gpu_hour\"]}/hr  ctx={e[\"max_context_tokens\"]}  {e[\"health\"]}")
'
