#!/usr/bin/env bash
# TokenFlow Router — quickstart script
# Starts the router, registers 2 mock NIM endpoints, and runs a test request.

set -e

ROUTER="http://localhost:8080"

echo "==> Starting TokenFlow Router..."
docker-compose up -d tokenflow nim-mock-1 nim-mock-2
sleep 3

echo "==> Registering NIM endpoints..."

curl -s -X POST "$ROUTER/admin/endpoints" \
  -H "Content-Type: application/json" \
  -d '{
    "name": "nim-h100-llama3-8b",
    "nim_url": "http://nim-mock-1:8001",
    "model_name": "meta/llama-3.1-8b-instruct",
    "gpu_name": "H100",
    "cost_class": "premium",
    "cost_per_gpu_hour": 8.0,
    "max_context_tokens": 32768
  }' | python3 -m json.tool

curl -s -X POST "$ROUTER/admin/endpoints" \
  -H "Content-Type: application/json" \
  -d '{
    "name": "nim-l40s-llama3-8b",
    "nim_url": "http://nim-mock-2:8002",
    "model_name": "meta/llama-3.1-8b-instruct",
    "gpu_name": "L40S",
    "cost_class": "standard",
    "cost_per_gpu_hour": 2.5,
    "max_context_tokens": 16384
  }' | python3 -m json.tool

echo ""
echo "==> Sending a test inference request..."
RESPONSE=$(curl -s -X POST "$ROUTER/v1/chat/completions" \
  -H "Content-Type: application/json" \
  -H "x-tenant-id: default" \
  -H "x-priority-tier: standard" \
  -d '{
    "model": "meta/llama-3.1-8b-instruct",
    "messages": [{"role": "user", "content": "Hello, which endpoint am I talking to?"}],
    "max_tokens": 64
  }')

echo "$RESPONSE" | python3 -m json.tool

REQUEST_ID=$(echo "$RESPONSE" | python3 -c "import sys,json; d=json.load(sys.stdin); print(d.get('_tokenflow',{}).get('request_id',''))" 2>/dev/null)

if [ -n "$REQUEST_ID" ]; then
  echo ""
  echo "==> Explaining routing decision for request $REQUEST_ID..."
  curl -s "$ROUTER/admin/routes/explain/$REQUEST_ID" | python3 -m json.tool
fi

echo ""
echo "==> Health check..."
curl -s "$ROUTER/health" | python3 -m json.tool

echo ""
echo "Done! Swagger UI available at: $ROUTER/docs"
