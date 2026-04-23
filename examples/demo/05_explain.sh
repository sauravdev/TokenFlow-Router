#!/usr/bin/env bash
# Scenario 5 — routing decision deep dive.
# Send two requests with different shapes back-to-back, then inspect the
# explanation for each. This shows that the router produces a full decision
# trace per request: winner, losers, per-endpoint utility components, and
# which rules fired. Useful for debugging "why did my request land there?"

ROUTER="${ROUTER:-http://localhost:8080}"

send() {
  curl -s -X POST "$ROUTER/v1/chat/completions" \
    -H "Content-Type: application/json" \
    -H "x-tenant-id: default" \
    -H "x-priority-tier: $1" \
    -d "$2"
}

echo "=== Request A (priority=standard, short) ==="
RA=$(send standard '{
  "model": "Qwen/Qwen2.5-3B-Instruct",
  "messages": [{"role": "user", "content": "Hi"}],
  "max_tokens": 8,
  "stream": false
}')
echo "$RA" | python3 -m json.tool | head -20
RID_A=$(echo "$RA" | python3 -c 'import json,sys; d=json.load(sys.stdin); print(d.get("_tokenflow",{}).get("request_id",""))')

echo
echo "=== Request B (priority=premium, long decode) ==="
RB=$(send premium '{
  "model": "Qwen/Qwen2.5-7B-Instruct",
  "messages": [{"role": "user", "content": "Write a detailed 800-word essay on the tradeoffs between eventual and strong consistency in distributed databases."}],
  "max_tokens": 1024,
  "stream": false
}')
echo "$RB" | python3 -m json.tool | head -20
RID_B=$(echo "$RB" | python3 -c 'import json,sys; d=json.load(sys.stdin); print(d.get("_tokenflow",{}).get("request_id",""))')

echo
printf -- '─%.0s' {1..72}; echo
echo "# Decision trace — Request A ($RID_A)"
printf -- '─%.0s' {1..72}; echo
curl -s "$ROUTER/admin/routes/explain/$RID_A" | python3 -m json.tool

echo
printf -- '─%.0s' {1..72}; echo
echo "# Decision trace — Request B ($RID_B)"
printf -- '─%.0s' {1..72}; echo
curl -s "$ROUTER/admin/routes/explain/$RID_B" | python3 -m json.tool

echo
echo "=== Recent decisions (last 5) ==="
curl -s "$ROUTER/admin/routes/recent?limit=5" | python3 -m json.tool
