#!/usr/bin/env bash
# Scenario 6 — live policy swap without restarts.
# Send the same request under three policies: latency-first, cost-first,
# balanced. Observe that the routing winner (or at least the utility scores)
# shifts when the weights change. No backend restart required.

ROUTER="${ROUTER:-http://localhost:8080}"

probe() {
  local tag="$1"
  printf -- '─%.0s' {1..72}; echo
  echo "# Policy = $tag"
  printf -- '─%.0s' {1..72}; echo

  curl -s -X POST "$ROUTER/admin/policy/preset" \
    -H "Content-Type: application/json" \
    -d "{\"preset\": \"$tag\"}" | python3 -m json.tool | head -5
  echo

  RESP=$(curl -s -X POST "$ROUTER/v1/chat/completions" \
    -H "Content-Type: application/json" \
    -H "x-tenant-id: default" \
    -H "x-priority-tier: standard" \
    -d '{
      "model": "Qwen/Qwen2.5-3B-Instruct",
      "messages": [{"role": "user", "content": "What is 7 * 8?"}],
      "max_tokens": 16,
      "stream": false
    }')
  echo "$RESP" | python3 -m json.tool | grep -E '"_tokenflow"|request_id|endpoint' || echo "$RESP" | head -20

  RID=$(echo "$RESP" | python3 -c 'import json,sys; d=json.load(sys.stdin); print(d.get("_tokenflow",{}).get("request_id",""))')
  if [ -n "$RID" ]; then
    echo
    echo "-- utility breakdown --"
    curl -s "$ROUTER/admin/routes/explain/$RID" | python3 -c '
import json, sys
d = json.load(sys.stdin)
dec = d.get("decision") or {}
winner = dec.get("selected_endpoint_name") or d.get("selected_endpoint_name") or "?"
print("winner: " + str(winner))
cands = dec.get("candidate_scores") or d.get("candidate_scores") or []
for c in cands:
  if isinstance(c, dict):
    name = c.get("endpoint_name", "?")
    util = c.get("utility_score", c.get("utility", "?"))
    print("  - " + str(name) + ": utility=" + str(util))
'
  fi
  echo
}

probe latency-first
probe cost-first
probe balanced

echo "=== Reverting to balanced (deploy default) ==="
curl -s -X POST "$ROUTER/admin/policy/preset" \
  -H "Content-Type: application/json" \
  -d '{"preset": "balanced"}' | python3 -m json.tool | head -5
