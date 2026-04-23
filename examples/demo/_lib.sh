#!/usr/bin/env bash
# Shared helpers for the demo scenario scripts.
# Each scenario sources this file, then calls `demo_run`.

set -euo pipefail

ROUTER="${ROUTER:-http://localhost:8080}"

demo_need() {
  command -v "$1" >/dev/null 2>&1 || { echo "missing dep: $1" >&2; exit 1; }
}
demo_need curl
demo_need python3
demo_need jq 2>/dev/null || true  # jq is nice but not required

demo_hr() { printf -- '─%.0s' {1..72}; echo; }

demo_run() {
  local title="$1" headers_json="$2" body_json="$3"

  demo_hr
  echo "# $title"
  demo_hr

  local h_args=()
  while IFS= read -r line; do
    [ -n "$line" ] && h_args+=(-H "$line")
  done < <(echo "$headers_json" | python3 -c 'import json,sys
for k,v in json.load(sys.stdin).items():
    print(f"{k}: {v}")')

  echo ">> POST $ROUTER/v1/chat/completions"
  local resp
  resp=$(curl -s -X POST "$ROUTER/v1/chat/completions" \
    -H "Content-Type: application/json" \
    "${h_args[@]}" \
    -d "$body_json")

  echo "$resp" | python3 -m json.tool || echo "$resp"

  local rid
  rid=$(echo "$resp" | python3 -c '
import json, sys
try:
  d = json.load(sys.stdin)
  print(d.get("_tokenflow", {}).get("request_id", ""))
except Exception:
  pass' 2>/dev/null || true)

  if [ -n "$rid" ]; then
    echo
    echo ">> GET $ROUTER/admin/routes/explain/$rid"
    curl -s "$ROUTER/admin/routes/explain/$rid" | python3 -m json.tool || true
  fi
  echo
}
