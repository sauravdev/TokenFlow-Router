#!/usr/bin/env bash
# Scenario 4 — tenant-aware routing.
# tenant-premium-corp has allowed_gpu_classes = [B200, H200, H100, A100, L40S]
# and budget $50/hr. Both backends are registered under H100, so both are
# allowed. But the tenant also has priority_tier_override=premium, meaning
# even a request with no explicit tier gets upgraded. Expected: quality lane
# (the premium-tagged one) wins, cost is ignored.

source "$(dirname "$0")/_lib.sh"

demo_run "Scenario 4 — tenant override forces premium tier" \
  '{"x-tenant-id": "tenant-premium-corp"}' \
  '{
    "model": "Qwen/Qwen2.5-7B-Instruct",
    "messages": [{"role": "user", "content": "Explain the CAP theorem in two sentences."}],
    "max_tokens": 120,
    "stream": false
  }'
