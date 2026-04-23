#!/usr/bin/env bash
# Scenario 1 — short interactive chat.
# Small prompt, small output, standard tier. The default tenant has no GPU
# restriction, so the policy engine picks based on latency + cost. Expected:
# the fast lane (vllm-fast / Qwen2.5-3B) wins on utility — it's cheaper per
# request and its decode latency is lower for a small generation.

source "$(dirname "$0")/_lib.sh"

demo_run "Scenario 1 — short interactive chat (expect fast lane)" \
  '{"x-tenant-id": "default", "x-priority-tier": "standard"}' \
  '{
    "model": "Qwen/Qwen2.5-3B-Instruct",
    "messages": [{"role": "user", "content": "What is the capital of France? Answer in one word."}],
    "max_tokens": 16,
    "stream": false
  }'
