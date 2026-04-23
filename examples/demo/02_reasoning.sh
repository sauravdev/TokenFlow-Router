#!/usr/bin/env bash
# Scenario 2 — reasoning workload, premium tier.
# Multi-step analytical task. The `premium-traffic-on-premium-gpu` rule
# (priority 10) sets budget_sensitivity=0.0, meaning cost is ignored. The
# `reasoning-on-nim-preferred` rule (priority 15) tags this as reasoning.
# Combined effect: router should pick the quality lane (Qwen2.5-7B) even
# though it costs more per GPU-hour — SLO + model capability dominate.

source "$(dirname "$0")/_lib.sh"

demo_run "Scenario 2 — reasoning (expect quality lane)" \
  '{"x-tenant-id": "tenant-premium-corp", "x-priority-tier": "premium"}' \
  '{
    "model": "Qwen/Qwen2.5-7B-Instruct",
    "messages": [
      {"role": "system", "content": "You are a senior systems engineer. Think step by step before answering."},
      {"role": "user", "content": "A service has two queues: a primary and a fallback. The primary has 99.5% availability; the fallback has 99.0% independent availability. If the client retries the fallback once on primary failure, what is the end-to-end request success rate? Show the derivation."}
    ],
    "max_tokens": 512,
    "stream": false
  }'
