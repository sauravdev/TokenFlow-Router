#!/usr/bin/env bash
# Scenario 3 — prefill-heavy summarization.
# Long input (~2k tokens), short output. The `large-prompt-boost-slo` rule
# (priority 30) fires when input_tokens > 8000, but for a 2k input it relies
# on benchmarks.py's prefill_tps affinity scoring — the quality lane's larger
# model has worse per-token prefill, so the router should still prefer the
# fast lane for prefill-dominated short-context requests.

source "$(dirname "$0")/_lib.sh"

LONG_PROMPT=$(python3 -c '
import textwrap
paras = [
  "The Voyager 1 spacecraft, launched in September 1977, has become the most distant human-made object from Earth.",
  "After completing flybys of Jupiter in 1979 and Saturn in 1980, Voyager 1 embarked on an extended mission to explore the outer solar system and eventually interstellar space.",
  "In August 2012, Voyager 1 crossed the heliopause, entering the interstellar medium.",
  "Its plutonium-238 radioisotope thermoelectric generators continue to provide power, though at a steadily declining rate.",
  "Engineers at NASA periodically shut down instruments and heaters to extend the mission.",
  "The spacecraft carries a Golden Record, a phonograph record containing sounds and images selected to portray the diversity of life and culture on Earth.",
  "Communication with Voyager 1 takes over 22 hours one way at current distances.",
  "Its twin, Voyager 2, followed a different trajectory and conducted flybys of Uranus in 1986 and Neptune in 1989.",
  "Both spacecraft use the Deep Space Network for communication.",
  "As of the 2020s, both spacecraft are expected to continue operating until roughly 2030 before power becomes insufficient.",
]
print(" ".join(paras * 15))  # ~2k tokens
')

demo_run "Scenario 3 — prefill-heavy (expect fast lane via affinity)" \
  '{"x-tenant-id": "default", "x-priority-tier": "standard"}' \
  "$(python3 -c "
import json, sys
body = {
  'model': 'Qwen/Qwen2.5-3B-Instruct',
  'messages': [{'role': 'user', 'content': 'Summarise this in 3 bullet points:\n\n' + '''$LONG_PROMPT'''}],
  'max_tokens': 128,
  'stream': False
}
print(json.dumps(body))
")"
