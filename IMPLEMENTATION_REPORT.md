# TokenFlow Router implementation report

## Summary

This change set adds request-level routing intent (`latency` vs `throughput`), benchmark-aware backend priors for NIM/vLLM/SGLang/Dynamo/Ollama, and smarter lazy profile activation that only wakes backends relevant to the requested model/workload.

## What changed

### 1. User intent: latency vs throughput
- Added `OptimizationTarget` enum with `latency`, `throughput`, and `auto`
- Extended `RequestProfile` to carry `optimization_target`
- Added request parsing from:
  - `routing.optimize_for` in the JSON body
  - `X-Optimization-Target` header
- `auto` resolves to:
  - `latency` for streaming, reasoning, and prefill-heavy requests
  - `throughput` otherwise

### 2. Benchmark-aware backend routing
- Added `tokenflow/benchmarks.py`
- Encoded routing priors for:
  - NIM
  - vLLM
  - SGLang
  - Dynamo
  - Ollama
- Benchmarks are used as priors, not hard-coded absolutes
- Routing now blends:
  - backend affinity score
  - benchmark score
  - live SLO, queue, reliability, cost, and model-fit scoring

### 3. End-user benefit surfaced
Successful non-streaming responses now include `_tokenflow.optimization_target` and a short `end_user_benefit` explanation.

### 4. Smarter engine spin-up
- Replaced coarse `maybe_activate_for_workload(...)`
- Added `maybe_activate_for_request(profile)`
- Profile activation now filters by:
  - workload affinity
  - requested model/family via `activation_model_names`
- This reduces unnecessary multi-engine warm-up and lowers the odds of keeping redundant model copies resident just because they share a workload class.

## Why this benefits end users

### If the user chooses `latency`
They care about responsiveness:
- lower time-to-first-token
- less interactive jitter
- better UX for chat, copilots, support, and reasoning assistants
- lower risk of being routed to a cold or concurrency-optimized backend that feels sluggish at request start

### If the user chooses `throughput`
They care about aggregate work done:
- higher sustained tokens/sec under load
- better fleet utilization
- lower queue buildup during bursts
- improved economics for batch jobs, long generations, background summarization, and high-volume API traffic

### Why benchmark priors help
Live telemetry is great when present, but often sparse at cold start or for newly activated backends. Benchmark priors let the router make sane first decisions before enough request history accumulates.

### Why selective activation matters
Keeping every engine hot wastes VRAM and can force duplicate model residency across NIM/vLLM/SGLang/Ollama. Matching activation to the incoming model and workload reduces memory waste and makes it easier to reserve premium GPU memory for the engines that are actually delivering value.

## Validation status

### Completed
- Python syntax validation with `python3 -m compileall tokenflow tests`
- Added/updated unit and integration tests covering:
  - optimization target resolution
  - benchmark preference shifts
  - Ollama backend support
  - API parsing of routing intent

### Blocked in sandbox
- Full `pytest` execution was blocked because the environment cannot fetch Python build dependencies from PyPI (`hatchling` install failure via restricted proxy/network path)

## Remaining gap before PR
- Run the full test suite in a network-enabled/dev-ready environment
- Fix any integration regressions from real dependency resolution
- Push branch and open PR once repository credentials/access are available
