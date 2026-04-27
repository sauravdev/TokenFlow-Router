"""TokenFlow Router — production scenario benchmark.

A single 10-minute benchmark exercising five concrete TokenFlow advantages
against a head-to-head intent-based router, on the *same* fleet, with the
*same* seeded workload. The scenario models a multi-tenant SaaS LLM
platform (free / standard / enterprise tiers) handling a realistic mix
of chat, reasoning, summarization, long-context, and decode-heavy
traffic.

What this benchmark measures
============================

  1. Multi-cost-tier optimization
     vllm-fast (Qwen2.5-3B, $2.50/GPU-hr) vs vllm-quality (Qwen2.5-7B,
     $8/GPU-hr). Intent sends "hard" intents to premium regardless of
     cost. TokenFlow weights cost in the utility function.

  2. Hard constraints over inferred signals
     vllm-fast has max_context_tokens=4096. Long-context requests
     (~5,500 tokens) cannot fit. TokenFlow's hard filter rejects fast
     for these requests; intent's classifier may mis-label them.

  3. Per-tenant policy enforcement
     Three tenants with different headers, budget caps, GPU allowlists.
     Intent doesn't read tenants; TokenFlow does.

  4. Live policy swap (mid-run)
     At T+5min, switch from `balanced` to `cost-first` via
     `POST /admin/policy/preset`. TokenFlow shifts traffic mix
     immediately. Intent is unaffected (no equivalent concept).

  5. Apples-to-apples
     Both arms see the same backends, same workload, same seed, same
     duration. The only difference is the routing brain.

Output
======

    {
      "scenario": "saas-multi-tenant",
      "duration_s": 600,
      "phases": [...],
      "summary": [
        {"arm": "intent", "tenant": "free", ...},
        {"arm": "intent", "tenant": "standard", ...},
        {"arm": "intent", "tenant": "enterprise", ...},
        {"arm": "tokenflow", "tenant": "free", ...},
        ...
      ],
      "raw": {...},
      "policy_swap_at_s": 300
    }

Everything in `summary` is per-(arm, tenant) so we can show TokenFlow's
tenant-level enforcement vs intent's lack of it.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import random
import statistics
import sys
import time
from dataclasses import asdict, dataclass, field
from typing import Any

import httpx


# ---------------------------------------------------------------------------
# Tenant model
# ---------------------------------------------------------------------------


@dataclass
class Tenant:
    name: str
    weight: float                    # share of total traffic
    priority_tier: str               # standard / batch / premium
    budget_usd_per_hour: float       # cap (router enforces; intent ignores)
    allowed_gpu_classes: list[str]


TENANTS: list[Tenant] = [
    Tenant("free",
           weight=0.45,
           priority_tier="batch",
           budget_usd_per_hour=1.0,
           allowed_gpu_classes=["L4", "A10G", "L40S", "RTX4090"]),
    Tenant("standard",
           weight=0.40,
           priority_tier="standard",
           budget_usd_per_hour=10.0,
           allowed_gpu_classes=["H100", "A100", "L40S", "L4"]),
    Tenant("enterprise",
           weight=0.15,
           priority_tier="premium",
           budget_usd_per_hour=100.0,
           allowed_gpu_classes=["B200", "H200", "H100", "A100"]),
]


def pick_tenant(rng: random.Random) -> Tenant:
    r = rng.random()
    acc = 0.0
    for t in TENANTS:
        acc += t.weight
        if r <= acc:
            return t
    return TENANTS[-1]


# ---------------------------------------------------------------------------
# Workload — mixed shapes representative of real SaaS traffic
# ---------------------------------------------------------------------------


_LONG_DOC = " ".join([
    "The OpenAI Chat Completions API exposes streaming SSE for incremental token delivery, which most clients consume via the `data:` prefix protocol.",
    "Modern inference servers like vLLM, NIM, and SGLang implement this contract and additionally expose Prometheus metrics for queue depth, KV-cache fill, and TTFT histograms.",
    "Production LLM platforms commonly deploy heterogeneous fleets where small models handle chat-shape requests and larger models handle reasoning, summarization, and long-context workloads.",
    "Routing between these backends is typically driven either by prompt classification (intent-based) or by a multi-signal scoring engine that considers fleet state, request shape, and tenant policy.",
    "The two approaches differ most visibly when a request has a large input context: intent classifiers may mis-label, while fleet-aware routers reject context-incompatible backends as a hard constraint.",
]) * 25  # ~5,500 tokens


WORKLOAD_SHAPES = [
    {
        "name": "short_chat",
        "weight": 0.45,
        "slo_ms": 3000,
        "gen": lambda: {
            "messages": [{"role": "user", "content": random.choice([
                "What is the capital of France?",
                "How do I write a haiku?",
                "What year did the Berlin Wall fall?",
                "Convert 100 USD to EUR (use 0.92 rate).",
                "List 3 benefits of REST APIs.",
            ])}],
            "max_tokens": 32,
        },
    },
    {
        "name": "reasoning",
        "weight": 0.20,
        "slo_ms": 12000,
        "gen": lambda: {
            "messages": [
                {"role": "system", "content": "Think step by step before answering."},
                {"role": "user", "content": random.choice([
                    "If event A happens with probability 0.6 and event B independently with probability 0.4, what is P(A or B)? Show derivation.",
                    "A salesman visits 4 cities in random order. What's the probability he visits city A first? Derive it.",
                    "If a function is monotonic on [0,1] and continuous, must it be differentiable? Explain.",
                ])},
            ],
            "max_tokens": 256,
        },
    },
    {
        "name": "long_context",
        "weight": 0.15,
        "slo_ms": 10000,
        "gen": lambda: {
            "messages": [{"role": "user", "content":
                "Extract 3 key technical claims from the following text:\n\n" + _LONG_DOC
            }],
            "max_tokens": 120,
        },
    },
    {
        "name": "summarization",
        "weight": 0.12,
        "slo_ms": 8000,
        "gen": lambda: {
            "messages": [{"role": "user", "content":
                "Summarize the following technical document in 2 sentences:\n\n" + (_LONG_DOC[:1500])
            }],
            "max_tokens": 80,
        },
    },
    {
        "name": "decode_heavy",
        "weight": 0.08,
        "slo_ms": 15000,
        "gen": lambda: {
            "messages": [{"role": "user", "content": random.choice([
                "Write a 300-word short story about a lighthouse keeper who befriends a crow.",
                "Explain how B-tree indexes work in 300 words, with one concrete example.",
                "Write a 300-word essay on the design tradeoffs of CRDTs.",
            ])}],
            "max_tokens": 350,
        },
    },
]


def pick_shape(rng: random.Random) -> dict:
    r = rng.random()
    acc = 0.0
    for s in WORKLOAD_SHAPES:
        acc += s["weight"]
        if r <= acc:
            return s
    return WORKLOAD_SHAPES[-1]


# ---------------------------------------------------------------------------
# Intent-based router (Arm A) — simple keyword classifier
# ---------------------------------------------------------------------------


def classify_intent(messages: list[dict]) -> str:
    """Real-world keyword-based intent classifier."""
    text = " ".join(m.get("content", "") for m in messages).lower()
    if any(p in text for p in (
        "step by step", "derive", "probability", "explain why", "show derivation",
    )):
        return "reasoning"
    if any(p in text for p in (
        "summarise", "summarize", "extract", "bullet", "tl;dr",
    )):
        return "summarization"
    if any(p in text for p in (
        "write a", "essay", "300 words", "story", "explain how",
    )):
        return "generation"
    return "chat"


# Intent → backend map. Hard intents to premium 7B; chat to economy 3B.
INTENT_TO_BACKEND = {
    "reasoning":     "quality",
    "summarization": "quality",
    "generation":    "quality",
    "chat":          "fast",
}


# ---------------------------------------------------------------------------
# Per-request result + summary
# ---------------------------------------------------------------------------


@dataclass
class RequestRecord:
    arm: str
    tenant: str
    shape: str
    slo_ms: float
    endpoint_used: str
    ok: bool
    status: int
    latency_ms: float
    tokens_in: int
    tokens_out: int
    cost_usd: float
    phase: str = "main"


COST_PER_GPU_HOUR = {
    "vllm-fast":    2.5,
    "vllm-quality": 8.0,
    "unknown":      0.0,
}


def cost_for(endpoint: str, latency_ms: float) -> float:
    return COST_PER_GPU_HOUR.get(endpoint, 0.0) * latency_ms / 1000.0 / 3600.0


# ---------------------------------------------------------------------------
# Workload generator
# ---------------------------------------------------------------------------


@dataclass
class GeneratedRequest:
    idx: int
    tenant: Tenant
    shape: dict
    body: dict


def build_workload(n: int, seed: int) -> list[GeneratedRequest]:
    rng = random.Random(seed)
    out = []
    for i in range(n):
        tenant = pick_tenant(rng)
        shape = pick_shape(rng)
        body = shape["gen"]()
        out.append(GeneratedRequest(idx=i, tenant=tenant, shape=shape, body=body))
    return out


# ---------------------------------------------------------------------------
# HTTP arms
# ---------------------------------------------------------------------------


async def send_request(client, url: str, model: str, body: dict,
                       headers: dict | None = None,
                       ) -> tuple[int, dict | None, float]:
    t0 = time.perf_counter()
    try:
        r = await client.post(
            f"{url}/v1/chat/completions",
            json={"model": model, **body},
            headers=headers or {},
            timeout=60.0,
        )
        latency = (time.perf_counter() - t0) * 1000
        try:
            j = r.json()
        except Exception:
            j = None
        return r.status_code, j, latency
    except Exception as e:
        latency = (time.perf_counter() - t0) * 1000
        return 0, {"error": str(e)}, latency


async def _rate_gate(state: dict) -> None:
    rate = state.get("rate", 0)
    if rate <= 0:
        return
    async with state["lock"]:
        now = time.perf_counter()
        gap = now - state.get("last", 0.0)
        interval = 1.0 / rate
        if gap < interval:
            await asyncio.sleep(interval - gap)
        state["last"] = time.perf_counter()


async def run_arm_intent(workload, args, sem, rate_state) -> list[RequestRecord]:
    """Arm A — intent-based: keyword classifier → backend URL.
    Goes direct to the backend URLs; does not consult the router."""
    out: list[RequestRecord] = []
    async with httpx.AsyncClient() as client:
        async def one(req: GeneratedRequest):
            async with sem:
                await _rate_gate(rate_state)
                intent = classify_intent(req.body["messages"])
                lane = INTENT_TO_BACKEND[intent]
                url = args.fast if lane == "fast" else args.quality
                endpoint = "vllm-fast" if lane == "fast" else "vllm-quality"
                status, j, lat = await send_request(client, url, "qwen", req.body)
                ok = status == 200
                tokens_in = (j or {}).get("usage", {}).get("prompt_tokens", 0) if j else 0
                tokens_out = (j or {}).get("usage", {}).get("completion_tokens", 0) if j else 0
                out.append(RequestRecord(
                    arm="intent",
                    tenant=req.tenant.name,
                    shape=req.shape["name"],
                    slo_ms=req.shape["slo_ms"],
                    endpoint_used=endpoint,
                    ok=ok, status=status, latency_ms=lat,
                    tokens_in=tokens_in, tokens_out=tokens_out,
                    cost_usd=cost_for(endpoint, lat),
                ))
        await asyncio.gather(*(one(r) for r in workload))
    return out


async def run_arm_router(workload, args, sem, rate_state, policy_swap_at: float) -> list[RequestRecord]:
    """Arm B — TokenFlow router: send through router with tenant + priority headers.
    Mid-run, swap from `balanced` to `cost-first` to demonstrate live policy
    update."""
    out: list[RequestRecord] = []
    arm_start = time.perf_counter()
    swap_done = {"flag": False}
    async with httpx.AsyncClient() as client:
        async def one(req: GeneratedRequest):
            async with sem:
                await _rate_gate(rate_state)

                # Mid-run policy swap (only happens once)
                elapsed = time.perf_counter() - arm_start
                if not swap_done["flag"] and elapsed >= policy_swap_at:
                    swap_done["flag"] = True
                    try:
                        await client.post(
                            f"{args.router}/admin/policy/preset",
                            json={"preset": "cost-first"},
                            timeout=5.0,
                        )
                        print(f"  [swap] live preset → cost-first @ T+{elapsed:.0f}s",
                              flush=True)
                    except Exception:
                        pass

                phase = "post-swap" if swap_done["flag"] else "pre-swap"
                headers = {
                    "x-tenant-id": f"tenant-{req.tenant.name}",
                    "x-priority-tier": req.tenant.priority_tier,
                }
                status, j, lat = await send_request(
                    client, args.router, "qwen", req.body, headers,
                )
                ok = status == 200
                endpoint = "unknown"
                tokens_in = tokens_out = 0
                if j:
                    tf = j.get("_tokenflow", {})
                    endpoint = (tf.get("endpoint") or tf.get("endpoint_name") or "unknown")
                    if "usage" in j:
                        tokens_in = j["usage"].get("prompt_tokens", 0) or 0
                        tokens_out = j["usage"].get("completion_tokens", 0) or 0
                out.append(RequestRecord(
                    arm="tokenflow",
                    tenant=req.tenant.name,
                    shape=req.shape["name"],
                    slo_ms=req.shape["slo_ms"],
                    endpoint_used=endpoint,
                    ok=ok, status=status, latency_ms=lat,
                    tokens_in=tokens_in, tokens_out=tokens_out,
                    cost_usd=cost_for(endpoint, lat),
                    phase=phase,
                ))
        await asyncio.gather(*(one(r) for r in workload))
    return out


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------


def percentile(xs: list[float], q: float) -> float:
    if not xs:
        return 0.0
    xs = sorted(xs)
    k = max(0, min(len(xs) - 1, int(round((q / 100.0) * (len(xs) - 1)))))
    return xs[k]


def summarise_by(records: list[RequestRecord], by: tuple[str, ...]) -> list[dict]:
    """Group by a tuple of attribute names; emit aggregated stats per group."""
    groups: dict[tuple, list[RequestRecord]] = {}
    for r in records:
        key = tuple(getattr(r, k) for k in by)
        groups.setdefault(key, []).append(r)
    rows = []
    for key, rs in sorted(groups.items()):
        ok = [r for r in rs if r.ok]
        lats = [r.latency_ms for r in ok]
        tot_cost = sum(r.cost_usd for r in ok)
        tot_tok_out = sum(r.tokens_out for r in ok)
        slo_miss = sum(1 for r in ok if r.latency_ms > r.slo_ms)
        endpoint_dist: dict[str, int] = {}
        for r in ok:
            endpoint_dist[r.endpoint_used] = endpoint_dist.get(r.endpoint_used, 0) + 1
        row = {k: v for k, v in zip(by, key)}
        row.update({
            "requests":              len(rs),
            "success":               len(ok),
            "failed":                len(rs) - len(ok),
            "success_pct":           round(100.0 * len(ok) / len(rs), 1) if rs else 0.0,
            "p50_ms":                round(percentile(lats, 50), 1),
            "p95_ms":                round(percentile(lats, 95), 1),
            "p99_ms":                round(percentile(lats, 99), 1),
            "mean_ms":               round(statistics.mean(lats), 1) if lats else 0.0,
            "slo_miss":              slo_miss,
            "slo_miss_pct":          round(100.0 * slo_miss / max(1, len(ok)), 1),
            "total_cost_usd":        round(tot_cost, 6),
            "cost_per_1k_tok_usd":   round(1000 * tot_cost / max(1, tot_tok_out), 6),
            "endpoint_distribution": endpoint_dist,
        })
        rows.append(row)
    return rows


def print_table(rows: list[dict], title: str) -> None:
    print()
    print(f"=== {title} ===")
    if not rows:
        print("(no rows)")
        return
    cols = ["arm", "tenant", "requests", "success_pct", "p50_ms", "p95_ms",
            "p99_ms", "slo_miss_pct", "total_cost_usd"]
    cols = [c for c in cols if c in rows[0]]
    widths = [max(len(c), max(len(str(r.get(c, ""))) for r in rows)) + 1 for c in cols]
    line = " | ".join(f"{c:<{w}}" for c, w in zip(cols, widths))
    print(line)
    print("-" * len(line))
    for r in rows:
        print(" | ".join(f"{str(r.get(c, '')):<{w}}" for c, w in zip(cols, widths)))


# ---------------------------------------------------------------------------
# Pre-flight
# ---------------------------------------------------------------------------


async def preflight(args) -> None:
    print("[probe]")
    async with httpx.AsyncClient(timeout=5.0) as c:
        for label, url, required in [
            ("router", args.router, True),
            ("fast",   args.fast,   True),
            ("quality", args.quality, True),
        ]:
            try:
                r = await c.get(f"{url}/health")
                print(f"  {label} ({url}): HTTP {r.status_code}")
            except Exception as e:
                print(f"  {label} ({url}): FAIL — {e}", file=sys.stderr)
                if required:
                    sys.exit(1)

        # Reset router policy to balanced before run starts
        try:
            await c.post(
                f"{args.router}/admin/policy/preset",
                json={"preset": "balanced"}, timeout=3.0,
            )
            print("  reset router preset → balanced")
        except Exception as e:
            print(f"  preset reset failed: {e}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


async def main_async(args) -> None:
    rate_state = {"rate": args.rate, "lock": asyncio.Lock(), "last": 0.0}
    sem = asyncio.Semaphore(args.concurrency)

    workload = build_workload(args.n, args.seed)
    print(f"workload: {args.n} requests, seed={args.seed}, "
          f"rate={args.rate}/s, concurrency={args.concurrency}, "
          f"~{args.n / max(args.rate, 1e-6):.0f}s per arm")
    print(f"tenant mix: free={int(0.45*100)}% standard={int(0.40*100)}% enterprise={int(0.15*100)}%")
    print(f"workload mix: " + " ".join(f"{s['name']}={int(s['weight']*100)}%" for s in WORKLOAD_SHAPES))

    await preflight(args)

    print()
    print(f"=== arm A: intent-based ({args.n} requests) ===")
    t0 = time.perf_counter()
    intent_results = await run_arm_intent(workload, args, sem, rate_state)
    intent_wall = time.perf_counter() - t0
    print(f"  arm A done in {intent_wall:.0f}s")

    # reset policy + rate state for arm B
    rate_state["last"] = 0.0
    await preflight(args)  # also resets the policy

    print()
    swap_at = args.policy_swap_at if args.policy_swap_at > 0 else (args.n / max(args.rate, 1e-6)) / 2
    print(f"=== arm B: TokenFlow router ({args.n} requests, policy swap at T+{swap_at:.0f}s) ===")
    t0 = time.perf_counter()
    router_results = await run_arm_router(workload, args, sem, rate_state, swap_at)
    router_wall = time.perf_counter() - t0
    print(f"  arm B done in {router_wall:.0f}s")

    # ── Summarise ──────────────────────────────────────────────────────
    all_records = intent_results + router_results
    by_arm = summarise_by(all_records, by=("arm",))
    by_arm_tenant = summarise_by(all_records, by=("arm", "tenant"))
    by_arm_shape = summarise_by(all_records, by=("arm", "shape"))
    by_arm_phase = summarise_by(
        [r for r in router_results if r.arm == "tokenflow"],
        by=("arm", "phase"),
    )

    print_table(by_arm,         "TOTALS by arm")
    print_table(by_arm_tenant,  "by arm × tenant")
    print_table(by_arm_shape,   "by arm × shape")
    print_table(by_arm_phase,   "TokenFlow by phase (pre/post live policy swap)")

    # ── Persist ────────────────────────────────────────────────────────
    out = {
        "scenario": "saas-multi-tenant",
        "workload_size": args.n,
        "seed": args.seed,
        "concurrency": args.concurrency,
        "rate_rps": args.rate,
        "policy_swap_at_s": swap_at,
        "wall_s": {"intent": round(intent_wall, 1), "tokenflow": round(router_wall, 1)},
        "totals": by_arm,
        "by_tenant": by_arm_tenant,
        "by_shape": by_arm_shape,
        "tokenflow_by_phase": by_arm_phase,
        "raw": [asdict(r) for r in all_records],
    }
    out_path = args.out
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\n[saved] {out_path}")


def main():
    ap = argparse.ArgumentParser(description="TokenFlow production-scenario benchmark")
    ap.add_argument("--router",  default="http://localhost:8080")
    ap.add_argument("--fast",    default="http://localhost:8001",
                    help="economy backend (Qwen2.5-3B, $2.50/GPU-hr)")
    ap.add_argument("--quality", default="http://localhost:8002",
                    help="premium backend (Qwen2.5-7B, $8.00/GPU-hr)")
    ap.add_argument("--n", type=int, default=600, help="requests per arm")
    ap.add_argument("--rate", type=float, default=2.0,
                    help="rate limit (req/s globally) — controls run length")
    ap.add_argument("--concurrency", type=int, default=8)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--policy-swap-at", type=float, default=0.0,
                    help="seconds into arm B to swap preset; 0 = midpoint")
    ap.add_argument("--out", default="examples/production_demo/results/benchmark.json")
    args = ap.parse_args()
    asyncio.run(main_async(args))


if __name__ == "__main__":
    main()
