"""TokenFlow Router — telco / multi-workload production benchmark.

Models a real enterprise inference platform with six concurrent workloads
each carrying its own SLO, cost ceiling, and routing policy. Compares
TokenFlow's policy-aware routing against an intent-based baseline on the
*same* fleet, *same* workload, *same* seed.

Workloads exercised
===================

  customer_care_voice       voice-agent style turn (low-latency premium)
  rag_retrieval             RAG re-ranker + final answer (standard, balanced)
  esg_batch                 long-form ESG report classification (batch, cost-first)
  ai_assisted_migration     code-translation / refactor (standard, reasoning bias)
  trust_inventory           structured-query intent classification (standard)
  digital_twin_simulation   long-context analytical reasoning (premium)

Each workload is a *tenant* with its own:
  - x-tenant-id / x-priority-tier headers
  - budget cap and GPU allowlist (TokenFlow only — intent ignores)
  - representative request shape (input/output tokens)

The harness runs both arms sequentially, aggregates per-(arm, workload)
and per-arm totals, and writes the JSON to results/benchmark.json.
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
from typing import Any, Optional

import httpx


# ---------------------------------------------------------------------------
# Workloads — each one is a tenant + a request-shape generator
# ---------------------------------------------------------------------------


@dataclass
class Workload:
    name: str                         # e.g. "customer_care_voice"
    weight: float                     # share of total traffic
    tenant: str                       # x-tenant-id header
    priority_tier: str                # standard / batch / premium
    slo_ms: float                     # request-level SLO target
    body_factory: Any                 # callable() -> dict request body


_LONG_DOC = " ".join([
    "The Energy Performance Certificate Programme dataset for FY2025 includes 18,432 properties across nine regions.",
    "Carbon intensity reduction targets average 24% with regional variance from 18% to 31%.",
    "Embedded carbon costs in the supply chain are estimated at 41% of total scope-3 emissions.",
    "Migration of legacy workloads to renewable-energy-backed datacentres targets a 38% scope-2 reduction by FY2027.",
    "The carbon-disclosure framework follows TCFD with extended quantitative disclosures aligned to ISSB S2.",
    "Quarterly board reviews now include carbon-budget-vs-actual ratios alongside financial ratios.",
    "Network-equipment lifecycle assumptions use a 7-year useful-life model with 92% recyclable mass at end-of-life.",
    "Renewable-energy PPAs cover 64% of contracted demand; spot-market exposure remains 12% of total energy spend.",
    "ESG-aligned capex is reviewed against a $0.18/kgCO2e shadow price for project NPV calculations.",
]) * 12  # ~5,500 tokens — approximates a real ESG batch document


WORKLOADS: list[Workload] = [
    Workload(
        name="customer_care_voice",
        weight=0.30,
        tenant="customer-care",
        priority_tier="premium",
        slo_ms=1500,
        body_factory=lambda: {
            "messages": [{"role": "user", "content": random.choice([
                "I want to upgrade my plan to include international roaming for next month's trip.",
                "My bill seems higher than last month — can you check what changed?",
                "How do I activate the new SIM I just received?",
                "My internet has been slow this week. What's going on?",
                "I'd like to add a second line for my partner.",
            ])}],
            "max_tokens": 96,
        },
    ),
    Workload(
        name="rag_retrieval",
        weight=0.25,
        tenant="rag-platform",
        priority_tier="standard",
        slo_ms=4000,
        body_factory=lambda: {
            "messages": [{"role": "user", "content":
                "Given the retrieved context, answer the question: " +
                "What is the standard escalation path for a tier-3 service incident? " +
                "Context: " + " ".join(_LONG_DOC.split()[:300])
            }],
            "max_tokens": 256,
        },
    ),
    Workload(
        name="esg_batch",
        weight=0.10,
        tenant="esg-reporting",
        priority_tier="batch",
        slo_ms=15000,
        body_factory=lambda: {
            "messages": [{"role": "user", "content":
                "Extract the three most material ESG risks from the following report and "
                "rank them by financial impact:\n\n" + _LONG_DOC
            }],
            "max_tokens": 200,
        },
    ),
    Workload(
        name="ai_assisted_migration",
        weight=0.15,
        tenant="migration-tools",
        priority_tier="standard",
        slo_ms=8000,
        body_factory=lambda: {
            "messages": [
                {"role": "system", "content": "You are an expert code migration assistant. Think carefully before answering."},
                {"role": "user", "content": random.choice([
                    "Translate this COBOL FILLER PIC X(42) to a Python dataclass with the same byte layout. Preserve trailing-space semantics.",
                    "Rewrite this Java 8 stream pipeline as a Kotlin Flow with structured concurrency, preserving the back-pressure semantics.",
                    "Migrate this Spring 4.x XML bean wiring to Spring Boot 3 annotations, including the AOP advisor chain.",
                ])},
            ],
            "max_tokens": 400,
        },
    ),
    Workload(
        name="trust_inventory",
        weight=0.15,
        tenant="trust-inventory",
        priority_tier="standard",
        slo_ms=3000,
        body_factory=lambda: {
            "messages": [{"role": "user", "content":
                "Classify the following inventory event as one of: NEW_ASSET, MODIFIED, RETIRED, ANOMALY. "
                "Event: " + random.choice([
                    "router-fw-update applied to edge-pop-syd-04, version 8.2.1 → 8.2.3",
                    "new switch sku CRS-356-MEL provisioned in slot 17 of cabinet C-32",
                    "core-router cr-mel-02 retired after 11 years, decommissioned per change CHG-44219",
                    "unexpected MAC address learned on vlan 2034 — investigate",
                ])
            }],
            "max_tokens": 32,
        },
    ),
    Workload(
        name="digital_twin_simulation",
        weight=0.05,
        tenant="digital-twin",
        priority_tier="premium",
        slo_ms=12000,
        body_factory=lambda: {
            "messages": [
                {"role": "system", "content": "Reason step by step about the simulation parameters."},
                {"role": "user", "content":
                    "A network of 412 core nodes is exposed to a 0.3% per-day failure rate. With an MTTR of 4 hours and "
                    "redundancy=2 across geographically separate paths, what is the expected service-availability over a "
                    "30-day window? Show the derivation, including treatment of correlated failure modes (e.g., regional "
                    "outage covering multiple nodes simultaneously)."
                },
            ],
            "max_tokens": 512,
        },
    ),
]


def pick_workload(rng: random.Random) -> Workload:
    r = rng.random()
    acc = 0.0
    for w in WORKLOADS:
        acc += w.weight
        if r <= acc:
            return w
    return WORKLOADS[-1]


# ---------------------------------------------------------------------------
# Backends — three local lanes with explicit cost asymmetry
# ---------------------------------------------------------------------------


COST_PER_GPU_HOUR = {
    "vllm-economy":  2.50,    # Qwen 3B on 1 A100  — chat, batch
    "vllm-standard": 5.00,    # Qwen 14B on 1 A100 — RAG, classification
    "vllm-premium":  12.00,   # Qwen 72B on 2 A100 — reasoning, voice, digital twin
    "unknown":       0.0,
}


def cost_for(endpoint: str, latency_ms: float) -> float:
    return COST_PER_GPU_HOUR.get(endpoint, 0.0) * latency_ms / 1000.0 / 3600.0


# ---------------------------------------------------------------------------
# Intent-based router (Arm A) — keyword-based baseline
# ---------------------------------------------------------------------------


def classify_intent(messages: list[dict]) -> str:
    """Naive keyword classifier — what most teams ship as a 'smart router'."""
    text = " ".join(m.get("content", "") for m in messages).lower()
    if any(p in text for p in ("step by step", "derive", "reasoning", "show the derivation")):
        return "reasoning"
    if any(p in text for p in ("translate this", "rewrite this", "migrate this", "refactor", "code")):
        return "code"
    if any(p in text for p in ("classify the following", "extract the", "which of", "categorize")):
        return "classification"
    if any(p in text for p in ("given the retrieved context", "summarise", "summarize", "tldr")):
        return "rag"
    if any(p in text for p in ("upgrade my plan", "my bill", "my account", "internet has been")):
        return "voice"
    return "chat"


# Intent → backend. "Hard" intents go premium; others split between standard/economy.
INTENT_TO_BACKEND = {
    "reasoning":      "premium",
    "code":           "premium",
    "voice":          "premium",
    "rag":            "standard",
    "classification": "standard",
    "chat":           "economy",
}


# ---------------------------------------------------------------------------
# Result record
# ---------------------------------------------------------------------------


@dataclass
class RequestRecord:
    arm: str
    workload: str
    tenant: str
    priority_tier: str
    slo_ms: float
    endpoint_used: str
    ok: bool
    status: int
    latency_ms: float
    tokens_in: int
    tokens_out: int
    cost_usd: float
    intent_label: str = ""              # only set for arm A


# ---------------------------------------------------------------------------
# HTTP arms
# ---------------------------------------------------------------------------


async def _send(client, url, model, body, headers=None):
    t0 = time.perf_counter()
    try:
        r = await client.post(
            f"{url}/v1/chat/completions",
            json={"model": model, **body},
            headers=headers or {},
            timeout=90.0,
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


async def _rate_gate(state):
    if state["rate"] <= 0:
        return
    async with state["lock"]:
        now = time.perf_counter()
        gap = now - state["last"]
        ival = 1.0 / state["rate"]
        if gap < ival:
            await asyncio.sleep(ival - gap)
        state["last"] = time.perf_counter()


async def run_arm_intent(plan, args, sem, rate_state) -> list[RequestRecord]:
    """Arm A — intent-based: keyword classifier → premium / standard / economy."""
    out: list[RequestRecord] = []
    backend_url = {
        "economy":  args.economy,
        "standard": args.standard,
        "premium":  args.premium,
    }
    backend_name = {
        "economy":  "vllm-economy",
        "standard": "vllm-standard",
        "premium":  "vllm-premium",
    }
    async with httpx.AsyncClient() as client:
        async def one(item):
            wl: Workload = item["workload"]
            body = item["body"]
            async with sem:
                await _rate_gate(rate_state)
                intent = classify_intent(body["messages"])
                lane = INTENT_TO_BACKEND[intent]
                url = backend_url[lane]
                ep = backend_name[lane]
                status, j, lat = await _send(client, url, "qwen", body)
                ok = status == 200
                tin = (j or {}).get("usage", {}).get("prompt_tokens", 0) if j else 0
                tout = (j or {}).get("usage", {}).get("completion_tokens", 0) if j else 0
                out.append(RequestRecord(
                    arm="intent",
                    workload=wl.name,
                    tenant=wl.tenant,
                    priority_tier=wl.priority_tier,
                    slo_ms=wl.slo_ms,
                    endpoint_used=ep,
                    ok=ok, status=status, latency_ms=lat,
                    tokens_in=tin, tokens_out=tout,
                    cost_usd=cost_for(ep, lat),
                    intent_label=intent,
                ))
        await asyncio.gather(*(one(item) for item in plan))
    return out


async def run_arm_router(plan, args, sem, rate_state) -> list[RequestRecord]:
    """Arm B — TokenFlow router with workload-tagged tenant headers."""
    out: list[RequestRecord] = []
    async with httpx.AsyncClient() as client:
        async def one(item):
            wl: Workload = item["workload"]
            body = item["body"]
            async with sem:
                await _rate_gate(rate_state)
                headers = {
                    "x-tenant-id":     f"tenant-{wl.tenant}",
                    "x-priority-tier": wl.priority_tier,
                }
                status, j, lat = await _send(
                    client, args.router, "qwen", body, headers,
                )
                ok = status == 200
                ep = "unknown"
                tin = tout = 0
                if j:
                    tf = j.get("_tokenflow", {})
                    ep = tf.get("endpoint") or tf.get("endpoint_name") or "unknown"
                    if "usage" in j:
                        tin  = j["usage"].get("prompt_tokens", 0) or 0
                        tout = j["usage"].get("completion_tokens", 0) or 0
                out.append(RequestRecord(
                    arm="tokenflow",
                    workload=wl.name,
                    tenant=wl.tenant,
                    priority_tier=wl.priority_tier,
                    slo_ms=wl.slo_ms,
                    endpoint_used=ep,
                    ok=ok, status=status, latency_ms=lat,
                    tokens_in=tin, tokens_out=tout,
                    cost_usd=cost_for(ep, lat),
                ))
        await asyncio.gather(*(one(item) for item in plan))
    return out


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------


def percentile(xs, q):
    if not xs:
        return 0.0
    xs = sorted(xs)
    k = max(0, min(len(xs) - 1, int(round((q / 100.0) * (len(xs) - 1)))))
    return xs[k]


def summarise(records: list[RequestRecord], by: tuple[str, ...]) -> list[dict]:
    groups: dict[tuple, list[RequestRecord]] = {}
    for r in records:
        key = tuple(getattr(r, k) for k in by)
        groups.setdefault(key, []).append(r)
    rows = []
    for key, rs in sorted(groups.items()):
        ok = [r for r in rs if r.ok]
        lats = [r.latency_ms for r in ok]
        slo_miss = sum(1 for r in ok if r.latency_ms > r.slo_ms)
        cost = sum(r.cost_usd for r in ok)
        tok = sum(r.tokens_out for r in ok)
        ep_dist: dict[str, int] = {}
        for r in ok:
            ep_dist[r.endpoint_used] = ep_dist.get(r.endpoint_used, 0) + 1
        row = {k: v for k, v in zip(by, key)}
        row.update({
            "requests":            len(rs),
            "success":             len(ok),
            "failed":              len(rs) - len(ok),
            "success_pct":         round(100.0 * len(ok) / max(1, len(rs)), 1),
            "p50_ms":              round(percentile(lats, 50), 1),
            "p95_ms":              round(percentile(lats, 95), 1),
            "p99_ms":              round(percentile(lats, 99), 1),
            "mean_ms":             round(statistics.mean(lats), 1) if lats else 0.0,
            "slo_miss_pct":        round(100.0 * slo_miss / max(1, len(ok)), 1),
            "total_cost_usd":      round(cost, 6),
            "cost_per_1k_tok_usd": round(1000 * cost / max(1, tok), 6),
            "endpoint_distribution": ep_dist,
        })
        rows.append(row)
    return rows


def print_table(rows, title):
    print(f"\n=== {title} ===")
    if not rows:
        print("(no rows)")
        return
    cols = [c for c in (
        "arm", "workload", "tenant", "requests", "success_pct",
        "p50_ms", "p95_ms", "p99_ms", "slo_miss_pct", "total_cost_usd",
    ) if c in rows[0]]
    widths = [max(len(c), max(len(str(r.get(c, ""))) for r in rows)) + 1 for c in cols]
    print(" | ".join(f"{c:<{w}}" for c, w in zip(cols, widths)))
    print("-" * (sum(widths) + 3 * len(cols)))
    for r in rows:
        print(" | ".join(f"{str(r.get(c, '')):<{w}}" for c, w in zip(cols, widths)))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def build_plan(n: int, seed: int) -> list[dict]:
    rng = random.Random(seed)
    out = []
    for i in range(n):
        wl = pick_workload(rng)
        out.append({"idx": i, "workload": wl, "body": wl.body_factory()})
    return out


async def preflight(args):
    print("[probe]")
    async with httpx.AsyncClient(timeout=5.0) as c:
        for label, url, required in [
            ("router",   args.router,   True),
            ("economy",  args.economy,  True),
            ("standard", args.standard, True),
            ("premium",  args.premium,  True),
        ]:
            try:
                r = await c.get(f"{url}/health")
                print(f"  {label} ({url}): HTTP {r.status_code}")
            except Exception as exc:
                print(f"  {label} ({url}): FAIL — {exc}", file=sys.stderr)
                if required:
                    sys.exit(1)
        # reset router preset
        try:
            await c.post(f"{args.router}/admin/policy/preset",
                         json={"preset": "balanced"}, timeout=3.0)
            print("  reset router preset → balanced")
        except Exception:
            pass


async def main_async(args):
    rate_state = {"rate": args.rate, "lock": asyncio.Lock(), "last": 0.0}
    sem = asyncio.Semaphore(args.concurrency)
    plan = build_plan(args.n, args.seed)

    print(f"workload: {args.n} requests, seed={args.seed}, "
          f"rate={args.rate}/s, concurrency={args.concurrency}, "
          f"~{args.n / max(args.rate, 1e-6):.0f}s per arm")
    print(f"workloads: " + ", ".join(f"{w.name}={int(w.weight*100)}%" for w in WORKLOADS))

    await preflight(args)

    print(f"\n=== arm A: intent-based ({args.n} requests) ===")
    t0 = time.perf_counter()
    intent_results = await run_arm_intent(plan, args, sem, rate_state)
    intent_wall = time.perf_counter() - t0
    print(f"  arm A done in {intent_wall:.0f}s")

    rate_state["last"] = 0.0
    await preflight(args)

    print(f"\n=== arm B: TokenFlow router ({args.n} requests) ===")
    t0 = time.perf_counter()
    router_results = await run_arm_router(plan, args, sem, rate_state)
    router_wall = time.perf_counter() - t0
    print(f"  arm B done in {router_wall:.0f}s")

    all_records = intent_results + router_results
    by_arm          = summarise(all_records, by=("arm",))
    by_arm_workload = summarise(all_records, by=("arm", "workload"))

    print_table(by_arm,          "TOTALS by arm")
    print_table(by_arm_workload, "by arm × workload")

    out = {
        "scenario": "telco-multi-workload",
        "workload_size": args.n,
        "seed": args.seed,
        "concurrency": args.concurrency,
        "rate_rps": args.rate,
        "wall_s": {"intent": round(intent_wall, 1), "tokenflow": round(router_wall, 1)},
        "totals":      by_arm,
        "by_workload": by_arm_workload,
        "raw":         [asdict(r) for r in all_records],
    }
    with open(args.out, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\n[saved] {args.out}")


def main():
    ap = argparse.ArgumentParser(description="Telco-shape multi-workload benchmark")
    ap.add_argument("--router",   default="http://localhost:8080")
    ap.add_argument("--economy",  default="http://localhost:8001",
                    help="economy lane (Qwen 3B / 1× A100, $2.50/GPU-hr)")
    ap.add_argument("--standard", default="http://localhost:8002",
                    help="standard lane (Qwen 14B / 1× A100, $5/GPU-hr)")
    ap.add_argument("--premium",  default="http://localhost:8003",
                    help="premium lane (Qwen 72B / 2× A100, $12/GPU-hr)")
    ap.add_argument("--n", type=int, default=600)
    ap.add_argument("--rate", type=float, default=2.0)
    ap.add_argument("--concurrency", type=int, default=8)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default="examples/telco_demo/results/benchmark.json")
    args = ap.parse_args()
    asyncio.run(main_async(args))


if __name__ == "__main__":
    main()
