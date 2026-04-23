#!/usr/bin/env python3
"""Three-arm benchmark for TokenFlow Router.

Arms
----
  A (direct):       every request → vllm-fast directly
  B (round-robin):  alternate between vllm-fast and vllm-quality per request
  C (router):       route through TokenFlow with production-balanced policy

All three arms receive the *identical* request stream (same seed, same order).
We measure per-request TTFT (time to first byte of the response body, which
for non-streaming vLLM is effectively E2E minus network fluff), E2E latency,
and success. At the end we print a comparison table covering p50/p95 latency,
throughput, success rate, estimated cost, and SLO-miss rate.

The SLO target is configurable (default: 5000 ms E2E). Cost is estimated from
the registered cost_per_gpu_hour of whichever backend each request used,
amortised over the request's wall time — this mirrors what the router's
`_tokenflow.estimated_cost_usd` field reports.

Notes on fairness
-----------------
- Requests are sent concurrently (configurable `--concurrency`) so queueing
  behavior surfaces. With concurrency=1 you mostly measure single-request
  speed; with concurrency>1 you measure how well the arm exploits two lanes.
- Arm C gets the benefit of the router's policy; arms A and B do not. That
  is the point of the comparison.
- The workload mix is held constant across arms.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import random
import statistics
import sys
import time
from dataclasses import dataclass, field
from typing import Any

import httpx


# ------- workload -------
# A realistic mix: short chat, reasoning, prefill-heavy, decode-heavy.
_LONG_DOC = (" ".join([
    "The internet protocol suite, commonly known as TCP/IP, is the foundational set of communications protocols used for the Internet and most private networks.",
    "The Transmission Control Protocol provides reliable ordered byte streams; the User Datagram Protocol provides low-overhead datagrams; the Internet Protocol delivers addressed packets between hosts.",
    "The protocol stack was specified across several RFCs starting in the 1970s; the modern layering model is link, internet, transport, and application.",
    "Early deployments on ARPANET in the 1970s informed later refinement; DNS, BGP, and TLS built on top of the core specifications during the 1980s and 1990s.",
    "Routing on the internet uses the Border Gateway Protocol to exchange reachability information between autonomous systems, while interior gateway protocols such as OSPF and IS-IS handle routes within a single administrative domain.",
    "Congestion control algorithms like TCP Reno, Cubic, and BBR shape sender behavior to avoid collapse under load.",
    "The transition from IPv4 to IPv6 has progressed slowly, driven by address exhaustion in IPv4 and the need for larger address space.",
    "Modern application protocols including HTTP/2, HTTP/3, and gRPC multiplex many logical streams over a single transport connection.",
    "Security layers built on top of TCP include TLS 1.2 and TLS 1.3; QUIC integrates transport and crypto to reduce handshake latency.",
    "End-to-end measurement studies consistently show that application-level latency is dominated by round-trip time on long-haul links, motivating edge computing and CDN deployments.",
]) * 40)  # ~5500 tokens — exceeds vllm-fast's 4096 but fits vllm-quality's 16384


WORKLOAD_MIX: list[dict[str, Any]] = [
    {
        "shape": "short_chat",
        "weight": 0.40,
        "slo_ms": 3000,
        "gen": lambda: {
            "messages": [{"role": "user", "content": random.choice([
                "What is the capital of France?",
                "Convert 72 Fahrenheit to Celsius.",
                "What year did the Berlin Wall fall?",
                "What's the chemical symbol for tungsten?",
                "How many planets are in the solar system?",
            ])}],
            "max_tokens": 32,
        },
    },
    {
        "shape": "reasoning",
        "weight": 0.20,
        "slo_ms": 12000,
        "gen": lambda: {
            "messages": [
                {"role": "system", "content": "Think step by step."},
                {"role": "user", "content": random.choice([
                    "If a train leaves station A at 2pm going 60mph and another leaves station B at 3pm going 75mph toward A, and the stations are 300 miles apart, when do they meet?",
                    "A bag has 5 red and 3 blue marbles. If I draw 2 without replacement, what is the probability both are red?",
                    "I have 3 coins totaling 30 cents and one is not a nickel. What are they?",
                ])},
            ],
            "max_tokens": 300,
        },
    },
    {
        "shape": "long_context",
        "weight": 0.20,
        "slo_ms": 10000,
        "gen": lambda: {
            "messages": [{"role": "user", "content": (
                "Extract 3 factual bullets from the following technical document:\n\n" + _LONG_DOC
            )}],
            "max_tokens": 120,
        },
    },
    {
        "shape": "prefill_heavy",
        "weight": 0.15,
        "slo_ms": 8000,
        "gen": lambda: {
            "messages": [{"role": "user", "content": (
                "Summarise the following text in exactly 2 sentences:\n\n"
                + (" ".join([
                    "The internet protocol suite, commonly known as TCP/IP, is the set of communications protocols used for the Internet and similar computer networks.",
                    "The foundational protocols are the Transmission Control Protocol (TCP), the User Datagram Protocol (UDP), and the Internet Protocol (IP).",
                    "Early versions were developed in the 1970s by DARPA.",
                    "The protocol suite provides end-to-end data communication specifying how data should be packetised, addressed, transmitted, routed, and received.",
                    "The layers are: the link layer, the internet layer, the transport layer, and the application layer.",
                ]) * 10)
            )}],
            "max_tokens": 80,
        },
    },
    {
        "shape": "decode_heavy",
        "weight": 0.05,
        "slo_ms": 15000,
        "gen": lambda: {
            "messages": [{"role": "user", "content": random.choice([
                "Write a 400-word short story about a lighthouse keeper who befriends a crow.",
                "Explain in detail, with examples, how a B-tree index works in a relational database. Aim for 400 words.",
                "Write 400 words of creative exposition about the history of cartography.",
            ])}],
            "max_tokens": 500,
        },
    },
]


def pick_shape() -> dict[str, Any]:
    r = random.random()
    acc = 0.0
    for s in WORKLOAD_MIX:
        acc += s["weight"]
        if r <= acc:
            return s
    return WORKLOAD_MIX[-1]


def build_workload(n: int, seed: int, workload_noise: float = 0.0) -> list[dict[str, Any]]:
    """Build a seeded workload stream. `workload_noise` in [0,1] specifies
    the fraction of requests whose `max_tokens` is randomised to a value
    inconsistent with the actual output length — this is how we simulate
    clients sending misleading `max_tokens` ceilings in production, which
    breaks TokenFlow's `predicted_output_tokens` → `workload_type` inference.
    The noise does NOT affect the prompt text, so intent-based classification
    is unaffected by workload noise (and vice versa)."""
    random.seed(seed)
    noise_rng = random.Random(seed + 1)
    out = []
    for i in range(n):
        shape = pick_shape()
        body = shape["gen"]()
        if noise_rng.random() < workload_noise:
            # Replace max_tokens with a random value from across all shapes.
            # This makes TokenFlow's workload_type classifier misfire ~workload_noise fraction.
            body["max_tokens"] = noise_rng.choice([16, 64, 128, 300, 500, 1024, 1500])
        out.append({
            "idx": i,
            "shape": shape["shape"],
            "slo_ms": shape["slo_ms"],
            "body": body,
        })
    return out


# Intent noise — randomly replaces classifier output to simulate real ML
# classifier mis-classification (distilBERT/LLM-as-judge in prod typically
# has 5-15% error rate).
_INTENT_NOISE = {"fraction": 0.0, "rng": random.Random(42)}

def _maybe_noisy_intent(intent: str) -> str:
    if _INTENT_NOISE["rng"].random() < _INTENT_NOISE["fraction"]:
        return _INTENT_NOISE["rng"].choice(
            ["reasoning", "summarization", "generation", "chat"]
        )
    return intent


# ------- cost model -------
# Matches registration metadata (see deploy script).
COST_PER_GPU_HOUR = {
    # v4 layout — same 7B model, different configs; all on same H100 class
    "vllm-decode":  4.0,   # short-ctx decode-tuned
    "vllm-prefill": 4.0,   # long-ctx prefill-tuned
    "vllm-spec":    4.0,   # single 7B with n-gram speculative decoding
    # kept for backward compatibility with v3 result files
    "vllm-fast":    2.5,
    "vllm-quality": 8.0,
}


def cost_for(endpoint_name: str, latency_ms: float) -> float:
    """Amortise cost over the wall time of this request."""
    rate = COST_PER_GPU_HOUR.get(endpoint_name, 0.0)
    return rate * (latency_ms / 1000.0) / 3600.0


# ------- arm runners -------
@dataclass
class Result:
    idx: int
    shape: str
    slo_ms: float
    endpoint_used: str
    ok: bool
    latency_ms: float
    error: str = ""
    cost_usd: float = 0.0
    tenant_charge_usd: float = 0.0
    tokens_out: int = 0


async def call_openai(
    client: httpx.AsyncClient, url: str, model: str, body: dict[str, Any],
    headers: dict[str, str] | None = None,
) -> tuple[int, dict[str, Any] | None, float]:
    t0 = time.perf_counter()
    try:
        r = await client.post(
            f"{url}/v1/chat/completions",
            json={"model": model, **body},
            headers=headers or {},
            timeout=60.0,
        )
        latency_ms = (time.perf_counter() - t0) * 1000
        try:
            j = r.json()
        except Exception:
            j = None
        return r.status_code, j, latency_ms
    except Exception as e:
        latency_ms = (time.perf_counter() - t0) * 1000
        return 0, {"error": str(e)}, latency_ms


async def arm_direct(
    workload: list[dict[str, Any]], args, sem: asyncio.Semaphore
) -> list[Result]:
    """Arm A: every request → vllm-decode only (single backend, low-latency
    config). Fails long-context requests because max_model_len=4096."""
    async with httpx.AsyncClient() as client:
        async def one(req):
            async with sem:
                body = dict(req["body"])
                status, j, lat = await call_openai(
                    client, args.decode, "qwen", body,
                )
                ok = status == 200
                tok = 0
                if j and "usage" in j:
                    tok = int(j["usage"].get("completion_tokens", 0) or 0)
                err = "" if ok else json.dumps(j)[:200]
                return Result(
                    idx=req["idx"], shape=req["shape"], slo_ms=req["slo_ms"],
                    endpoint_used="vllm-decode", ok=ok, latency_ms=lat,
                    error=err, cost_usd=cost_for("vllm-decode", lat),
                    tokens_out=tok,
                )
        return await asyncio.gather(*(one(r) for r in workload))


async def arm_round_robin(
    workload: list[dict[str, Any]], args, sem: asyncio.Semaphore
) -> list[Result]:
    """Arm B: alternate between vllm-decode and vllm-prefill."""
    async with httpx.AsyncClient() as client:
        async def one(req):
            async with sem:
                if req["idx"] % 2 == 0:
                    url, name = args.decode, "vllm-decode"
                else:
                    url, name = args.prefill, "vllm-prefill"
                body = dict(req["body"])
                status, j, lat = await call_openai(client, url, "qwen", body)
                ok = status == 200
                tok = 0
                if j and "usage" in j:
                    tok = int(j["usage"].get("completion_tokens", 0) or 0)
                err = "" if ok else json.dumps(j)[:200]
                return Result(
                    idx=req["idx"], shape=req["shape"], slo_ms=req["slo_ms"],
                    endpoint_used=name, ok=ok, latency_ms=lat,
                    error=err, cost_usd=cost_for(name, lat), tokens_out=tok,
                )
        return await asyncio.gather(*(one(r) for r in workload))


# Routing hint per shape: what tenant/priority header to send so the router's
# rules fire meaningfully. A naive caller would just blast the router with
# `tenant-id: default, priority: standard`, but our router respects rule
# matching on priority_tier and workload_type (inferred).
SHAPE_HEADERS = {
    "short_chat":   {"x-tenant-id": "default",               "x-priority-tier": "standard"},
    "reasoning":    {"x-tenant-id": "tenant-premium-corp",   "x-priority-tier": "premium"},
    "long_context": {"x-tenant-id": "default",               "x-priority-tier": "standard"},
    "prefill_heavy":{"x-tenant-id": "default",               "x-priority-tier": "standard"},
    "decode_heavy": {"x-tenant-id": "default",               "x-priority-tier": "standard"},
}


# ------- intent-based router (Arm D baseline) -------
# A realistic "smart" naive router: reads the prompt, classifies intent by
# keyword, maps intent -> backend. Does NOT see queue depth, cost, context
# limits, priority tiers, or tenant policies. This is how most teams build
# their first "intelligent" router before learning that intent alone isn't
# enough.
def classify_intent(messages: list[dict[str, str]]) -> str:
    """Naive keyword classifier → one of reasoning / summarization /
    generation / chat. Fast (<0.1 ms) and deterministic. Applies
    `_INTENT_NOISE["fraction"]` random mis-classification to simulate
    realistic ML classifier error rates in production."""
    text = " ".join(m.get("content", "") for m in messages if m.get("role") in ("user", "system")).lower()

    if any(p in text for p in (
        "step by step", "think step", "derive", "derivation", "prove ",
        "probability", "show the derivation", "calculation", "analyse the",
        "analyze the",
    )):
        return _maybe_noisy_intent("reasoning")

    if any(p in text for p in (
        "summarise", "summarize", "summary", "tl;dr", "extract", "bullet",
        "in 3 bullets", "in 2 sentences",
    )):
        return _maybe_noisy_intent("summarization")

    if any(p in text for p in (
        "write a ", "essay", "short story", "400 words", "800 words",
        "2000-word", "detailed", "400-word",
    )):
        return _maybe_noisy_intent("generation")

    return _maybe_noisy_intent("chat")


# Intent → backend. Heavy intents go to the big model; chat stays small.
# Notice: every "hard" intent routes to vllm-quality regardless of load.
INTENT_TO_BACKEND = {
    # Intent-based routing sends "hard" intents to prefill lane (the bigger/
    # more-batched one) and "easy" chat to decode lane — matching how many
    # teams actually wire this up in production.
    "reasoning":     ("prefill", "vllm-prefill"),
    "summarization": ("prefill", "vllm-prefill"),
    "generation":    ("prefill", "vllm-prefill"),
    "chat":          ("decode",  "vllm-decode"),
}


async def arm_intent(
    workload: list[dict[str, Any]], args, sem: asyncio.Semaphore
) -> list[Result]:
    """Arm C: intent-based classifier routes by prompt text. No awareness of
    queue, cost, context limits, tenant policy, or backend config — just
    keywords in the user message."""
    async with httpx.AsyncClient() as client:
        async def one(req):
            async with sem:
                body = dict(req["body"])
                intent = classify_intent(body.get("messages", []))
                lane, name = INTENT_TO_BACKEND[intent]
                url = args.prefill if lane == "prefill" else args.decode
                status, j, lat = await call_openai(client, url, "qwen", body)
                ok = status == 200
                tok = 0
                if j and "usage" in j:
                    tok = int(j["usage"].get("completion_tokens", 0) or 0)
                err = "" if ok else json.dumps(j)[:200]
                return Result(
                    idx=req["idx"], shape=req["shape"], slo_ms=req["slo_ms"],
                    endpoint_used=name, ok=ok, latency_ms=lat,
                    error=err, cost_usd=cost_for(name, lat), tokens_out=tok,
                )
        return await asyncio.gather(*(one(r) for r in workload))


async def arm_spec_decode(
    workload: list[dict[str, Any]], args, sem: asyncio.Semaphore
) -> list[Result]:
    """Arm E: single vLLM with n-gram speculative decoding — no routing.
    This is the honest alternative to routing when the only reason you had
    two backends was 'use the big model for hard stuff, small for easy'.
    Spec decode gives you a single backend that draft-tokenises through
    n-gram lookup and verifies on the 7B, gaining throughput without the
    multi-backend complexity."""
    async with httpx.AsyncClient() as client:
        async def one(req):
            async with sem:
                body = dict(req["body"])
                status, j, lat = await call_openai(
                    client, args.spec, "qwen", body,
                )
                ok = status == 200
                tok = 0
                if j and "usage" in j:
                    tok = int(j["usage"].get("completion_tokens", 0) or 0)
                err = "" if ok else json.dumps(j)[:200]
                return Result(
                    idx=req["idx"], shape=req["shape"], slo_ms=req["slo_ms"],
                    endpoint_used="vllm-spec", ok=ok, latency_ms=lat,
                    error=err, cost_usd=cost_for("vllm-spec", lat),
                    tokens_out=tok,
                )
        return await asyncio.gather(*(one(r) for r in workload))


async def arm_router(
    workload: list[dict[str, Any]], args, sem: asyncio.Semaphore
) -> list[Result]:
    """Arm D: everything through the router."""
    async with httpx.AsyncClient() as client:
        async def one(req):
            async with sem:
                body = dict(req["body"])
                status, j, lat = await call_openai(
                    client, args.router, "qwen", body,
                    headers=SHAPE_HEADERS.get(req["shape"], {}),
                )
                ok = status == 200
                endpoint = "unknown"
                tok = 0
                if j:
                    tf = j.get("_tokenflow", {}) if isinstance(j, dict) else {}
                    endpoint = (
                        tf.get("endpoint")
                        or tf.get("endpoint_name")
                        or tf.get("selected_endpoint")
                        or "unknown"
                    )
                    if "usage" in j:
                        tok = int(j["usage"].get("completion_tokens", 0) or 0)
                err = "" if ok else json.dumps(j)[:200]
                return Result(
                    idx=req["idx"], shape=req["shape"], slo_ms=req["slo_ms"],
                    endpoint_used=endpoint, ok=ok, latency_ms=lat,
                    error=err, cost_usd=cost_for(endpoint, lat), tokens_out=tok,
                )
        return await asyncio.gather(*(one(r) for r in workload))


# ------- reporting -------
def p(xs, q):
    xs = sorted(xs)
    if not xs:
        return 0.0
    k = int(round((q / 100.0) * (len(xs) - 1)))
    return xs[k]


def summarise(arm: str, results: list[Result], wall_s: float) -> dict[str, Any]:
    ok = [r for r in results if r.ok]
    lats = [r.latency_ms for r in ok]
    by_endpoint: dict[str, int] = {}
    for r in ok:
        by_endpoint[r.endpoint_used] = by_endpoint.get(r.endpoint_used, 0) + 1
    slo_miss = sum(1 for r in ok if r.latency_ms > r.slo_ms)
    total_cost = sum(r.cost_usd for r in ok)
    total_tokens = sum(r.tokens_out for r in ok)
    return {
        "arm": arm,
        "requests": len(results),
        "success": len(ok),
        "failed": len(results) - len(ok),
        "throughput_rps": round(len(ok) / wall_s, 2) if wall_s > 0 else 0.0,
        "p50_ms": round(p(lats, 50), 1) if lats else 0.0,
        "p95_ms": round(p(lats, 95), 1) if lats else 0.0,
        "p99_ms": round(p(lats, 99), 1) if lats else 0.0,
        "mean_ms": round(statistics.mean(lats), 1) if lats else 0.0,
        "slo_miss": slo_miss,
        "slo_miss_rate_pct": round(100.0 * slo_miss / len(ok), 1) if ok else 0.0,
        "total_cost_usd": round(total_cost, 6),
        "cost_per_1k_tok_usd": round(1000 * total_cost / total_tokens, 6) if total_tokens else 0.0,
        "endpoint_distribution": by_endpoint,
    }


def print_table(rows: list[dict[str, Any]]) -> None:
    cols = [
        ("arm",                 "Arm",               10),
        ("requests",            "Req",                5),
        ("success",             "OK",                 5),
        ("failed",              "Fail",               5),
        ("throughput_rps",      "RPS",                6),
        ("p50_ms",              "p50 ms",             8),
        ("p95_ms",              "p95 ms",             8),
        ("p99_ms",              "p99 ms",             8),
        ("slo_miss_rate_pct",   "SLO miss %",        11),
        ("total_cost_usd",      "$ total",           10),
        ("cost_per_1k_tok_usd", "$/1k tok",          11),
    ]
    header = " | ".join(f"{h:>{w}}" for _, h, w in cols)
    print(header)
    print("-" * len(header))
    for r in rows:
        print(" | ".join(f"{str(r[k]):>{w}}" for k, _, w in cols))


# ------- main -------
async def main_async(args):
    # Configure noise sources before generating workload
    _INTENT_NOISE["fraction"] = args.intent_noise
    _INTENT_NOISE["rng"] = random.Random(args.seed + 2)
    workload = build_workload(args.n, args.seed, workload_noise=args.workload_noise)
    if args.intent_noise or args.workload_noise:
        print(f"[noise] intent={args.intent_noise:.2f}  workload={args.workload_noise:.2f}")

    # probe health
    async with httpx.AsyncClient(timeout=5.0) as c:
        probes = [("router", args.router), ("decode", args.decode), ("prefill", args.prefill)]
        if args.spec:
            probes.append(("spec", args.spec))
        for label, url in probes:
            try:
                r = await c.get(f"{url}/health")
                print(f"[probe] {label} ({url}): HTTP {r.status_code}")
            except Exception as e:
                print(f"[probe] {label} ({url}): FAIL — {e}", file=sys.stderr)
                sys.exit(1)

    sem = asyncio.Semaphore(args.concurrency)

    rows = []
    all_raw: dict[str, list[Result]] = {}
    plan = [
        ("A direct",             arm_direct),
        ("B round-robin",        arm_round_robin),
        ("C intent-based",       arm_intent),
        ("D router",             arm_router),
    ]
    if args.spec:
        plan.append(("E spec-decode",        arm_spec_decode))
    for arm_name, fn in plan:
        print(f"\n>> Running arm: {arm_name} ({args.n} requests, concurrency={args.concurrency})")
        t0 = time.perf_counter()
        res = await fn(workload, args, sem)
        wall = time.perf_counter() - t0
        summary = summarise(arm_name, res, wall)
        rows.append(summary)
        all_raw[arm_name] = res
        print(f"   took {wall:.1f}s  endpoint_distribution={summary['endpoint_distribution']}")

    print("\n" + "=" * 40 + " RESULTS " + "=" * 40)
    print_table(rows)

    # persist both summary and per-request raw data for post-hoc analysis
    if args.out:
        with open(args.out, "w") as f:
            json.dump({
                "workload_size": args.n,
                "seed": args.seed,
                "concurrency": args.concurrency,
                "summary": rows,
                "raw": {
                    arm: [
                        {
                            "idx": r.idx,
                            "shape": r.shape,
                            "slo_ms": r.slo_ms,
                            "endpoint_used": r.endpoint_used,
                            "ok": r.ok,
                            "latency_ms": round(r.latency_ms, 2),
                            "tokens_out": r.tokens_out,
                            "cost_usd": round(r.cost_usd, 8),
                        }
                        for r in raw
                    ]
                    for arm, raw in all_raw.items()
                },
            }, f, indent=2)
        print(f"\nResults saved to {args.out}")


def main():
    ap = argparse.ArgumentParser(description="TokenFlow 3-arm benchmark")
    ap.add_argument("--router",  default="http://localhost:8080")
    ap.add_argument("--decode",  default="http://localhost:8001",
                    help="low-latency decode-tuned backend (v4) or vllm-fast (v3 compat)")
    ap.add_argument("--prefill", default="http://localhost:8002",
                    help="long-ctx prefill-tuned backend (v4) or vllm-quality (v3 compat)")
    ap.add_argument("--spec",    default=None,
                    help="optional: single vLLM with n-gram speculative decoding for arm E")
    # v3-compat aliases
    ap.add_argument("--fast",    dest="decode",  help=argparse.SUPPRESS)
    ap.add_argument("--quality", dest="prefill", help=argparse.SUPPRESS)
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--concurrency", type=int, default=32)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default="benchmark_results.json")
    ap.add_argument("--intent-noise", type=float, default=0.0,
                    help="fraction [0,1] of intent classifications that are randomised; "
                         "simulates real ML classifier mis-classification rate")
    ap.add_argument("--workload-noise", type=float, default=0.0,
                    help="fraction [0,1] of requests sent with a misleading max_tokens, "
                         "which confuses TokenFlow's workload_type inference")
    args = ap.parse_args()
    asyncio.run(main_async(args))


if __name__ == "__main__":
    main()
