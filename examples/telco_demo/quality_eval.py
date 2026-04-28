"""Quality evaluation — does the NVIDIA-shape classifier improve answer
quality enough to justify the latency / cost overhead?

Sends the same hard prompts through TokenFlow with the classifier ON
and OFF, captures the actual response text, and scores both with
GPT-4o-mini as judge on:

  - correctness  (factually right?)
  - completeness (covers what was asked?)
  - reasoning    (shows valid step-by-step where required?)

Prints a per-prompt comparison table and an aggregate verdict.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import statistics
import time
from dataclasses import dataclass, asdict
from typing import Any, Optional

import httpx


# Hard-leaning prompts — the ones where the classifier-routed-to-larger-model
# difference should matter most. Drawn from the benchmark workload mix.
HARD_PROMPTS = [
    {
        "id": "migration_cobol",
        "workload": "ai_assisted_migration",
        "tenant": "tenant-migration-tools",
        "messages": [
            {"role": "system", "content": "You are an expert code migration assistant. Think carefully before answering."},
            {"role": "user", "content": (
                "Translate this COBOL FILLER PIC X(42) to a Python dataclass with the same byte layout. "
                "Preserve trailing-space semantics and explain why your choice handles the byte alignment correctly."
            )},
        ],
        "judge_criteria": [
            "Does the answer correctly explain that PIC X(42) is a fixed-width 42-byte ASCII string?",
            "Does it preserve trailing-space (rather than null-trimmed) semantics in the Python representation?",
            "Does it explain WHY (not just give code)?",
        ],
    },
    {
        "id": "twin_availability",
        "workload": "digital_twin_simulation",
        "tenant": "tenant-digital-twin",
        "messages": [
            {"role": "system", "content": "Reason step by step about the simulation parameters."},
            {"role": "user", "content": (
                "A network of 412 core nodes is exposed to a 0.3% per-day failure rate. With an MTTR of 4 hours and "
                "redundancy=2 across geographically separate paths, what is the expected service-availability over a "
                "30-day window? Show the derivation, including treatment of correlated failure modes (e.g., regional "
                "outage covering multiple nodes simultaneously)."
            )},
        ],
        "judge_criteria": [
            "Does it correctly compute per-node availability from MTBF/MTTR?",
            "Does it correctly handle redundancy=2 (probability that BOTH paths fail simultaneously)?",
            "Does it acknowledge the correlation question (independent vs. correlated failure)?",
            "Is the math right at each step?",
        ],
    },
    {
        "id": "twin_capacity",
        "workload": "digital_twin_simulation",
        "tenant": "tenant-digital-twin",
        "messages": [
            {"role": "user", "content": (
                "We're sizing a Kubernetes cluster for an inference service that has p95 request rate 1200 rps, "
                "p99 4000 rps, and per-replica throughput of 35 rps at p95 latency 200ms. We have a 30s liveness "
                "probe interval, 90s pod startup time, and we want headroom for a 1-zone outage in a 3-zone deployment. "
                "How many replicas should we run, and how does the answer change if we add HPA with 60s reaction time?"
            )},
        ],
        "judge_criteria": [
            "Does it derive replica count from rps / per-replica capacity correctly?",
            "Does it factor in 1-zone outage (need to over-provision by ~50%)?",
            "Does it discuss the HPA reaction-time vs. p99 burst trade-off?",
            "Are the numerical answers reasonable?",
        ],
    },
    {
        "id": "esg_extract",
        "workload": "esg_batch",
        "tenant": "tenant-esg-reporting",
        "messages": [
            {"role": "user", "content": (
                "Extract the three most material ESG risks from the following report and rank them by financial impact:\n\n"
                "The Energy Performance Certificate Programme dataset for FY2025 includes 18,432 properties across nine regions. "
                "Carbon intensity reduction targets average 24% with regional variance from 18% to 31%. Embedded carbon costs in "
                "the supply chain are estimated at 41% of total scope-3 emissions. Migration of legacy workloads to renewable-energy-"
                "backed datacentres targets a 38% scope-2 reduction by FY2027. The carbon-disclosure framework follows TCFD with "
                "extended quantitative disclosures aligned to ISSB S2. Quarterly board reviews now include carbon-budget-vs-actual "
                "ratios alongside financial ratios. Network-equipment lifecycle assumptions use a 7-year useful-life model with 92% "
                "recyclable mass at end-of-life. Renewable-energy PPAs cover 64% of contracted demand; spot-market exposure remains "
                "12% of total energy spend. ESG-aligned capex is reviewed against a $0.18/kgCO2e shadow price for project NPV calculations."
            )},
        ],
        "judge_criteria": [
            "Are the three identified risks actually the most material from the document?",
            "Is the financial-impact ranking defensible from the numbers given (e.g., 41% scope-3 vs 24% intensity)?",
            "Does the answer cite specific figures from the document, not invent new ones?",
        ],
    },
    {
        "id": "migration_kotlin",
        "workload": "ai_assisted_migration",
        "tenant": "tenant-migration-tools",
        "messages": [
            {"role": "system", "content": "You are an expert code migration assistant."},
            {"role": "user", "content": (
                "Rewrite this Java 8 stream pipeline as a Kotlin Flow with structured concurrency, preserving the "
                "back-pressure semantics:\n\n"
                "List<String> result = source.stream()\n"
                "    .parallel()\n"
                "    .filter(s -> s.length() > 0)\n"
                "    .map(transform::apply)\n"
                "    .limit(100)\n"
                "    .collect(Collectors.toList());\n\n"
                "Explain how Flow's back-pressure differs from parallel streams, and what concurrency primitive "
                "(channelFlow / flowOn / etc.) preserves the original parallelism."
            )},
        ],
        "judge_criteria": [
            "Does the Kotlin Flow code correctly use channelFlow or flowOn for parallelism?",
            "Does it preserve the take(100) / limit semantics?",
            "Does it correctly explain that Flow's back-pressure is suspending vs streams' fork-join pull model?",
            "Is the concurrency primitive choice (Dispatchers, structured concurrency) defensible?",
        ],
    },
]


@dataclass
class Response:
    arm: str                  # "tf_no_classifier" | "tf_with_classifier"
    prompt_id: str
    workload: str
    endpoint: str             # which backend served it
    latency_ms: float
    text: str
    tokens_in: int
    tokens_out: int


async def send(client: httpx.AsyncClient, url: str, prompt: dict) -> Response:
    body = {
        "model": "qwen",
        "messages": prompt["messages"],
        "max_tokens": 600,
        "temperature": 0.0,
    }
    headers = {
        "x-tenant-id": prompt["tenant"],
        "x-priority-tier": "premium" if prompt["workload"] == "digital_twin_simulation" else "standard",
    }
    t0 = time.perf_counter()
    r = await client.post(f"{url}/v1/chat/completions", json=body, headers=headers, timeout=120.0)
    lat = (time.perf_counter() - t0) * 1000
    r.raise_for_status()
    j = r.json()
    text = j["choices"][0]["message"]["content"]
    tf = j.get("_tokenflow", {})
    return Response(
        arm="placeholder",
        prompt_id=prompt["id"],
        workload=prompt["workload"],
        endpoint=tf.get("endpoint", "unknown"),
        latency_ms=lat,
        text=text,
        tokens_in=j.get("usage", {}).get("prompt_tokens", 0),
        tokens_out=j.get("usage", {}).get("completion_tokens", 0),
    )


JUDGE_PROMPT = """You are evaluating two AI-generated answers to the same prompt. For each answer, score it on a 1-10 scale across four dimensions:

  - correctness:  factually correct? math/code right?
  - completeness: addresses what was asked, including all sub-parts?
  - reasoning:    shows valid reasoning where the prompt asks for it (otherwise score 5/10)?
  - usefulness:   would a domain expert actually use this answer?

Also produce a one-sentence verdict on which answer is better and why.

Original prompt:
================
{prompt}

Specific things the answer SHOULD do:
{criteria}

Answer A (from {a_endpoint}):
========
{a_text}

Answer B (from {b_endpoint}):
========
{b_text}

Output JSON ONLY (no prose). Schema:
{{
  "a": {{"correctness": 1-10, "completeness": 1-10, "reasoning": 1-10, "usefulness": 1-10}},
  "b": {{"correctness": 1-10, "completeness": 1-10, "reasoning": 1-10, "usefulness": 1-10}},
  "winner": "a" | "b" | "tie",
  "verdict": "<one sentence>"
}}"""


async def judge(client: httpx.AsyncClient, openai_key: str, a: Response, b: Response, prompt_def: dict) -> dict:
    prompt_text = "\n".join(m["content"] for m in prompt_def["messages"])
    body = {
        "model": "gpt-4o-mini",
        "messages": [
            {"role": "system", "content": "You are an impartial expert reviewer."},
            {"role": "user", "content": JUDGE_PROMPT.format(
                prompt=prompt_text,
                criteria="\n".join(f"  - {c}" for c in prompt_def["judge_criteria"]),
                a_endpoint=a.endpoint,
                a_text=a.text,
                b_endpoint=b.endpoint,
                b_text=b.text,
            )},
        ],
        "temperature": 0.0,
        "response_format": {"type": "json_object"},
    }
    r = await client.post(
        "https://api.openai.com/v1/chat/completions",
        json=body,
        headers={"Authorization": f"Bearer {openai_key}", "Content-Type": "application/json"},
        timeout=60.0,
    )
    r.raise_for_status()
    return json.loads(r.json()["choices"][0]["message"]["content"])


async def main_async(args):
    openai_key = os.getenv("OPENAI_API_KEY") or args.openai_key
    if not openai_key:
        raise SystemExit("set OPENAI_API_KEY or pass --openai-key")

    print(f"[1/3] sending {len(HARD_PROMPTS)} prompts to TokenFlow at {args.router}")
    print(f"      arm assumption: classifier = {args.arm_label}")
    async with httpx.AsyncClient() as client:
        responses = []
        for p in HARD_PROMPTS:
            try:
                r = await send(client, args.router, p)
                r.arm = args.arm_label
                responses.append(r)
                print(f"   {p['id']:24s}  endpoint={r.endpoint:14s}  lat={r.latency_ms:>7.0f}ms  out_tok={r.tokens_out}")
            except Exception as e:
                print(f"   {p['id']}: FAILED — {e}")

    out = {"arm": args.arm_label, "router": args.router, "responses": [asdict(r) for r in responses]}
    with open(args.out, "w") as f:
        json.dump(out, f, indent=2)
    print(f"[saved] {args.out}")

    if not args.judge:
        return

    print()
    print(f"[2/3] judging against {args.judge_against}")
    other = json.loads(open(args.judge_against).read())
    other_by_id = {r["prompt_id"]: r for r in other["responses"]}

    judgments = []
    async with httpx.AsyncClient() as client:
        for r in responses:
            if r.prompt_id not in other_by_id:
                continue
            other_r = other_by_id[r.prompt_id]
            other_resp = Response(**other_r)
            prompt_def = next(p for p in HARD_PROMPTS if p["id"] == r.prompt_id)
            try:
                # judge B = current run, A = other run, so:
                a_resp = other_resp
                b_resp = r
                v = await judge(client, openai_key, a_resp, b_resp, prompt_def)
                v["prompt_id"] = r.prompt_id
                v["a_arm"] = other["arm"]
                v["b_arm"] = args.arm_label
                v["a_endpoint"] = a_resp.endpoint
                v["b_endpoint"] = b_resp.endpoint
                v["a_latency_ms"] = a_resp.latency_ms
                v["b_latency_ms"] = b_resp.latency_ms
                judgments.append(v)
                a_total = sum(v["a"].values())
                b_total = sum(v["b"].values())
                print(f"   {r.prompt_id:24s}  A({other['arm']:>20s})={a_total:>3}  B({args.arm_label:>20s})={b_total:>3}  winner={v['winner']:>4s}")
            except Exception as e:
                print(f"   {r.prompt_id}: JUDGE FAILED — {e}")

    print()
    print("[3/3] aggregate")
    a_wins = sum(1 for j in judgments if j["winner"] == "a")
    b_wins = sum(1 for j in judgments if j["winner"] == "b")
    ties   = sum(1 for j in judgments if j["winner"] == "tie")
    print(f"   {other['arm']} wins: {a_wins}")
    print(f"   {args.arm_label} wins: {b_wins}")
    print(f"   ties: {ties}")
    if judgments:
        a_avg = statistics.mean(sum(j["a"].values()) for j in judgments)
        b_avg = statistics.mean(sum(j["b"].values()) for j in judgments)
        print(f"   avg total score: {other['arm']}={a_avg:.1f}  {args.arm_label}={b_avg:.1f}  (max=40)")

    with open(args.judgments_out, "w") as f:
        json.dump({
            "arm_a": other["arm"],
            "arm_b": args.arm_label,
            "judgments": judgments,
            "summary": {"a_wins": a_wins, "b_wins": b_wins, "ties": ties,
                        "a_avg_score": a_avg if judgments else 0,
                        "b_avg_score": b_avg if judgments else 0},
        }, f, indent=2)
    print(f"[saved] {args.judgments_out}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--router", default="http://localhost:8080")
    ap.add_argument("--arm-label", required=True,
                    help='e.g. "tf_no_classifier" or "tf_with_classifier"')
    ap.add_argument("--out", required=True, help="path to write responses JSON")
    ap.add_argument("--judge", action="store_true",
                    help="if set, judge against --judge-against and produce verdict")
    ap.add_argument("--judge-against", default="",
                    help="path to other arm's responses JSON")
    ap.add_argument("--judgments-out", default="examples/telco_demo/results/judgments.json")
    ap.add_argument("--openai-key", default="")
    args = ap.parse_args()
    asyncio.run(main_async(args))


if __name__ == "__main__":
    main()
