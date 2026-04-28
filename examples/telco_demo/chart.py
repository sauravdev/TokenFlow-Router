"""Render the headline + per-workload charts from results/benchmark.json."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "axes.edgecolor": "#1f2937",
    "axes.labelcolor": "#111827",
    "axes.titlecolor": "#111827",
    "xtick.color": "#111827",
    "ytick.color": "#111827",
    "text.color": "#111827",
})

INTENT_COLOR = "#2563eb"
ROUTER_COLOR = "#059669"


def style(ax, title, ylabel, ymin=0, ymax=None):
    ax.set_title(title, fontsize=12.5, color="#111827", pad=12, fontweight="bold")
    ax.set_ylabel(ylabel, fontsize=10.5, color="#374151")
    ax.tick_params(axis="y", labelsize=10, colors="#374151")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#cbd5e1")
    ax.spines["bottom"].set_color("#cbd5e1")
    ax.grid(axis="y", linestyle="--", alpha=0.5, color="#e5e7eb")
    ax.set_axisbelow(True)
    if ymax is not None:
        ax.set_ylim(ymin, ymax)


def headline(data, out):
    by_arm = {row["arm"]: row for row in data["totals"]}
    intent = by_arm.get("intent", {})
    router = by_arm.get("tokenflow", {})

    metrics = [
        ("Success rate ↑",        "% of requests",
         [intent.get("success_pct", 0), router.get("success_pct", 0)],
         lambda v: f"{v:.1f}%", 0, 110),
        ("p95 latency ↓",         "milliseconds",
         [intent.get("p95_ms", 0), router.get("p95_ms", 0)],
         lambda v: f"{v:,.0f}", 0,
         max(intent.get("p95_ms", 0), router.get("p95_ms", 0)) * 1.22 or 1),
        ("Total cost ↓",          "USD",
         [intent.get("total_cost_usd", 0), router.get("total_cost_usd", 0)],
         lambda v: f"${v:.3f}", 0,
         max(intent.get("total_cost_usd", 0), router.get("total_cost_usd", 0)) * 1.25 or 0.001),
        ("Cost per 1k tokens ↓",  "USD / 1k tok",
         [intent.get("cost_per_1k_tok_usd", 0), router.get("cost_per_1k_tok_usd", 0)],
         lambda v: f"${v:.4f}", 0,
         max(intent.get("cost_per_1k_tok_usd", 0), router.get("cost_per_1k_tok_usd", 0)) * 1.25 or 0.0001),
    ]

    fig, axes = plt.subplots(1, 4, figsize=(18, 5.6))
    labels = ["Intent-based", "TokenFlow"]
    colors = [INTENT_COLOR, ROUTER_COLOR]

    for i, (title, ylabel, vals, fmt, ymin, ymax) in enumerate(metrics):
        ax = axes[i]
        b = ax.bar(labels, vals, color=colors, width=0.55, edgecolor="none")
        style(ax, title, ylabel, ymin=ymin, ymax=ymax)
        for bb, v in zip(b, vals):
            ax.text(bb.get_x() + bb.get_width() / 2, v + ymax * 0.01, fmt(v),
                    ha="center", va="bottom", fontsize=12,
                    color="#111827", fontweight="bold")

    fig.suptitle(
        f"Telco multi-workload — TokenFlow vs intent-based "
        f"({data['workload_size']} requests, ~{data['workload_size']/max(data['rate_rps'],1e-6):.0f}s)",
        fontsize=14.5, color="#111827", fontweight="bold", y=0.99,
    )
    fig.text(
        0.5, 0.02,
        "6 workloads (customer_care_voice / rag_retrieval / esg_batch / "
        "ai_assisted_migration / trust_inventory / digital_twin_simulation) — same fleet, same workload, same seed",
        ha="center", fontsize=10, color="#4b5563", style="italic",
    )
    plt.tight_layout(rect=[0, 0.05, 1, 0.94])
    fig.savefig(out, dpi=200, facecolor="white", bbox_inches="tight")
    print(f"wrote {out}")


def per_workload(data, out):
    rows = data.get("by_workload", [])
    workloads = sorted({r["workload"] for r in rows})

    intent_cost = []
    router_cost = []
    intent_p95  = []
    router_p95  = []
    intent_succ = []
    router_succ = []
    for w in workloads:
        i = next((r for r in rows if r["arm"] == "intent"    and r["workload"] == w), {})
        t = next((r for r in rows if r["arm"] == "tokenflow" and r["workload"] == w), {})
        intent_cost.append(i.get("total_cost_usd", 0))
        router_cost.append(t.get("total_cost_usd", 0))
        intent_p95.append (i.get("p95_ms", 0))
        router_p95.append (t.get("p95_ms", 0))
        intent_succ.append(i.get("success_pct", 0))
        router_succ.append(t.get("success_pct", 0))

    fig, axes = plt.subplots(1, 3, figsize=(20, 6))
    x = list(range(len(workloads)))
    w = 0.36

    for ax, intent_v, router_v, title, ylabel, fmt, ymax in [
        (axes[0], intent_cost, router_cost, "Cost per workload ↓",   "USD",
         lambda v: f"${v:.3f}", max(max(intent_cost+router_cost+[0.001])) * 1.25),
        (axes[1], intent_p95,  router_p95,  "p95 per workload ↓",    "ms",
         lambda v: f"{v:,.0f}", max(max(intent_p95+router_p95+[1])) * 1.22),
        (axes[2], intent_succ, router_succ, "Success per workload ↑", "%",
         lambda v: f"{v:.0f}%", 110),
    ]:
        ax.bar([i - w/2 for i in x], intent_v, w, label="Intent-based", color=INTENT_COLOR)
        ax.bar([i + w/2 for i in x], router_v, w, label="TokenFlow",    color=ROUTER_COLOR)
        style(ax, title, ylabel, ymin=0, ymax=ymax)
        ax.set_xticks(x)
        ax.set_xticklabels(workloads, fontsize=9, rotation=22, ha="right")
        ax.legend(loc="upper right", frameon=False, fontsize=10)
        for xi, iv, rv in zip(x, intent_v, router_v):
            ax.text(xi - w/2, iv + ymax * 0.01, fmt(iv),
                    ha="center", va="bottom", fontsize=8.5, color="#111827", fontweight="bold")
            ax.text(xi + w/2, rv + ymax * 0.01, fmt(rv),
                    ha="center", va="bottom", fontsize=8.5, color="#111827", fontweight="bold")

    fig.suptitle("Per-workload breakdown — same workload, same fleet",
                 fontsize=14, color="#111827", fontweight="bold", y=0.99)
    plt.tight_layout(rect=[0, 0.04, 1, 0.94])
    fig.savefig(out, dpi=200, facecolor="white", bbox_inches="tight")
    print(f"wrote {out}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--json",         default="examples/telco_demo/results/benchmark.json")
    ap.add_argument("--out-headline", default="examples/telco_demo/results/chart_headline.png")
    ap.add_argument("--out-workload", default="examples/telco_demo/results/chart_per_workload.png")
    args = ap.parse_args()

    data = json.loads(Path(args.json).read_text())
    Path(args.out_headline).parent.mkdir(parents=True, exist_ok=True)
    headline(data, Path(args.out_headline))
    per_workload(data, Path(args.out_workload))


if __name__ == "__main__":
    main()
