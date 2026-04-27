"""Generate the headline chart for the production-scenario benchmark.

Reads `results/benchmark.json` (produced by benchmark.py) and emits a
4-panel comparison plus a per-tenant breakdown.

Usage:
    pip install matplotlib
    python examples/production_demo/chart.py --json results/benchmark.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt


# Light theme — renders the same in any viewer (LinkedIn, GitHub, Slack).
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


# Brand palette
INTENT_COLOR = "#2563eb"   # blue
ROUTER_COLOR = "#059669"   # green
NEUTRAL = "#94a3b8"


def style_ax(ax, title, ylabel, ymax=None):
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
        ax.set_ylim(0, ymax)


def headline_chart(data: dict, out: Path) -> None:
    """4-panel: success rate, p95 latency, total cost, cost per 1k tok."""
    by_arm = {row["arm"]: row for row in data["totals"]}
    intent = by_arm.get("intent", {})
    router = by_arm.get("tokenflow", {})

    metrics = [
        ("Success rate ↑",          "% of requests",
         [intent.get("success_pct", 0), router.get("success_pct", 0)],
         lambda v: f"{v:.1f}%"),
        ("p95 latency ↓",           "milliseconds",
         [intent.get("p95_ms", 0), router.get("p95_ms", 0)],
         lambda v: f"{v:,.0f}"),
        ("Total cost ↓",            "USD",
         [intent.get("total_cost_usd", 0), router.get("total_cost_usd", 0)],
         lambda v: f"${v:.3f}"),
        ("Cost per 1k tokens ↓",    "USD / 1k tok",
         [intent.get("cost_per_1k_tok_usd", 0), router.get("cost_per_1k_tok_usd", 0)],
         lambda v: f"${v:.4f}"),
    ]

    fig, axes = plt.subplots(1, 4, figsize=(17, 5.4))
    labels = ["Intent-based", "TokenFlow"]
    colors = [INTENT_COLOR, ROUTER_COLOR]

    for i, (title, ylabel, values, fmt) in enumerate(metrics):
        ax = axes[i]
        bars = ax.bar(labels, values, color=colors, width=0.55, edgecolor="none")
        ymax = max(max(values) * 1.22, 0.001) if "Success" not in title else 110
        style_ax(ax, title, ylabel, ymax=ymax)
        for b, v in zip(bars, values):
            ax.text(b.get_x() + b.get_width() / 2, v + ymax * 0.01, fmt(v),
                    ha="center", va="bottom", fontsize=12,
                    color="#111827", fontweight="bold")

    fig.suptitle(
        f"TokenFlow Router vs Intent-based — SaaS multi-tenant scenario "
        f"({data['workload_size']} req/arm, ~{data['workload_size']/max(data['rate_rps'], 1e-6):.0f}s)",
        fontsize=14.5, color="#111827", fontweight="bold", y=0.99,
    )
    fig.text(
        0.5, 0.02,
        "Same fleet, same workload, same seed. Three tenants (free/standard/enterprise) with budget caps + GPU allowlists. "
        "Live policy swap from balanced→cost-first at midpoint.",
        ha="center", fontsize=10, color="#4b5563", style="italic",
    )

    plt.tight_layout(rect=[0, 0.05, 1, 0.94])
    fig.savefig(out, dpi=200, facecolor="white", bbox_inches="tight")
    print(f"wrote {out}")


def per_tenant_chart(data: dict, out: Path) -> None:
    """Stacked-by-tenant comparison so the per-tenant policy story shows up."""
    rows = data.get("by_tenant", [])
    tenants = sorted({r["tenant"] for r in rows})

    intent_costs = []
    router_costs = []
    intent_success = []
    router_success = []
    for t in tenants:
        i = next((r for r in rows if r["arm"] == "intent" and r["tenant"] == t), {})
        r = next((r for r in rows if r["arm"] == "tokenflow" and r["tenant"] == t), {})
        intent_costs.append(i.get("total_cost_usd", 0))
        router_costs.append(r.get("total_cost_usd", 0))
        intent_success.append(i.get("success_pct", 0))
        router_success.append(r.get("success_pct", 0))

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    x = list(range(len(tenants)))
    width = 0.38

    ax = axes[0]
    ax.bar([i - width / 2 for i in x], intent_costs, width, label="Intent-based", color=INTENT_COLOR)
    ax.bar([i + width / 2 for i in x], router_costs, width, label="TokenFlow",    color=ROUTER_COLOR)
    style_ax(ax, "Cost per tenant ↓", "USD",
             ymax=max(max(intent_costs + router_costs) * 1.25, 0.001))
    ax.set_xticks(x)
    ax.set_xticklabels(tenants)
    ax.legend(loc="upper left", frameon=False, fontsize=10.5)
    for xi, ic, rc in zip(x, intent_costs, router_costs):
        ax.text(xi - width / 2, ic, f"${ic:.3f}", ha="center", va="bottom",
                fontsize=10, fontweight="bold")
        ax.text(xi + width / 2, rc, f"${rc:.3f}", ha="center", va="bottom",
                fontsize=10, fontweight="bold")

    ax = axes[1]
    ax.bar([i - width / 2 for i in x], intent_success, width, label="Intent-based", color=INTENT_COLOR)
    ax.bar([i + width / 2 for i in x], router_success, width, label="TokenFlow",    color=ROUTER_COLOR)
    style_ax(ax, "Success rate per tenant ↑", "% of requests", ymax=110)
    ax.set_xticks(x)
    ax.set_xticklabels(tenants)
    ax.legend(loc="lower right", frameon=False, fontsize=10.5)
    for xi, isucc, rsucc in zip(x, intent_success, router_success):
        ax.text(xi - width / 2, isucc, f"{isucc:.1f}%", ha="center", va="bottom",
                fontsize=10, fontweight="bold")
        ax.text(xi + width / 2, rsucc, f"{rsucc:.1f}%", ha="center", va="bottom",
                fontsize=10, fontweight="bold")

    fig.suptitle("Per-tenant breakdown — same workload, same backends",
                 fontsize=13.5, color="#111827", fontweight="bold", y=0.99)
    plt.tight_layout(rect=[0, 0.04, 1, 0.94])
    fig.savefig(out, dpi=200, facecolor="white", bbox_inches="tight")
    print(f"wrote {out}")


def main():
    ap = argparse.ArgumentParser(description="Render production-scenario charts")
    ap.add_argument("--json",     default="examples/production_demo/results/benchmark.json")
    ap.add_argument("--out-headline", default="examples/production_demo/results/chart_headline.png")
    ap.add_argument("--out-tenant",   default="examples/production_demo/results/chart_per_tenant.png")
    args = ap.parse_args()

    data = json.loads(Path(args.json).read_text())
    Path(args.out_headline).parent.mkdir(parents=True, exist_ok=True)
    headline_chart(data, Path(args.out_headline))
    per_tenant_chart(data, Path(args.out_tenant))


if __name__ == "__main__":
    main()
