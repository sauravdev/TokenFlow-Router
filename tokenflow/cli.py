"""
TokenFlow Router CLI.

Usage:
  tokenflow serve              — start the router
  tokenflow register           — register a NIM endpoint
  tokenflow list               — list registered endpoints
  tokenflow simulate           — run a simulation
  tokenflow policy preset      — switch routing preset
  tokenflow explain <id>       — explain a routing decision
"""

from __future__ import annotations

import asyncio
import json
import sys

import typer
from rich.console import Console
from rich.table import Table

app = typer.Typer(name="tokenflow", help="TokenFlow Router CLI")
console = Console()


# ---------------------------------------------------------------------------
# init — interactive onboarding wizard
# ---------------------------------------------------------------------------


@app.command()
def init(
    workdir: str = typer.Option(".", help="Project directory to write configs into"),
    apply: bool = typer.Option(
        False, help="After generating configs, immediately bring up the router via docker compose"
    ),
    resume: bool = typer.Option(
        False, help="Resume a previous interactive session from .tokenflow/onboarding.json"
    ),
) -> None:
    """Interactive setup. Walk through environment, backends, and policy choice; emit
    policy.yaml + .env + register_endpoints.sh. No YAML editing required."""
    from pathlib import Path

    from tokenflow.onboarding import run_onboarding

    run_onboarding(workdir=Path(workdir), apply=apply, resume=resume)


# ---------------------------------------------------------------------------
# serve
# ---------------------------------------------------------------------------


@app.command()
def serve(
    host: str = typer.Option("0.0.0.0", help="Bind host"),
    port: int = typer.Option(8080, help="Bind port"),
    workers: int = typer.Option(1, help="Uvicorn worker count"),
    log_level: str = typer.Option("info", help="Log level"),
    policy_file: str = typer.Option(None, help="Path to policy YAML file"),
    reload: bool = typer.Option(False, help="Enable hot reload (dev only)"),
) -> None:
    """Start the TokenFlow Router server."""
    import os
    import uvicorn

    if policy_file:
        os.environ["TOKENFLOW_POLICY_FILE"] = policy_file

    console.print(f"[bold green]TokenFlow Router[/] starting on {host}:{port}")
    uvicorn.run(
        "tokenflow.main:app",
        host=host,
        port=port,
        workers=workers,
        log_level=log_level,
        reload=reload,
    )


# ---------------------------------------------------------------------------
# register
# ---------------------------------------------------------------------------


@app.command()
def register(
    name: str = typer.Option(..., help="Endpoint name"),
    url: str = typer.Option(..., help="NIM base URL"),
    model: str = typer.Option(..., help="Model name served"),
    gpu: str = typer.Option("UNKNOWN", help="GPU class (H100, A100, L40S, L4, ...)"),
    cost_class: str = typer.Option("standard", help="Cost class (premium/standard/economy)"),
    cost_per_gpu_hour: float = typer.Option(3.0, help="USD per GPU-hour"),
    max_context: int = typer.Option(8192, help="Max context window tokens"),
    router_url: str = typer.Option("http://localhost:8080", help="TokenFlow router URL"),
) -> None:
    """Register a NIM endpoint with the router."""
    import httpx

    payload = {
        "name": name,
        "nim_url": url,
        "model_name": model,
        "gpu_name": gpu,
        "cost_class": cost_class,
        "cost_per_gpu_hour": cost_per_gpu_hour,
        "max_context_tokens": max_context,
    }

    try:
        resp = httpx.post(f"{router_url}/admin/endpoints", json=payload, timeout=10)
        resp.raise_for_status()
        data = resp.json()
        console.print(f"[green]Registered[/] endpoint [bold]{data['name']}[/] (id: {data['id']})")
    except Exception as exc:
        console.print(f"[red]Error:[/] {exc}", err=True)
        raise typer.Exit(1)


# ---------------------------------------------------------------------------
# list
# ---------------------------------------------------------------------------


@app.command(name="list")
def list_endpoints(
    router_url: str = typer.Option("http://localhost:8080", help="TokenFlow router URL"),
) -> None:
    """List all registered endpoints."""
    import httpx

    try:
        resp = httpx.get(f"{router_url}/admin/endpoints", timeout=10)
        resp.raise_for_status()
        endpoints = resp.json()
    except Exception as exc:
        console.print(f"[red]Error:[/] {exc}", err=True)
        raise typer.Exit(1)

    table = Table(title="Registered Endpoints")
    table.add_column("Name", style="cyan")
    table.add_column("Model")
    table.add_column("GPU")
    table.add_column("Cost Class")
    table.add_column("Health")
    table.add_column("Enabled")

    for ep in endpoints:
        health_color = {
            "healthy": "green",
            "degraded": "yellow",
            "unhealthy": "red",
            "unknown": "dim",
        }.get(ep.get("health", "unknown"), "dim")

        table.add_row(
            ep["name"],
            ep["model_name"],
            ep.get("gpu_name", "?"),
            ep.get("cost_class", "?"),
            f"[{health_color}]{ep.get('health', 'unknown')}[/]",
            "✓" if ep.get("enabled") else "✗",
        )

    console.print(table)


# ---------------------------------------------------------------------------
# simulate
# ---------------------------------------------------------------------------


@app.command()
def simulate(
    preset: str = typer.Option("balanced", help="Policy preset (latency-first/balanced/cost-first)"),
    requests: int = typer.Option(100, help="Number of synthetic requests to simulate"),
    model: str = typer.Option("meta/llama-3.1-8b-instruct", help="Model to test with"),
) -> None:
    """Run a simulation with synthetic requests against a standard fleet."""

    async def _run():
        from tokenflow.models import RoutingPolicy
        from tokenflow.simulator.engine import (
            make_request_body,
            run_simulation,
            standard_fleet,
        )

        policy = RoutingPolicy(name=preset, preset=preset)
        fleet = standard_fleet()

        console.print(f"[bold]Simulating[/] {requests} requests with [cyan]{preset}[/] policy...")

        req_list = []
        workloads = ["prefill_heavy", "decode_heavy", "balanced"]
        for i in range(requests):
            wl = workloads[i % len(workloads)]
            if wl == "prefill_heavy":
                body = make_request_body(model=model, input_tokens=8000, output_tokens=200)
            elif wl == "decode_heavy":
                body = make_request_body(model=model, input_tokens=200, output_tokens=1500)
            else:
                body = make_request_body(model=model, input_tokens=500, output_tokens=300)
            req_list.append(body)

        result = await run_simulation(req_list, policy, fleet)

        table = Table(title=f"Simulation Results — {preset}")
        table.add_column("Metric", style="cyan")
        table.add_column("Value")
        table.add_row("Total Requests", str(result.total_requests))
        table.add_row("Routed", str(result.routed))
        table.add_row("Rejected", str(result.rejected))
        table.add_row("Fallback Used", str(result.fallback_used))
        table.add_row("Avg Decision Latency", f"{result.avg_decision_ms:.2f} ms")
        table.add_row("Avg Predicted TTFT", f"{result.avg_predicted_ttft_ms:.0f} ms")
        table.add_row("Avg Predicted E2E", f"{result.avg_predicted_e2e_ms:.0f} ms")
        table.add_row("Avg Estimated Cost", f"${result.avg_estimated_cost_usd:.6f}")
        table.add_row("Total Estimated Cost", f"${result.total_estimated_cost_usd:.4f}")
        table.add_row("SLO Attainment", f"{result.slo_attainment_rate:.1%}")
        console.print(table)

        if result.endpoint_distribution:
            dist_table = Table(title="Endpoint Distribution")
            dist_table.add_column("Endpoint", style="cyan")
            dist_table.add_column("Requests")
            dist_table.add_column("Share")
            total = sum(result.endpoint_distribution.values())
            for ep_name, count in sorted(
                result.endpoint_distribution.items(), key=lambda x: -x[1]
            ):
                dist_table.add_row(ep_name, str(count), f"{count/total:.1%}")
            console.print(dist_table)

    asyncio.run(_run())


# ---------------------------------------------------------------------------
# policy
# ---------------------------------------------------------------------------


policy_app = typer.Typer(help="Policy management commands")
app.add_typer(policy_app, name="policy")


@policy_app.command("preset")
def set_preset(
    preset: str = typer.Argument(help="Preset name (latency-first/balanced/cost-first)"),
    router_url: str = typer.Option("http://localhost:8080"),
) -> None:
    """Switch the active routing policy preset."""
    import httpx

    try:
        resp = httpx.post(
            f"{router_url}/admin/policy/preset",
            json={"preset": preset},
            timeout=10,
        )
        resp.raise_for_status()
        data = resp.json()
        console.print(f"[green]Preset switched to[/] [bold]{preset}[/]")
        console.print(json.dumps(data, indent=2))
    except Exception as exc:
        console.print(f"[red]Error:[/] {exc}", err=True)
        raise typer.Exit(1)


# ---------------------------------------------------------------------------
# explain
# ---------------------------------------------------------------------------


@app.command()
def explain(
    request_id: str = typer.Argument(help="Request ID to explain"),
    router_url: str = typer.Option("http://localhost:8080"),
) -> None:
    """Explain a past routing decision."""
    import httpx

    try:
        resp = httpx.get(
            f"{router_url}/admin/routes/explain/{request_id}", timeout=10
        )
        resp.raise_for_status()
        data = resp.json()
    except Exception as exc:
        console.print(f"[red]Error:[/] {exc}", err=True)
        raise typer.Exit(1)

    decision = data.get("decision", {})
    profile = data.get("request_profile", {})

    console.print(f"\n[bold]Request[/] {request_id}")
    console.print(f"  Model: {profile.get('model_requested')}")
    console.print(f"  Workload: {profile.get('workload_type')}")
    console.print(f"  Input tokens: {profile.get('input_tokens')}")
    console.print(f"  Output tokens: {profile.get('predicted_output_tokens')}")
    console.print(f"  Priority: {profile.get('priority_tier')}")

    console.print(f"\n[bold]Decision[/]")
    console.print(f"  Selected: [green]{decision.get('selected_endpoint_name')}[/]")
    console.print(f"  Outcome: {decision.get('outcome')}")
    console.print(f"  Decision latency: {decision.get('decision_latency_ms', 0):.2f} ms")
    console.print(f"  Predicted TTFT: {decision.get('predicted_ttft_ms', 0):.0f} ms")
    console.print(f"  Predicted E2E: {decision.get('predicted_e2e_ms', 0):.0f} ms")
    console.print(f"  Estimated cost: ${decision.get('estimated_cost_usd', 0):.6f}")

    candidates = decision.get("candidate_scores", [])
    if candidates:
        table = Table(title="Candidate Scores")
        table.add_column("Endpoint", style="cyan")
        table.add_column("Utility")
        table.add_column("SLO")
        table.add_column("Cost")
        table.add_column("Queue")
        table.add_column("GPU")
        table.add_column("Reliability")
        table.add_column("Status")

        for c in sorted(candidates, key=lambda x: -x.get("utility_score", 0)):
            if c.get("hard_rejected"):
                status = f"[red]rejected: {c.get('rejection_reason', '')}[/]"
                table.add_row(
                    c["endpoint_name"], "—", "—", "—", "—", "—", "—", status
                )
            else:
                table.add_row(
                    c["endpoint_name"],
                    f"{c.get('utility_score', 0):.3f}",
                    f"{c.get('slo_score', 0):.3f}",
                    f"{c.get('cost_score', 0):.3f}",
                    f"{c.get('queue_score', 0):.3f}",
                    f"{c.get('gpu_affinity_score', 0):.3f}",
                    f"{c.get('reliability_score', 0):.3f}",
                    "[green]selected[/]" if c["endpoint_id"] == decision.get("selected_endpoint_id") else "candidate",
                )
        console.print(table)


if __name__ == "__main__":
    app()
