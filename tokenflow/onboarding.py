"""Interactive onboarding wizard.

Walks a user through environment detection, backend registration, policy
choice, and (optionally) deployment — without ever asking them to hand-edit
YAML. Outputs:

  - .env                        (TOKENFLOW_* environment overrides)
  - examples/configs/policy.yaml  (the routing policy)
  - .tokenflow/onboarding.json  (replay metadata, also used by `tokenflow
                                 init --resume` if user bails midway)

Run with `tokenflow init` (registered as a Typer subcommand in cli.py).
The wizard is deliberately resumable — every answered question is
serialised so you can re-run without losing state.
"""
from __future__ import annotations

import json
import os
import platform
import shutil
import subprocess
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import typer
import yaml
from rich.console import Console
from rich.panel import Panel
from rich.prompt import Confirm, IntPrompt, Prompt
from rich.table import Table

console = Console()


# ---------------------------------------------------------------------------
# Data model — everything the wizard collects
# ---------------------------------------------------------------------------


@dataclass
class Endpoint:
    name: str
    backend_type: str   # nim | vllm | sglang | dynamo | ollama
    nim_url: str        # base URL
    model_name: str
    gpu_name: str
    cost_class: str
    cost_per_gpu_hour: float
    max_context_tokens: int
    supports_reasoning: bool = False


@dataclass
class TenantPolicy:
    name: str
    allowed_gpu_classes: list[str]
    max_rpm: int
    budget_usd_per_hour: float
    priority_tier_override: str = ""


@dataclass
class OnboardingState:
    deployment_target: str = "docker"  # docker | kubernetes | bare-metal
    cluster_kind: str = ""             # eks | aks | gke | k3s | minikube
    policy_preset: str = "balanced"    # latency-first | balanced | cost-first
    enable_spot: bool = False
    spot_provider: str = ""            # aws | azure | gcp
    enable_dormant: bool = False
    endpoints: list[Endpoint] = field(default_factory=list)
    tenants: list[TenantPolicy] = field(default_factory=list)
    completed_steps: list[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Environment detection
# ---------------------------------------------------------------------------


def detect_environment() -> dict[str, Any]:
    """Probe the host for tools the user already has."""
    env = {
        "platform": platform.system(),
        "docker": shutil.which("docker") is not None,
        "docker_compose": _docker_compose_available(),
        "kubectl": shutil.which("kubectl") is not None,
        "helm": shutil.which("helm") is not None,
        "aws": shutil.which("aws") is not None,
        "az": shutil.which("az") is not None,
        "gcloud": shutil.which("gcloud") is not None,
        "nvidia_smi": shutil.which("nvidia-smi") is not None,
        "gpu_count": _detect_gpu_count(),
        "current_k8s_context": _current_k8s_context(),
    }
    return env


def _docker_compose_available() -> bool:
    if shutil.which("docker") is None:
        return False
    try:
        out = subprocess.run(
            ["docker", "compose", "version"], capture_output=True, timeout=2,
        )
        return out.returncode == 0
    except Exception:
        return False


def _detect_gpu_count() -> int:
    if shutil.which("nvidia-smi") is None:
        return 0
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
            capture_output=True, text=True, timeout=2,
        )
        if out.returncode != 0:
            return 0
        return len([line for line in out.stdout.strip().splitlines() if line.strip()])
    except Exception:
        return 0


def _current_k8s_context() -> str:
    if shutil.which("kubectl") is None:
        return ""
    try:
        out = subprocess.run(
            ["kubectl", "config", "current-context"],
            capture_output=True, text=True, timeout=2,
        )
        return out.stdout.strip() if out.returncode == 0 else ""
    except Exception:
        return ""


# ---------------------------------------------------------------------------
# Wizard steps
# ---------------------------------------------------------------------------


def banner() -> None:
    console.print(
        Panel.fit(
            "[bold cyan]TokenFlow Router — interactive setup[/bold cyan]\n"
            "[dim]No YAML required. Resume anytime with `tokenflow init --resume`.[/dim]",
            border_style="cyan",
        )
    )


def step_environment(state: OnboardingState, env: dict[str, Any]) -> None:
    table = Table(title="Detected environment", show_header=False, box=None)
    table.add_column("", style="cyan")
    table.add_column("")
    for label, key in [
        ("Platform", "platform"),
        ("Docker", "docker"),
        ("Docker Compose v2", "docker_compose"),
        ("kubectl", "kubectl"),
        ("Helm", "helm"),
        ("AWS CLI", "aws"),
        ("Azure CLI", "az"),
        ("gcloud", "gcloud"),
        ("nvidia-smi", "nvidia_smi"),
        ("GPU count", "gpu_count"),
        ("Current k8s context", "current_k8s_context"),
    ]:
        val = env.get(key)
        if isinstance(val, bool):
            val = "[green]yes[/green]" if val else "[dim]not found[/dim]"
        elif val is None or val == "":
            val = "[dim]–[/dim]"
        table.add_row(label, str(val))
    console.print(table)


def step_deployment_target(state: OnboardingState, env: dict[str, Any]) -> None:
    console.rule("[bold]Where do you want to run the router?[/bold]")
    options = ["docker"]
    if env.get("kubectl"):
        options.append("kubernetes")
    options.append("bare-metal")

    choice = Prompt.ask(
        "Deployment target",
        choices=options,
        default=state.deployment_target if state.deployment_target in options else options[0],
    )
    state.deployment_target = choice

    if choice == "kubernetes":
        ctx_default = env.get("current_k8s_context") or ""
        kinds = ["eks", "aks", "gke", "k3s", "minikube", "other"]
        state.cluster_kind = Prompt.ask(
            "Cluster type",
            choices=kinds,
            default=state.cluster_kind or "eks",
        )
        if ctx_default:
            console.print(f"  ↳ will deploy into kubectl context: [cyan]{ctx_default}[/cyan]")
        if not env.get("helm"):
            console.print(
                "[yellow]warn:[/yellow] helm not found on PATH. "
                "The wizard will still write the chart, but you'll need to install helm to apply it."
            )


def step_policy_preset(state: OnboardingState) -> None:
    console.rule("[bold]Which routing policy preset?[/bold]")
    table = Table(show_header=True, header_style="bold cyan")
    table.add_column("Preset")
    table.add_column("Use it when…")
    table.add_row("latency-first", "interactive / chat / SLO-bound traffic — minimise p95")
    table.add_row("balanced", "general-purpose, mixed workloads (default)")
    table.add_row("cost-first", "batch / offline / agentic pipelines — minimise spend")
    console.print(table)

    state.policy_preset = Prompt.ask(
        "Preset",
        choices=["latency-first", "balanced", "cost-first"],
        default=state.policy_preset,
    )


def step_endpoints(state: OnboardingState) -> None:
    console.rule("[bold]Register backend endpoints[/bold]")
    console.print(
        "[dim]At least one endpoint is required. You can add more later via "
        "POST /admin/endpoints or by re-running this wizard.[/dim]"
    )

    if state.endpoints:
        console.print(f"[green]Already registered: {len(state.endpoints)}[/green]")
        if not Confirm.ask("Add another?", default=False):
            return

    while True:
        ep = _prompt_endpoint(default_index=len(state.endpoints) + 1)
        state.endpoints.append(ep)
        if not Confirm.ask("Add another endpoint?", default=False):
            break


def _prompt_endpoint(default_index: int) -> Endpoint:
    name = Prompt.ask("  Name", default=f"backend-{default_index}")
    backend_type = Prompt.ask(
        "  Backend type", choices=["nim", "vllm", "sglang", "dynamo", "ollama"], default="vllm",
    )
    default_url = {
        "nim": "http://localhost:8000",
        "vllm": "http://localhost:8000",
        "sglang": "http://localhost:30000",
        "dynamo": "http://localhost:8000",
        "ollama": "http://localhost:11434",
    }[backend_type]
    nim_url = Prompt.ask("  Base URL", default=default_url)
    model_name = Prompt.ask("  Model name", default="meta/llama-3.1-8b-instruct")
    gpu_name = Prompt.ask(
        "  GPU class",
        choices=["B200", "H200", "H100", "A100", "L40S", "L4", "A10G", "RTX_PRO_6000", "CPU"],
        default="H100",
    )
    cost_class = Prompt.ask(
        "  Cost class",
        choices=["premium", "standard", "economy"],
        default="standard",
    )
    cost_default = {"premium": 8.0, "standard": 4.0, "economy": 2.5}[cost_class]
    cost = float(Prompt.ask(f"  $/GPU-hour", default=str(cost_default)))
    ctx = IntPrompt.ask("  max_context_tokens", default=8192)
    reasoning = Confirm.ask("  Supports reasoning workloads?", default=cost_class == "premium")
    return Endpoint(
        name=name, backend_type=backend_type, nim_url=nim_url,
        model_name=model_name, gpu_name=gpu_name, cost_class=cost_class,
        cost_per_gpu_hour=cost, max_context_tokens=ctx, supports_reasoning=reasoning,
    )


def step_advanced(state: OnboardingState, env: dict[str, Any]) -> None:
    console.rule("[bold]Advanced (optional)[/bold]")

    state.enable_dormant = Confirm.ask(
        "Enable dormant-backend auto-spinup? "
        "(register backends as profiles instead of always-on; saves GPU-hours)",
        default=False,
    )

    state.enable_spot = Confirm.ask(
        "Enable spot / preemptible instance support for capacity scaling?",
        default=False,
    )
    if state.enable_spot:
        providers = []
        if env.get("aws"):
            providers.append("aws")
        if env.get("az"):
            providers.append("azure")
        if env.get("gcloud"):
            providers.append("gcp")
        if not providers:
            providers = ["aws", "azure", "gcp"]
            console.print(
                "[yellow]warn:[/yellow] no cloud CLI detected. "
                "Wizard will still write spot config; you'll need credentials at runtime."
            )
        state.spot_provider = Prompt.ask("Spot provider", choices=providers, default=providers[0])


# ---------------------------------------------------------------------------
# Output renderers
# ---------------------------------------------------------------------------


PRESET_WEIGHTS = {
    "latency-first":   dict(slo=0.50, cost=0.05, queue=0.15, gpu_affinity=0.10, model_fit=0.10, reliability=0.10),
    "balanced":        dict(slo=0.30, cost=0.20, queue=0.15, gpu_affinity=0.15, model_fit=0.10, reliability=0.10),
    "cost-first":      dict(slo=0.15, cost=0.45, queue=0.10, gpu_affinity=0.10, model_fit=0.10, reliability=0.10),
}


def render_policy_yaml(state: OnboardingState) -> str:
    weights = PRESET_WEIGHTS[state.policy_preset]
    policy: dict[str, Any] = {
        "name": f"onboarded-{state.policy_preset}",
        "description": "Generated by `tokenflow init`. Edit freely.",
        "preset": state.policy_preset,
        "slo_weight": weights["slo"],
        "cost_weight": weights["cost"],
        "queue_weight": weights["queue"],
        "gpu_affinity_weight": weights["gpu_affinity"],
        "model_fit_weight": weights["model_fit"],
        "reliability_weight": weights["reliability"],
        "slo_ttft_ms": 500,
        "slo_itl_ms": 50,
        "slo_e2e_ms": 5000,
        "max_queue_depth": 100,
        "min_health_score": 0.5,
        "max_error_rate": 0.10,
        "rules": [
            dict(name="premium-traffic-on-premium-gpu", priority=10,
                 conditions=dict(priority_tier="premium"),
                 actions=dict(set_budget_sensitivity=0.0)),
            dict(name="batch-to-economy-lane", priority=20,
                 conditions=dict(priority_tier="batch"),
                 actions=dict(set_budget_sensitivity=1.0)),
        ],
    }
    if state.tenants:
        policy["tenant_policies"] = {
            t.name: {
                "allowed_gpu_classes": t.allowed_gpu_classes,
                "max_rpm": t.max_rpm,
                "budget_usd_per_hour": t.budget_usd_per_hour,
                **({"priority_tier_override": t.priority_tier_override}
                   if t.priority_tier_override else {}),
            }
            for t in state.tenants
        }
    return yaml.safe_dump(policy, sort_keys=False, default_flow_style=False)


def render_env_file(state: OnboardingState) -> str:
    lines = [
        "# TokenFlow Router — environment overrides (generated by `tokenflow init`)",
        "TOKENFLOW_HOST=0.0.0.0",
        "TOKENFLOW_PORT=8080",
        "TOKENFLOW_LOG_LEVEL=INFO",
        f"TOKENFLOW_DEFAULT_POLICY={state.policy_preset}",
        "TOKENFLOW_POLICY_FILE=/app/configs/policy.yaml",
        "TOKENFLOW_ENABLE_METRICS=true",
        "TOKENFLOW_ENDPOINT_WARMUP_GRACE_S=120",
    ]
    if state.enable_dormant:
        lines.append("# dormant-profile flow enabled — see examples/autoscale/")
    if state.enable_spot:
        lines.append(f"# spot capacity provider configured: {state.spot_provider}")
    return "\n".join(lines) + "\n"


def render_register_script(state: OnboardingState) -> str:
    """Emit a small bash script that POSTs each endpoint after the router is up."""
    lines = [
        "#!/usr/bin/env bash",
        "# Generated by `tokenflow init`. Re-run any time after the router is up.",
        "set -euo pipefail",
        'ROUTER="${ROUTER:-http://localhost:8080}"',
        "",
    ]
    for ep in state.endpoints:
        body = json.dumps({
            "name": ep.name,
            "nim_url": ep.nim_url,
            "backend_type": ep.backend_type,
            "model_name": ep.model_name,
            "gpu_name": ep.gpu_name,
            "cost_class": ep.cost_class,
            "cost_per_gpu_hour": ep.cost_per_gpu_hour,
            "max_context_tokens": ep.max_context_tokens,
            "supports_reasoning": ep.supports_reasoning,
        })
        target = "endpoints" if not state.enable_dormant else "profiles"
        lines.append(f'echo "==> Registering {ep.name} ({ep.backend_type}) on {ep.gpu_name}"')
        lines.append(
            f"curl -s -X POST \"$ROUTER/admin/{target}\" "
            f"-H 'Content-Type: application/json' "
            f"-d '{body}' | python3 -m json.tool | head -3"
        )
        lines.append("")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Public entry point — called from cli.py `tokenflow init`
# ---------------------------------------------------------------------------


def run_onboarding(
    workdir: Path,
    apply: bool = False,
    resume: bool = False,
) -> OnboardingState:
    """Run the full wizard. If `apply` is True, also bring up the router."""
    workdir = workdir.resolve()
    workdir.mkdir(parents=True, exist_ok=True)
    state_path = workdir / ".tokenflow" / "onboarding.json"

    if resume and state_path.exists():
        raw = json.loads(state_path.read_text())
        state = OnboardingState(
            **{k: v for k, v in raw.items()
               if k not in ("endpoints", "tenants")},
        )
        state.endpoints = [Endpoint(**e) for e in raw.get("endpoints", [])]
        state.tenants = [TenantPolicy(**t) for t in raw.get("tenants", [])]
        console.print(f"[green]resumed[/green] from {state_path}")
    else:
        state = OnboardingState()

    banner()
    env = detect_environment()
    step_environment(state, env)
    step_deployment_target(state, env)
    step_policy_preset(state)
    step_endpoints(state)
    step_advanced(state, env)

    # Persist replay state
    state_path.parent.mkdir(parents=True, exist_ok=True)
    state_path.write_text(json.dumps(asdict(state), indent=2))

    # Render outputs
    policy_path = workdir / "examples" / "configs" / "policy.yaml"
    env_path = workdir / ".env"
    register_path = workdir / ".tokenflow" / "register_endpoints.sh"

    policy_path.parent.mkdir(parents=True, exist_ok=True)
    policy_path.write_text(render_policy_yaml(state))
    env_path.write_text(render_env_file(state))
    register_path.write_text(render_register_script(state))
    register_path.chmod(0o755)

    console.print()
    console.print(
        Panel.fit(
            f"[green]✓ wrote[/green] {policy_path.relative_to(workdir)}\n"
            f"[green]✓ wrote[/green] {env_path.relative_to(workdir)}\n"
            f"[green]✓ wrote[/green] {register_path.relative_to(workdir)}\n"
            f"[green]✓ saved state[/green] {state_path.relative_to(workdir)}",
            title="generated",
            border_style="green",
        )
    )

    # Next-steps
    target = state.deployment_target
    if target == "docker":
        console.print(
            Panel(
                "[bold]Next:[/bold]\n"
                "  [cyan]docker compose up -d[/cyan]                # start router + mocks\n"
                "  [cyan].tokenflow/register_endpoints.sh[/cyan]    # register the backends you just configured\n"
                "  [cyan]curl http://localhost:8080/health[/cyan]\n"
                "  [cyan]open http://localhost:8080/docs[/cyan]     # OpenAPI swagger UI",
                title="docker",
            )
        )
        if apply and env.get("docker"):
            _run_docker_apply(workdir)
    elif target == "kubernetes":
        ns = "tokenflow"
        console.print(
            Panel(
                f"[bold]Next:[/bold]\n"
                f"  [cyan]kubectl create namespace {ns}[/cyan]\n"
                f"  [cyan]helm install tokenflow deploy/k8s/helm/tokenflow-router \\\n"
                f"    --namespace {ns} \\\n"
                f"    --set-file policy.content={policy_path}[/cyan]\n"
                f"  [cyan]kubectl -n {ns} port-forward svc/tokenflow 8080:8080 &[/cyan]\n"
                f"  [cyan].tokenflow/register_endpoints.sh[/cyan]",
                title=f"kubernetes ({state.cluster_kind or 'generic'})",
            )
        )
    else:
        console.print(
            Panel(
                "[bold]Next:[/bold]\n"
                "  [cyan]pip install -e .[/cyan]\n"
                "  [cyan]tokenflow serve --policy-file examples/configs/policy.yaml[/cyan]\n"
                "  [cyan].tokenflow/register_endpoints.sh[/cyan]",
                title="bare-metal",
            )
        )

    if state.enable_spot:
        console.print(
            "[yellow]Reminder:[/yellow] spot support is wired through "
            "[cyan]examples/autoscale/spot_adapter.py[/cyan]. Set your "
            f"{state.spot_provider.upper()} credentials before the controller can launch instances."
        )

    return state


def _run_docker_apply(workdir: Path) -> None:
    if not Confirm.ask("Bring up the router via `docker compose up -d` now?", default=False):
        return
    subprocess.run(["docker", "compose", "up", "-d"], cwd=workdir, check=False)
    register = workdir / ".tokenflow" / "register_endpoints.sh"
    if register.exists() and Confirm.ask("Register endpoints now?", default=True):
        subprocess.run(["bash", str(register)], check=False)
