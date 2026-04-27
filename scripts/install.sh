#!/usr/bin/env bash
# TokenFlow Router — one-click installer.
#
#   curl -fsSL https://raw.githubusercontent.com/sauravdev/TokenFlow-Router/main/scripts/install.sh | bash
#
# or, after cloning:
#
#   ./scripts/install.sh
#
# What it does:
#   1. Verifies python3 + (optionally) docker
#   2. Creates a venv and pip-installs the package in editable mode
#   3. Runs the interactive onboarding wizard (`tokenflow init`)
#   4. Optionally brings the router up via `docker compose up -d`
#
# Re-running is safe — the wizard supports `--resume` and won't clobber
# existing config unless you confirm.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(dirname "$SCRIPT_DIR")"
cd "$REPO_DIR"

# ── pretty-print helpers ────────────────────────────────────────────────────
bold()    { printf "\033[1m%s\033[0m" "$1"; }
green()   { printf "\033[32m%s\033[0m" "$1"; }
yellow()  { printf "\033[33m%s\033[0m" "$1"; }
red()     { printf "\033[31m%s\033[0m" "$1"; }
hr()      { printf -- '─%.0s' {1..72}; echo; }

step() { hr; echo "$(bold "==>") $1"; }

# ── 0. quick sanity ────────────────────────────────────────────────────────
step "Checking host"
if ! command -v python3 >/dev/null 2>&1; then
    echo "  $(red 'missing:') python3 — install python ≥ 3.11 first"
    exit 1
fi
PY_VERSION=$(python3 -c 'import sys; print(f"{sys.version_info[0]}.{sys.version_info[1]}")')
echo "  $(green '✓') python3 ($PY_VERSION)"

if command -v docker >/dev/null 2>&1; then
    echo "  $(green '✓') docker"
    HAS_DOCKER=1
else
    echo "  $(yellow '–') docker not found (you can still bare-metal run, or use the helm chart on a cluster)"
    HAS_DOCKER=0
fi

# ── 1. venv + pip install ──────────────────────────────────────────────────
step "Setting up Python venv at .venv"
if [[ ! -d .venv ]]; then
    python3 -m venv .venv
fi
# shellcheck disable=SC1091
source .venv/bin/activate

echo "  installing tokenflow + extras (this may take ~30s on first run)"
pip install --quiet --upgrade pip
pip install --quiet -e ".[dev]"
echo "  $(green '✓') tokenflow CLI is on \$PATH"

# ── 2. interactive wizard ──────────────────────────────────────────────────
step "Launching onboarding wizard"
echo "  ($(yellow 'tip:') re-run with $(bold './scripts/install.sh --resume') to pick up where you left off)"
echo

WIZARD_ARGS=()
if [[ "${1:-}" == "--resume" ]]; then
    WIZARD_ARGS+=(--resume)
fi
if [[ "$HAS_DOCKER" == "1" ]]; then
    WIZARD_ARGS+=(--apply)
fi

tokenflow init "${WIZARD_ARGS[@]}" --workdir "$REPO_DIR"

# ── 3. final pointer ───────────────────────────────────────────────────────
step "Done"
cat <<EOF
  $(bold 'Quick reference:')
    activate the venv ........  source .venv/bin/activate
    start the router .........  docker compose up -d   (if you skipped --apply)
    register backends ........  .tokenflow/register_endpoints.sh
    health check .............  curl http://localhost:8080/health
    swagger UI ...............  http://localhost:8080/docs
    raw demos ................  examples/demo/

  $(bold 'Need a different deployment target?')
    Kubernetes (EKS/AKS/GKE) .  helm install tokenflow deploy/k8s/helm/tokenflow-router
    Spot/preemptible scaling .  examples/autoscale/spot_adapter.py
EOF
