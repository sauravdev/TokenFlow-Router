"""Minimal capacity controller for v6 benchmark.

Polls TokenFlow's /admin/profiles every 2s. For each template it sees:
  - If activated=true and the docker container isn't running → `docker start <container>`
  - If activated=false and the container is running → `docker stop <container>`

Emits one-line status updates on stdout.
"""
from __future__ import annotations

import json
import subprocess
import sys
import time
import urllib.request


TOKENFLOW_URL = "http://localhost:8080"
POLL_INTERVAL_S = 2.0

# Map template name → docker container name
TEMPLATE_TO_CONTAINER = {
    "vllm-prefill-dormant": "vllm-prefill",
}


def fetch_profiles() -> list[dict]:
    try:
        with urllib.request.urlopen(f"{TOKENFLOW_URL}/admin/profiles", timeout=3) as r:
            return json.load(r)
    except Exception as e:
        print(f"[ctl] fetch failed: {e}", flush=True)
        return []


def container_is_running(name: str) -> bool:
    r = subprocess.run(
        ["docker", "ps", "--filter", f"name=^{name}$", "--format", "{{.Names}}"],
        capture_output=True, text=True, timeout=5,
    )
    return name in r.stdout


def docker_action(verb: str, name: str) -> None:
    print(f"[ctl] {verb} {name}", flush=True)
    r = subprocess.run(["docker", verb, name], capture_output=True, text=True, timeout=30)
    if r.returncode != 0:
        print(f"[ctl]   → ERR: {r.stderr.strip()}", flush=True)
    else:
        print(f"[ctl]   → OK", flush=True)


def main() -> None:
    print(f"[ctl] starting, polling {TOKENFLOW_URL}/admin/profiles every {POLL_INTERVAL_S}s", flush=True)
    last_state: dict[str, bool] = {}
    while True:
        profiles = fetch_profiles()
        for p in profiles:
            name = p.get("name")
            if name not in TEMPLATE_TO_CONTAINER:
                continue
            activated = bool(p.get("activated"))
            container = TEMPLATE_TO_CONTAINER[name]
            running = container_is_running(container)
            # Edge detection to log transitions
            prev = last_state.get(name)
            if prev != activated:
                print(f"[ctl] template {name}: activated={activated} (container running={running})", flush=True)
                last_state[name] = activated
            if activated and not running:
                docker_action("start", container)
            elif not activated and running:
                docker_action("stop", container)
        time.sleep(POLL_INTERVAL_S)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit(0)
