#!/usr/bin/env python3
"""Sample controller that watches TokenFlow profile flags and starts/stops engines.

This is intentionally a *sample*, not a production autoscaler.

What it watches
---------------
- `/admin/profiles` -> template `activated` state is treated as desired state
- optional response headers from callers such as:
  - `X-TokenFlow-Active-Backend`
  - `X-TokenFlow-Active-Endpoint`
  - `X-TokenFlow-Turn-Down-Candidates`

What it does
------------
- if a profile template becomes `activated`, ensure the corresponding backend
  process / endpoint is up
- if a profile template becomes deactivated, stop the corresponding backend
  if configured to do so

Current engine hooks
--------------------
- vLLM: default `vllm serve ...` command pattern
- SGLang: default `python -m sglang.launch_server ...` command pattern
- Ollama: default `ollama pull` + `ollama stop` pattern
- NIM / Dynamo: deployment-specific in practice, so this sample expects explicit
  `start_command` / `stop_command` in config for those engines

Suggested documentation to verify against in your environment
-------------------------------------------------------------
- vLLM OpenAI-compatible server docs
- SGLang launch/server docs
- Ollama API / CLI docs
- NVIDIA NIM getting-started / deployment docs
- NVIDIA Dynamo quickstart / deployment docs

I could not live-fetch those docs from this sandbox while authoring this sample due
network/DNS restrictions, so treat the built-in defaults as **reference examples**
and verify the exact commands/flags against the latest vendor docs before running
outside a dev environment.

Because deployment topologies vary wildly, the safest interface is a config file
that maps TokenFlow template names to concrete engine commands.
"""

from __future__ import annotations

import argparse
import json
import os
import shlex
import subprocess
import sys
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Any


@dataclass
class EngineTarget:
    template_name: str
    backend_type: str
    model_name: str
    base_url: str
    start_command: list[str] | None = None
    stop_command: list[str] | None = None
    healthcheck_url: str | None = None
    working_dir: str | None = None
    env: dict[str, str] | None = None
    stop_when_inactive: bool = True


class TokenFlowCapacityController:
    def __init__(
        self,
        tokenflow_url: str,
        config: dict[str, Any],
        admin_api_key: str | None,
        interval_seconds: int,
        dry_run: bool,
    ) -> None:
        self.tokenflow_url = tokenflow_url.rstrip("/")
        self.admin_api_key = admin_api_key
        self.interval_seconds = interval_seconds
        self.dry_run = dry_run
        self.targets = {
            item["template_name"]: EngineTarget(**item)
            for item in config["targets"]
        }
        self.processes: dict[str, subprocess.Popen] = {}

    def run_forever(self) -> None:
        print("Starting TokenFlow capacity controller")
        print(f"TokenFlow: {self.tokenflow_url}")
        while True:
            try:
                profiles = self.fetch_profiles()
                self.reconcile(profiles)
            except Exception as exc:
                print(f"[warn] reconcile failed: {exc}", file=sys.stderr)
            time.sleep(self.interval_seconds)

    def fetch_profiles(self) -> list[dict[str, Any]]:
        req = urllib.request.Request(f"{self.tokenflow_url}/admin/profiles")
        if self.admin_api_key:
            req.add_header("X-Admin-API-Key", self.admin_api_key)
        with urllib.request.urlopen(req, timeout=10) as resp:
            return json.loads(resp.read().decode("utf-8"))

    def reconcile(self, profiles: list[dict[str, Any]]) -> None:
        for profile in profiles:
            template_name = profile["name"]
            target = self.targets.get(template_name)
            if target is None:
                continue

            desired_active = bool(profile.get("activated", False))
            if desired_active:
                self.ensure_running(target, profile)
            elif target.stop_when_inactive:
                self.ensure_stopped(target, profile)

    def ensure_running(self, target: EngineTarget, profile: dict[str, Any]) -> None:
        if self.healthcheck_ok(target):
            return
        cmd = self.render_command(target, profile, action="start")
        if not cmd:
            print(f"[warn] no start command for {target.template_name}")
            return
        self.exec_command(cmd, target, prefix="start")

    def ensure_stopped(self, target: EngineTarget, profile: dict[str, Any]) -> None:
        cmd = self.render_command(target, profile, action="stop")
        if not cmd:
            return
        self.exec_command(cmd, target, prefix="stop")

    def healthcheck_ok(self, target: EngineTarget) -> bool:
        if not target.healthcheck_url:
            return False
        try:
            with urllib.request.urlopen(target.healthcheck_url, timeout=3) as resp:
                return 200 <= resp.status < 300
        except Exception:
            return False

    def render_command(
        self,
        target: EngineTarget,
        profile: dict[str, Any],
        action: str,
    ) -> list[str] | None:
        explicit = target.start_command if action == "start" else target.stop_command
        variables = {
            "model": target.model_name,
            "url": target.base_url,
            "template_name": target.template_name,
            "backend": target.backend_type,
            "gpu": profile.get("gpu_name", "UNKNOWN"),
        }
        if explicit:
            return [part.format(**variables) for part in explicit]

        if action == "start":
            if target.backend_type == "vllm":
                return [
                    "vllm",
                    "serve",
                    target.model_name,
                    "--host",
                    "0.0.0.0",
                    "--port",
                    self._port_from_url(target.base_url),
                ]
            if target.backend_type == "sglang":
                return [
                    sys.executable,
                    "-m",
                    "sglang.launch_server",
                    "--model-path",
                    target.model_name,
                    "--host",
                    "0.0.0.0",
                    "--port",
                    self._port_from_url(target.base_url),
                ]
            if target.backend_type == "ollama":
                return ["ollama", "pull", target.model_name]
            # NIM and Dynamo are usually deployment-specific (docker/k8s/helm/systemd)
            return None

        if action == "stop":
            if target.backend_type == "ollama":
                return ["ollama", "stop", target.model_name]
            # For vLLM/SGLang/NIM/Dynamo, stop hooks are highly deployment-specific.
            return None

        return None

    def exec_command(self, cmd: list[str], target: EngineTarget, prefix: str) -> None:
        pretty = shlex.join(cmd)
        print(f"[{prefix}] {target.template_name}: {pretty}")
        if self.dry_run:
            return
        env = os.environ.copy()
        if target.env:
            env.update(target.env)
        subprocess.Popen(
            cmd,
            cwd=target.working_dir,
            env=env,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )

    @staticmethod
    def _port_from_url(url: str) -> str:
        if ":" not in url.rsplit("/", 1)[-1]:
            return "8000"
        return url.rsplit(":", 1)[-1].rstrip("/")


def load_config(path: str) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def main() -> int:
    parser = argparse.ArgumentParser(description="TokenFlow sample capacity controller")
    parser.add_argument("--tokenflow-url", default="http://localhost:8080")
    parser.add_argument("--config", required=True)
    parser.add_argument("--admin-api-key", default=os.getenv("TOKENFLOW_ADMIN_API_KEY"))
    parser.add_argument("--interval-seconds", type=int, default=10)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    config = load_config(args.config)
    controller = TokenFlowCapacityController(
        tokenflow_url=args.tokenflow_url,
        config=config,
        admin_api_key=args.admin_api_key,
        interval_seconds=args.interval_seconds,
        dry_run=args.dry_run,
    )
    controller.run_forever()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
