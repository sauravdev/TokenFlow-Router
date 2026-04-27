"""Spot / preemptible instance adapters for the TokenFlow capacity controller.

Three cloud providers, one interface. Each adapter knows how to:

  - request a spot/preemptible instance (start)
  - cancel / terminate it (stop)
  - check whether the cloud has decided to evict it (preemption signal)

When you set `enable_spot=true` in `tokenflow init`, this module is wired
into the capacity controller's start/stop hooks. The controller polls
`/admin/profiles` for `activated=true` flags and calls into here.

Production note
---------------
This file is the **interface and reference implementation**. Every
provider's spot APIs have edge cases (capacity-pool exhaustion, AZ
fallback, hibernation vs termination, IMDSv2 vs v1, regional vs zonal
quotas). Validate the exact CLI flags and IAM permissions against the
latest vendor docs before running in production. The adapter shells out
to the vendor CLI rather than embedding boto3/azure-sdk so we don't
add heavyweight cloud SDKs as runtime dependencies.

Cloud-CLI prerequisites
-----------------------
  AWS:    `aws` CLI configured (`aws configure`) with iam:RunInstances,
          ec2:RequestSpotInstances, ec2:TerminateInstances, etc.
  Azure:  `az login` + `az account set`. Service principal recommended
          for unattended use.
  GCP:    `gcloud auth login` + `gcloud config set project`. SA JSON
          via GOOGLE_APPLICATION_CREDENTIALS for unattended use.
"""
from __future__ import annotations

import json
import logging
import shutil
import subprocess
import time
from dataclasses import dataclass, field
from typing import Any, Optional

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Common config
# ---------------------------------------------------------------------------


@dataclass
class SpotInstanceTarget:
    """Configuration for a single spot/preemptible engine target."""

    template_name: str
    backend_type: str               # vllm | nim | sglang | dynamo | ollama
    model_name: str
    base_url_template: str          # e.g. "http://{instance_ip}:8000"
    healthcheck_path: str = "/health"

    # ── Cloud selection ────────────────────────────────────────────────
    provider: str = "aws"           # aws | azure | gcp
    region: str = ""                # required
    availability_zone: str = ""

    # ── Instance shape ────────────────────────────────────────────────
    instance_type: str = ""         # e.g. g5.2xlarge / Standard_NC24ads_A100_v4 / a2-highgpu-1g
    image_id: str = ""              # AMI / image / managed-image
    keypair_name: str = ""
    security_groups: list[str] = field(default_factory=list)
    subnet_id: str = ""
    iam_instance_profile: str = ""  # AWS only

    # ── Spot economics ────────────────────────────────────────────────
    max_price_usd_per_hour: float = 0.0  # 0 = market price
    interruption_behavior: str = "terminate"  # terminate | stop | hibernate (AWS)

    # ── Bring-up ──────────────────────────────────────────────────────
    user_data_script: str = ""      # shell script run on first boot
    boot_timeout_s: int = 600       # how long to wait for healthcheck

    # ── Stop policy ───────────────────────────────────────────────────
    stop_when_inactive: bool = True
    cooldown_seconds: int = 60      # min time between stop/start of same target


@dataclass
class SpotInstanceState:
    """Runtime state tracked per target."""

    instance_id: Optional[str] = None
    instance_ip: Optional[str] = None
    last_started_at: float = 0.0
    last_stopped_at: float = 0.0
    last_preemption_check_at: float = 0.0


# ---------------------------------------------------------------------------
# Public protocol — what the capacity controller calls
# ---------------------------------------------------------------------------


class SpotAdapter:
    """Base class. Subclass for each provider."""

    name: str = "base"

    def __init__(self, target: SpotInstanceTarget) -> None:
        self.target = target
        self.state = SpotInstanceState()
        self._sanity_check()

    def _sanity_check(self) -> None:
        if not shutil.which(self._cli):
            logger.warning(
                "spot adapter %s: %s CLI not found on PATH. Spot ops will fail at runtime.",
                self.name, self._cli,
            )

    @property
    def _cli(self) -> str:
        return {"aws": "aws", "azure": "az", "gcp": "gcloud"}[self.name]

    # ── Lifecycle hooks called from the capacity controller ───────────

    def start(self) -> bool:
        """Request a spot instance. Returns True if at least the request
        succeeded; the controller waits for healthcheck separately."""
        raise NotImplementedError

    def stop(self) -> bool:
        """Terminate / cancel the spot instance."""
        raise NotImplementedError

    def is_preempted(self) -> bool:
        """Has the cloud signaled imminent preemption? Called periodically;
        if True, the controller pre-emptively re-routes traffic."""
        return False

    def base_url(self) -> Optional[str]:
        """Render the configured base_url_template with current state."""
        if not self.state.instance_ip:
            return None
        return self.target.base_url_template.format(
            instance_ip=self.state.instance_ip,
            instance_id=self.state.instance_id or "",
        )

    # ── helpers ────────────────────────────────────────────────────────

    def _run(self, argv: list[str], timeout: int = 60) -> dict[str, Any]:
        """Run a CLI command and parse JSON output. Raise on non-zero exit."""
        logger.debug("running: %s", " ".join(argv))
        out = subprocess.run(argv, capture_output=True, text=True, timeout=timeout)
        if out.returncode != 0:
            logger.error("%s failed: %s", argv[0], out.stderr.strip())
            raise RuntimeError(f"{argv[0]} failed: {out.stderr.strip()}")
        if not out.stdout.strip():
            return {}
        try:
            return json.loads(out.stdout)
        except json.JSONDecodeError:
            return {"raw": out.stdout}


# ---------------------------------------------------------------------------
# AWS spot adapter
# ---------------------------------------------------------------------------


class AWSSpotAdapter(SpotAdapter):
    """Uses `aws ec2 run-instances --instance-market-options Spot`."""

    name = "aws"

    def start(self) -> bool:
        t = self.target
        argv = [
            "aws", "ec2", "run-instances",
            "--region", t.region,
            "--instance-type", t.instance_type,
            "--image-id", t.image_id,
            "--count", "1",
            "--instance-market-options",
            json.dumps({
                "MarketType": "spot",
                "SpotOptions": {
                    "InstanceInterruptionBehavior": t.interruption_behavior,
                    **({"MaxPrice": str(t.max_price_usd_per_hour)} if t.max_price_usd_per_hour > 0 else {}),
                },
            }),
            "--tag-specifications",
            f"ResourceType=instance,Tags=[{{Key=tokenflow:template,Value={t.template_name}}}]",
            "--output", "json",
        ]
        if t.keypair_name:
            argv.extend(["--key-name", t.keypair_name])
        if t.subnet_id:
            argv.extend(["--subnet-id", t.subnet_id])
        if t.security_groups:
            argv.extend(["--security-group-ids", *t.security_groups])
        if t.iam_instance_profile:
            argv.extend(["--iam-instance-profile", f"Name={t.iam_instance_profile}"])
        if t.user_data_script:
            argv.extend(["--user-data", t.user_data_script])

        result = self._run(argv, timeout=120)
        instances = result.get("Instances", [])
        if not instances:
            logger.error("aws run-instances returned no instances: %s", result)
            return False
        inst = instances[0]
        self.state.instance_id = inst.get("InstanceId")
        self.state.instance_ip = inst.get("PrivateIpAddress")
        self.state.last_started_at = time.time()
        logger.info("aws spot started: id=%s ip=%s", self.state.instance_id, self.state.instance_ip)
        return True

    def stop(self) -> bool:
        if not self.state.instance_id:
            return True
        self._run([
            "aws", "ec2", "terminate-instances",
            "--region", self.target.region,
            "--instance-ids", self.state.instance_id,
            "--output", "json",
        ])
        self.state.last_stopped_at = time.time()
        logger.info("aws spot terminated: id=%s", self.state.instance_id)
        self.state.instance_id = None
        self.state.instance_ip = None
        return True

    def is_preempted(self) -> bool:
        """Check the EC2 spot termination notice via instance metadata.

        On a real EC2 spot instance, you'd hit
        http://169.254.169.254/latest/meta-data/spot/termination-time
        from the *guest*. The controller, running outside the instance,
        instead polls describe-spot-instance-requests to see if the
        request status code is `instance-terminated-by-price` or similar.
        """
        if not self.state.instance_id:
            return False
        result = self._run([
            "aws", "ec2", "describe-instances",
            "--region", self.target.region,
            "--instance-ids", self.state.instance_id,
            "--query", "Reservations[].Instances[].State.Name",
            "--output", "json",
        ])
        states = result.get("raw") or result
        if isinstance(states, list) and states and states[0] in ("shutting-down", "terminated", "stopping", "stopped"):
            return True
        return False


# ---------------------------------------------------------------------------
# Azure spot adapter
# ---------------------------------------------------------------------------


class AzureSpotAdapter(SpotAdapter):
    """Uses `az vm create --priority Spot --eviction-policy Deallocate`."""

    name = "azure"

    def start(self) -> bool:
        t = self.target
        vm_name = f"tokenflow-{t.template_name}-{int(time.time())}"
        argv = [
            "az", "vm", "create",
            "--name", vm_name,
            "--resource-group", t.region,         # in azure, "region" is the RG
            "--image", t.image_id,
            "--size", t.instance_type,
            "--priority", "Spot",
            "--eviction-policy", "Deallocate" if t.interruption_behavior == "stop" else "Delete",
            "--tags", f"tokenflow:template={t.template_name}",
            "--output", "json",
        ]
        if t.max_price_usd_per_hour > 0:
            argv.extend(["--max-price", str(t.max_price_usd_per_hour)])
        if t.subnet_id:
            argv.extend(["--subnet", t.subnet_id])
        if t.user_data_script:
            argv.extend(["--custom-data", t.user_data_script])

        result = self._run(argv, timeout=180)
        self.state.instance_id = result.get("id") or vm_name
        self.state.instance_ip = result.get("privateIpAddress") or result.get("publicIpAddress")
        self.state.last_started_at = time.time()
        logger.info("azure spot started: id=%s ip=%s", self.state.instance_id, self.state.instance_ip)
        return True

    def stop(self) -> bool:
        if not self.state.instance_id:
            return True
        # Azure 'id' is a full resource path; vm delete works on it.
        argv = [
            "az", "vm", "delete",
            "--ids", self.state.instance_id,
            "--yes", "--no-wait",
        ]
        self._run(argv, timeout=60)
        self.state.last_stopped_at = time.time()
        logger.info("azure spot deleted: id=%s", self.state.instance_id)
        self.state.instance_id = None
        self.state.instance_ip = None
        return True

    def is_preempted(self) -> bool:
        """Azure surfaces a Scheduled Event for spot eviction. The
        controller, running outside the VM, polls vm get for the
        powerState — which flips to `deallocated` on eviction."""
        if not self.state.instance_id:
            return False
        result = self._run([
            "az", "vm", "get-instance-view",
            "--ids", self.state.instance_id,
            "--query", "instanceView.statuses[?starts_with(code, 'PowerState/')].displayStatus",
            "--output", "json",
        ])
        states = result.get("raw") or result
        if isinstance(states, list) and any("dealloc" in str(s).lower() or "stopped" in str(s).lower() for s in states):
            return True
        return False


# ---------------------------------------------------------------------------
# GCP preemptible / Spot VM adapter
# ---------------------------------------------------------------------------


class GCPSpotAdapter(SpotAdapter):
    """Uses `gcloud compute instances create --provisioning-model=SPOT`."""

    name = "gcp"

    def start(self) -> bool:
        t = self.target
        instance_name = f"tokenflow-{t.template_name}-{int(time.time())}"
        argv = [
            "gcloud", "compute", "instances", "create", instance_name,
            "--zone", t.availability_zone or t.region,
            "--machine-type", t.instance_type,
            "--image", t.image_id,
            "--provisioning-model=SPOT",
            "--instance-termination-action=DELETE",
            f"--labels=tokenflow-template={t.template_name}",
            "--format=json",
        ]
        if t.max_price_usd_per_hour > 0:
            argv.append(f"--max-run-duration={int(3600 * 24)}s")  # bound runtime; price is market on GCP
        if t.subnet_id:
            argv.extend(["--subnet", t.subnet_id])
        if t.user_data_script:
            # GCP uses metadata startup-script
            argv.extend(["--metadata", f"startup-script={t.user_data_script}"])

        result = self._run(argv, timeout=180)
        # gcloud returns a list of created instances
        if isinstance(result, list) and result:
            inst = result[0]
        elif isinstance(result, dict):
            inst = result
        else:
            return False
        self.state.instance_id = inst.get("name") or instance_name
        nics = inst.get("networkInterfaces", [])
        self.state.instance_ip = nics[0].get("networkIP") if nics else None
        self.state.last_started_at = time.time()
        logger.info("gcp spot started: name=%s ip=%s", self.state.instance_id, self.state.instance_ip)
        return True

    def stop(self) -> bool:
        if not self.state.instance_id:
            return True
        self._run([
            "gcloud", "compute", "instances", "delete", self.state.instance_id,
            "--zone", self.target.availability_zone or self.target.region,
            "--quiet",
        ], timeout=120)
        self.state.last_stopped_at = time.time()
        logger.info("gcp spot deleted: name=%s", self.state.instance_id)
        self.state.instance_id = None
        self.state.instance_ip = None
        return True

    def is_preempted(self) -> bool:
        """GCP Spot VMs can be preempted at any time. The controller polls
        the instance's status — `TERMINATED` indicates preemption."""
        if not self.state.instance_id:
            return False
        result = self._run([
            "gcloud", "compute", "instances", "describe", self.state.instance_id,
            "--zone", self.target.availability_zone or self.target.region,
            "--format=value(status)",
        ])
        status = (result.get("raw") or "").strip()
        return status in ("TERMINATED", "STOPPING")


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------


def adapter_for(target: SpotInstanceTarget) -> SpotAdapter:
    """Pick the right adapter class for a target."""
    cls = {
        "aws": AWSSpotAdapter,
        "azure": AzureSpotAdapter,
        "gcp": GCPSpotAdapter,
    }.get(target.provider)
    if cls is None:
        raise ValueError(f"unknown provider: {target.provider}")
    return cls(target)


# ---------------------------------------------------------------------------
# Capacity-controller integration helpers
# ---------------------------------------------------------------------------


def install_into_controller(controller: Any, targets: list[SpotInstanceTarget]) -> None:
    """Wire a list of spot targets into an existing TokenFlowCapacityController.

    The controller in `tokenflow_capacity_controller.py` already has a
    pluggable adapter abstraction (`command`, `systemd`, `docker`, `k8s`).
    This function plugs the spot adapters in by registering callbacks for
    each template name. Call once at controller startup."""
    for target in targets:
        sa = adapter_for(target)
        controller.register_external_lifecycle(
            template_name=target.template_name,
            on_start=sa.start,
            on_stop=sa.stop,
            on_health_check=sa.is_preempted,
            base_url_provider=sa.base_url,
        )


# ---------------------------------------------------------------------------
# Standalone smoke-test runner
# ---------------------------------------------------------------------------


def main() -> None:
    """Smoke-test all three adapters by listing what they would do (no real
    cloud calls). Run with `python examples/autoscale/spot_adapter.py`."""
    import argparse

    ap = argparse.ArgumentParser(description="TokenFlow spot adapter smoke test")
    ap.add_argument("--provider", choices=["aws", "azure", "gcp"], default="aws")
    args = ap.parse_args()

    target = SpotInstanceTarget(
        template_name="vllm-spot-h100",
        backend_type="vllm",
        model_name="meta/llama-3.1-8b-instruct",
        base_url_template="http://{instance_ip}:8000",
        provider=args.provider,
        region={"aws": "us-east-1", "azure": "tokenflow-rg", "gcp": "us-central1"}[args.provider],
        availability_zone="us-central1-a" if args.provider == "gcp" else "",
        instance_type={
            "aws": "g5.2xlarge",
            "azure": "Standard_NC24ads_A100_v4",
            "gcp": "a2-highgpu-1g",
        }[args.provider],
        image_id="ami-0000000000000000",  # replace with your AMI
        max_price_usd_per_hour=1.50,
    )
    sa = adapter_for(target)
    print(f"adapter: {sa.name}")
    print(f"would start: instance_type={target.instance_type} max_price=${target.max_price_usd_per_hour}/hr")
    print(f"would stop:  via {sa._cli} when activated=false")
    print(f"would check preemption: yes")
    print()
    print(f"to run for real, configure {sa._cli} credentials and a real image_id, then:")
    print(f"  python -c 'from spot_adapter import *; sa = adapter_for(...); sa.start()'")


if __name__ == "__main__":
    main()
