"""Global configuration via environment variables."""

from __future__ import annotations

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="TOKENFLOW_", env_file=".env")

    # Server
    host: str = "0.0.0.0"
    port: int = 8080
    log_level: str = "INFO"
    workers: int = 1

    # Routing
    route_timeout_ms: int = 10  # hard SLO for routing decision itself
    fallback_timeout_ms: int = 20
    max_fallback_attempts: int = 3
    default_policy: str = "balanced"  # latency-first | balanced | cost-first

    # Telemetry
    telemetry_scrape_interval_s: int = 10
    telemetry_stale_threshold_s: int = 60  # treat as stale after N seconds
    telemetry_smoothing_alpha: float = 0.3  # EMA alpha for smoothing metrics
    # Warmup grace for freshly-registered endpoints (e.g. dormant-profile
    # auto-activations). Probe failures within this window after
    # `registered_at` do not mark the endpoint UNHEALTHY — covers the
    # period while a newly-started container boots its serving process.
    endpoint_warmup_grace_s: int = 120

    # Optional upstream classifier (e.g. NVIDIA AI Blueprints LLM Router
    # v2 in `intent` profile, or any HTTP service speaking
    # POST /recommendation with messages + metadata → {intent, ...}).
    # When set, every routed request gets classified by this service
    # first; the canonical intent label is folded into the request
    # profile's workload_type before scoring.
    # Empty / unset = use the local keyword-based heuristic only.
    external_classifier_url: str = ""
    external_classifier_timeout_s: float = 0.5

    # Policy
    policy_file: str | None = None

    # Observability
    enable_metrics: bool = True
    explain_log_path: str | None = None

    # Auth (optional)
    admin_api_key: str | None = None

    # CORS — comma-separated list of allowed origins; "*" only for local dev
    allowed_origins: str = "*"

    @property
    def cors_origins(self) -> list[str]:
        return [o.strip() for o in self.allowed_origins.split(",") if o.strip()]

    # Upstream request forwarding
    upstream_timeout_s: int = 120
    upstream_connect_timeout_s: int = 5


settings = Settings()
