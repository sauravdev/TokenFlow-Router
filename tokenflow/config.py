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

    # Policy
    policy_file: str | None = None

    # Observability
    enable_metrics: bool = True
    explain_log_path: str | None = None

    # Auth (optional)
    admin_api_key: str | None = None

    # Upstream request forwarding
    upstream_timeout_s: int = 120
    upstream_connect_timeout_s: int = 5


settings = Settings()
