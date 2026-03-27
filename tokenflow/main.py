"""
TokenFlow Router — application entrypoint.

Wires together registry, telemetry, policy engine, decision engine,
gateway, and admin API into a single FastAPI application.
"""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager

import structlog
import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from tokenflow.admin.routes import router as admin_router
from tokenflow.config import settings
from tokenflow.gateway.proxy import UpstreamProxy
from tokenflow.gateway.routes import router as gateway_router
from tokenflow.observability import TraceStore
from tokenflow.policy_engine.engine import PolicyEngine, load_policy_from_yaml
from tokenflow.profiles import ProfileManager
from tokenflow.registry import EndpointRegistry
from tokenflow.router import DecisionEngine, _apply_preset
from tokenflow.telemetry import TelemetryCollector, TelemetryStore

# ---------------------------------------------------------------------------
# Logging setup
# ---------------------------------------------------------------------------

structlog.configure(
    processors=[
        structlog.contextvars.merge_contextvars,
        structlog.processors.add_log_level,
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.dev.ConsoleRenderer()
        if settings.log_level == "DEBUG"
        else structlog.processors.JSONRenderer(),
    ],
    wrapper_class=structlog.make_filtering_bound_logger(
        getattr(logging, settings.log_level.upper(), logging.INFO)
    ),
    context_class=dict,
    logger_factory=structlog.PrintLoggerFactory(),
)

logger = structlog.get_logger(__name__)


# ---------------------------------------------------------------------------
# Application factory
# ---------------------------------------------------------------------------


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup and shutdown lifecycle."""
    logger.info("tokenflow_starting", version="0.1.0")

    # Core stores
    registry = EndpointRegistry()
    telemetry_store = TelemetryStore()
    trace_store = TraceStore()

    # Policy engine
    policy_engine = PolicyEngine()
    if settings.policy_file:
        try:
            policy = load_policy_from_yaml(settings.policy_file)
            policy_engine.set_policy(policy)
            logger.info("policy_loaded_from_file", path=settings.policy_file)
        except Exception as exc:
            logger.warning("policy_file_load_failed", path=settings.policy_file, error=str(exc))

    # Decision engine
    decision_engine = DecisionEngine(registry=registry, store=telemetry_store)
    decision_engine.set_policy(_apply_preset(policy_engine.policy))

    # Telemetry collector
    telemetry_collector = TelemetryCollector(store=telemetry_store)
    await telemetry_collector.start()

    # Proxy
    proxy = UpstreamProxy()

    # Profile manager (dynamic backend lazy activation)
    profile_manager = ProfileManager()
    profile_manager.attach(registry, telemetry_collector)

    # Attach to app state
    app.state.registry = registry
    app.state.telemetry_store = telemetry_store
    app.state.telemetry_collector = telemetry_collector
    app.state.trace_store = trace_store
    app.state.policy_engine = policy_engine
    app.state.decision_engine = decision_engine
    app.state.proxy = proxy
    app.state.profile_manager = profile_manager

    logger.info("tokenflow_ready")
    yield

    # Shutdown
    logger.info("tokenflow_shutting_down")
    await telemetry_collector.stop()
    await proxy.close()
    logger.info("tokenflow_stopped")


def create_app() -> FastAPI:
    app = FastAPI(
        title="TokenFlow Router",
        description=(
            "Request-aware policy router for NVIDIA NIM inference endpoints. "
            "Routes every token to the right GPU lane."
        ),
        version="0.1.0",
        lifespan=lifespan,
        docs_url="/docs",
        redoc_url="/redoc",
    )

    cors_origins = settings.cors_origins
    app.add_middleware(
        CORSMiddleware,
        allow_origins=cors_origins,
        allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
        allow_headers=["Content-Type", "Authorization", "X-Admin-API-Key",
                       "X-Tenant-ID", "X-App-ID", "X-Priority-Tier", "X-Optimization-Target", "X-Budget-Sensitivity"],
        allow_credentials=cors_origins != ["*"],
    )

    app.include_router(gateway_router, tags=["inference"])
    app.include_router(admin_router, tags=["admin"])

    return app


app = create_app()


if __name__ == "__main__":
    uvicorn.run(
        "tokenflow.main:app",
        host=settings.host,
        port=settings.port,
        log_level=settings.log_level.lower(),
        workers=settings.workers,
    )
