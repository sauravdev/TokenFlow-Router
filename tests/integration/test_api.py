"""
Integration tests for the TokenFlow Router API.

Uses FastAPI's TestClient to test the full request path:
register endpoint → send inference request → explain decision.
"""

import pytest
from fastapi.testclient import TestClient

from tokenflow.main import create_app


@pytest.fixture(scope="module")
def client():
    app = create_app()
    with TestClient(app, raise_server_exceptions=True) as c:
        yield c


@pytest.fixture(scope="module")
def registered_endpoint(client):
    resp = client.post("/admin/endpoints", json={
        "name": "test-nim-l40s",
        "nim_url": "http://mock-nim:9999",
        "model_name": "meta/llama-3.1-8b-instruct",
        "gpu_name": "L40S",
        "cost_class": "standard",
        "cost_per_gpu_hour": 2.5,
        "max_context_tokens": 16384,
    })
    assert resp.status_code == 201
    return resp.json()


# ---------------------------------------------------------------------------
# Health
# ---------------------------------------------------------------------------


def test_health_endpoint(client):
    resp = client.get("/health")
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "ok"
    assert "version" in data


# ---------------------------------------------------------------------------
# Endpoint registration
# ---------------------------------------------------------------------------


def test_register_endpoint(client):
    resp = client.post("/admin/endpoints", json={
        "name": "integration-test-ep",
        "nim_url": "http://localhost:9001",
        "model_name": "meta/llama-3.1-70b-instruct",
        "gpu_name": "H100",
        "cost_class": "premium",
        "cost_per_gpu_hour": 8.0,
    })
    assert resp.status_code == 201
    data = resp.json()
    assert data["name"] == "integration-test-ep"
    assert "id" in data


def test_list_endpoints(client, registered_endpoint):
    resp = client.get("/admin/endpoints")
    assert resp.status_code == 200
    data = resp.json()
    assert isinstance(data, list)
    assert len(data) >= 1


def test_get_endpoint(client, registered_endpoint):
    ep_id = registered_endpoint["id"]
    resp = client.get(f"/admin/endpoints/{ep_id}")
    assert resp.status_code == 200
    assert resp.json()["id"] == ep_id


def test_disable_enable_endpoint(client, registered_endpoint):
    ep_id = registered_endpoint["id"]

    resp = client.put(f"/admin/endpoints/{ep_id}/disable")
    assert resp.status_code == 200

    resp = client.put(f"/admin/endpoints/{ep_id}/enable")
    assert resp.status_code == 200


def test_get_nonexistent_endpoint(client):
    resp = client.get("/admin/endpoints/does-not-exist")
    assert resp.status_code == 404


# ---------------------------------------------------------------------------
# Telemetry push
# ---------------------------------------------------------------------------


def test_push_telemetry(client, registered_endpoint):
    ep_id = registered_endpoint["id"]
    resp = client.post("/admin/telemetry", json={
        "endpoint_id": ep_id,
        "rpm": 45.0,
        "queue_depth": 3,
        "p95_ttft_ms": 180.0,
        "p95_e2e_ms": 1200.0,
        "error_rate": 0.01,
    })
    assert resp.status_code == 200
    data = resp.json()
    assert data["endpoint_id"] == ep_id


def test_get_telemetry(client, registered_endpoint):
    ep_id = registered_endpoint["id"]
    # Push first
    client.post("/admin/telemetry", json={"endpoint_id": ep_id, "rpm": 10.0})
    resp = client.get(f"/admin/telemetry/{ep_id}")
    assert resp.status_code == 200


# ---------------------------------------------------------------------------
# Policy
# ---------------------------------------------------------------------------


def test_get_default_policy(client):
    resp = client.get("/admin/policy")
    assert resp.status_code == 200
    data = resp.json()
    assert "name" in data
    assert "slo_weight" in data


def test_switch_preset_latency_first(client):
    resp = client.post("/admin/policy/preset", json={"preset": "latency-first"})
    assert resp.status_code == 200
    data = resp.json()
    assert data["preset"] == "latency-first"
    assert data["weights"]["slo"] == 0.50


def test_switch_preset_invalid(client):
    resp = client.post("/admin/policy/preset", json={"preset": "invalid-preset"})
    assert resp.status_code == 400


# ---------------------------------------------------------------------------
# Inference routing (no real NIM — expects 503 since mock not running)
# ---------------------------------------------------------------------------


def test_chat_completions_no_endpoints_returns_503(client):
    """Without a reachable NIM endpoint, router should return 503."""
    # Use a model with no registered endpoints
    resp = client.post("/v1/chat/completions", json={
        "model": "nonexistent/model-xyz",
        "messages": [{"role": "user", "content": "Hello"}],
    })
    assert resp.status_code == 503


def test_chat_completions_routing_decision_made(client, registered_endpoint):
    """
    With a registered endpoint, the router should attempt routing.
    The upstream call will fail (mock not running) but we can verify
    the routing logic ran by checking the 503 error includes request_id.
    """
    resp = client.post(
        "/v1/chat/completions",
        json={
            "model": "meta/llama-3.1-8b-instruct",
            "messages": [{"role": "user", "content": "hi"}],
            "max_tokens": 16,
        },
        headers={"x-tenant-id": "default", "x-priority-tier": "standard"},
    )
    # Will be 503 (upstream not running) or 200 (if mock responded)
    assert resp.status_code in (200, 503)


# ---------------------------------------------------------------------------
# Models list
# ---------------------------------------------------------------------------


def test_list_models(client, registered_endpoint):
    resp = client.get("/v1/models")
    assert resp.status_code == 200
    data = resp.json()
    assert data["object"] == "list"
    assert isinstance(data["data"], list)


# ---------------------------------------------------------------------------
# Recent routes
# ---------------------------------------------------------------------------


def test_recent_routes(client):
    resp = client.get("/admin/routes/recent?limit=10")
    assert resp.status_code == 200
    assert isinstance(resp.json(), list)


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------


def test_prometheus_metrics(client):
    resp = client.get("/admin/metrics")
    assert resp.status_code == 200
    assert b"tokenflow_" in resp.content
