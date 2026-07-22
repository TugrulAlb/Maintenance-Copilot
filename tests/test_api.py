"""FastAPI tests for the Maintenance Copilot API."""

from __future__ import annotations

from unittest.mock import MagicMock, patch
import sys
from pathlib import Path

from fastapi.testclient import TestClient

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from api.main import app
from api.rate_limit import reset_rate_limiter


client = TestClient(app)


def test_health_endpoint() -> None:
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_index_ui_endpoint() -> None:
    response = client.get("/")
    assert response.status_code == 200
    assert "Maintenance Copilot" in response.text


def test_metrics_endpoint_exists() -> None:
    response = client.get("/metrics")
    assert response.status_code == 200
    assert "maintenance_requests_total" in response.text


@patch("api.main.get_graph_app")
def test_ask_endpoint_uses_stubbed_graph(mock_get_graph_app: MagicMock) -> None:
    mock_graph = MagicMock()
    mock_graph.invoke.return_value = {
        "answer": "Motor failure found on Line 1",
        "query_type": "semantic",
        "thread_id": "thread-123",
        "node_trace": ["retrieve", "rerank", "answer"],
        "citations": ["Record #1: Line 1 / motor failure"],
        "filters": {"production_line": "Line 1"},
        "router_confidence": 0.9,
        "router_reasoning": "semantic issue lookup",
    }
    mock_get_graph_app.return_value = mock_graph

    response = client.post(
        "/ask",
        json={"question": "Motor failure on Line 1?", "thread_id": "thread-123"},
    )

    assert response.status_code == 200
    assert "X-Request-ID" in response.headers
    body = response.json()
    assert body["answer"] == "Motor failure found on Line 1"
    assert body["query_type"] == "semantic"
    assert body["thread_id"] == "thread-123"
    assert body["node_trace"] == ["retrieve", "rerank", "answer"]
    assert body["citations"] == ["Record #1: Line 1 / motor failure"]
    assert body["filters"] == {"production_line": "Line 1"}
    assert body["router_confidence"] == 0.9
    assert body["router_reasoning"] == "semantic issue lookup"
    mock_graph.invoke.assert_called_once()


@patch("api.main.get_graph_app")
def test_ask_endpoint_generates_thread_id_when_missing(mock_get_graph_app: MagicMock) -> None:
    mock_graph = MagicMock()
    mock_graph.invoke.return_value = {
        "answer": "Sensor issue logged",
        "query_type": "semantic",
        "node_trace": ["retrieve", "answer"],
        "citations": ["Record #2: Line 2 / sensor error"],
    }
    mock_get_graph_app.return_value = mock_graph

    response = client.post("/ask", json={"question": "Sensör hatası olan makineler"})

    assert response.status_code == 200
    assert "X-Request-ID" in response.headers
    body = response.json()
    assert body["answer"] == "Sensor issue logged"
    assert body["query_type"] == "semantic"
    assert isinstance(body["thread_id"], str)
    assert len(body["thread_id"]) > 0
    assert body["node_trace"] == ["retrieve", "answer"]
    assert body["citations"] == ["Record #2: Line 2 / sensor error"]


@patch("api.main.get_graph_app")
def test_authentication_and_authorization(mock_get_graph_app: MagicMock, monkeypatch) -> None:
    monkeypatch.setenv("MAINTENANCE_COPILOT_API_KEYS", "user-key,admin-key")
    monkeypatch.setenv("MAINTENANCE_COPILOT_ADMIN_KEYS", "admin-key")
    reset_rate_limiter()

    mock_graph = MagicMock()
    mock_graph.invoke.return_value = {
        "answer": "Analytical response",
        "query_type": "analytical",
        "thread_id": "thread-1",
        "node_trace": ["classify", "analytical", "answer", "compose"],
        "citations": ["SQLite: maintenance_logs query -> SELECT 1"],
    }
    mock_get_graph_app.return_value = mock_graph

    no_key_response = client.post("/ask", json={"question": "Motor failure on Line 1?"})
    assert no_key_response.status_code == 401

    user_response = client.post(
        "/ask",
        json={"question": "Motor failure on Line 1?"},
        headers={"X-API-Key": "user-key"},
    )
    assert user_response.status_code == 200
    assert user_response.json()["answer"] == "Analytical response"

    metrics_denied = client.get("/metrics", headers={"X-API-Key": "user-key"})
    assert metrics_denied.status_code == 403

    metrics_allowed = client.get("/metrics", headers={"X-API-Key": "admin-key"})
    assert metrics_allowed.status_code == 200
    assert "maintenance_requests_total" in metrics_allowed.text


def test_public_demo_mode_can_be_disabled(monkeypatch) -> None:
    monkeypatch.delenv("MAINTENANCE_COPILOT_API_KEYS", raising=False)
    monkeypatch.setenv("MAINTENANCE_COPILOT_ALLOW_PUBLIC", "false")
    reset_rate_limiter()

    response = client.post("/ask", json={"question": "Motor failure on Line 1?"})

    assert response.status_code == 401
    assert response.json()["detail"] == "API keys are required"


@patch("api.main.get_graph_app")
def test_rate_limit_returns_429(mock_get_graph_app: MagicMock, monkeypatch) -> None:
    monkeypatch.setenv("MAINTENANCE_COPILOT_API_KEYS", "user-key")
    monkeypatch.setenv("MAINTENANCE_COPILOT_RATE_LIMIT_PER_MINUTE", "1")
    reset_rate_limiter()

    mock_graph = MagicMock()
    mock_graph.invoke.return_value = {
        "answer": "ok",
        "query_type": "semantic",
        "thread_id": "thread-rate",
        "node_trace": ["classify", "semantic", "answer", "compose"],
        "citations": [],
    }
    mock_get_graph_app.return_value = mock_graph

    first = client.post("/ask", json={"question": "Sensor error?"}, headers={"X-API-Key": "user-key"})
    second = client.post("/ask", json={"question": "Sensor error?"}, headers={"X-API-Key": "user-key"})

    assert first.status_code == 200
    assert second.status_code == 429
    assert second.headers["Retry-After"]
