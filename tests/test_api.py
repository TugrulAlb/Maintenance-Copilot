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


client = TestClient(app)


def test_health_endpoint() -> None:
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


@patch("api.main.get_graph_app")
def test_ask_endpoint_uses_stubbed_graph(mock_get_graph_app: MagicMock) -> None:
    mock_graph = MagicMock()
    mock_graph.invoke.return_value = {
        "answer": "Motor failure found on Line 1",
        "query_type": "semantic",
        "thread_id": "thread-123",
        "node_trace": ["retrieve", "rerank", "answer"],
    }
    mock_get_graph_app.return_value = mock_graph

    response = client.post(
        "/ask",
        json={"question": "Motor failure on Line 1?", "thread_id": "thread-123"},
    )

    assert response.status_code == 200
    body = response.json()
    assert body["answer"] == "Motor failure found on Line 1"
    assert body["query_type"] == "semantic"
    assert body["thread_id"] == "thread-123"
    assert body["node_trace"] == ["retrieve", "rerank", "answer"]
    mock_graph.invoke.assert_called_once()


@patch("api.main.get_graph_app")
def test_ask_endpoint_generates_thread_id_when_missing(mock_get_graph_app: MagicMock) -> None:
    mock_graph = MagicMock()
    mock_graph.invoke.return_value = {
        "answer": "Sensor issue logged",
        "query_type": "semantic",
        "node_trace": ["retrieve", "answer"],
    }
    mock_get_graph_app.return_value = mock_graph

    response = client.post("/ask", json={"question": "Sensör hatası olan makineler"})

    assert response.status_code == 200
    body = response.json()
    assert body["answer"] == "Sensor issue logged"
    assert body["query_type"] == "semantic"
    assert isinstance(body["thread_id"], str)
    assert len(body["thread_id"]) > 0
    assert body["node_trace"] == ["retrieve", "answer"]