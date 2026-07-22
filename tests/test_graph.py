"""Tests for the real LangGraph workflow.

These tests patch the external dependencies so the graph can be verified without
calling an embedding API or reading a real SQLite database during test runs.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import graph.build_graph as build_graph
import graph.nodes as nodes


class _FakeRetriever:
    def search(self, query: str, top_k_each: int = 15, filters: dict[str, Any] | None = None) -> list[dict[str, Any]]:
        assert filters is None or isinstance(filters, dict)
        return [
            {
                "id": "1",
                "document": "timestamp: 2026-07-01\nproduction_line: Line 2\nmachine_id: MC-101\nfault_category: motor failure\ndescription: motor overheated and stopped",
                "metadata": {"production_line": "Line 2", "fault_category": "motor failure"},
                "rrf_score": 0.04,
            }
        ]


class _FakeReranker:
    def rerank(self, query: str, candidates: list[dict[str, Any]], top_k: int = 5) -> list[dict[str, Any]]:
        return [dict(candidates[0], rerank_score=1.5)]


def test_graph_analytical_branch(monkeypatch) -> None:
    monkeypatch.setattr(nodes, "classify_intent", lambda question: "analytical")
    monkeypatch.setattr(nodes, "run_sql_agent", lambda question, filters, conversation_history=None: {
        "sql_query": "SELECT * FROM maintenance_logs WHERE fault_category = 'motor failure'",
        "reasoning": "stubbed test query",
        "rows": [
            {
                "id": 1,
                "timestamp": "2026-07-01T00:00:00+00:00",
                "production_line": "Line 1",
                "machine_id": "MC-100",
                "fault_category": "motor failure",
                "description": "motor stopped",
                "severity": "high",
                "resolution_time_minutes": 120,
                "resolved_by": "Alice",
            },
            {
                "id": 2,
                "timestamp": "2026-07-02T00:00:00+00:00",
                "production_line": "Line 2",
                "machine_id": "MC-101",
                "fault_category": "motor failure",
                "description": "motor overheated",
                "severity": "medium",
                "resolution_time_minutes": 80,
                "resolved_by": "Ben",
            },
        ],
        "citations": ["SQLite: maintenance_logs query -> SELECT * FROM maintenance_logs WHERE fault_category = 'motor failure'"],
    })

    result = build_graph.get_graph_app().invoke(
        {
            "question": "motor failure nedeniyle duran hatlar hangileri?",
            "thread_id": "t-1",
            "query_type": "analytical",
        }
    )

    assert result["query_type"] == "analytical"
    assert "Line 1" in result["answer"] and "Line 2" in result["answer"]
    assert result["node_trace"] == ["classify", "analytical", "answer", "compose"]
    assert result["citations"]


def test_graph_semantic_branch(monkeypatch) -> None:
    monkeypatch.setattr(nodes, "classify_intent", lambda question: "semantic")
    monkeypatch.setattr(nodes, "load_rows", lambda: [])
    monkeypatch.setattr(nodes, "_make_hybrid_retriever", _FakeRetriever)
    monkeypatch.setattr(nodes, "_make_reranker", _FakeReranker)

    result = build_graph.get_graph_app().invoke(
        {
            "question": "benzer bakım kayıtlarını göster",
            "thread_id": "t-2",
            "query_type": "semantic",
        }
    )

    assert result["query_type"] == "semantic"
    assert "Line 2" in result["answer"]
    assert result["node_trace"] == ["classify", "semantic", "answer", "compose"]
    assert result["citations"]


def test_graph_hybrid_branch_combines_sql_and_retrieval(monkeypatch) -> None:
    monkeypatch.setattr(nodes, "classify_intent", lambda question: "hybrid")
    monkeypatch.setattr(nodes, "_make_hybrid_retriever", _FakeRetriever)
    monkeypatch.setattr(nodes, "_make_reranker", _FakeReranker)
    monkeypatch.setattr(nodes, "run_sql_agent", lambda question, filters, conversation_history=None: {
        "sql_query": "SELECT production_line, COUNT(*) AS issue_count FROM maintenance_logs GROUP BY production_line LIMIT 10",
        "reasoning": "stubbed hybrid query",
        "rows": [
            {
                "production_line": "Line 2",
                "fault_category": "motor failure",
                "machine_id": "MC-101",
            }
        ],
        "citations": ["SQLite: maintenance_logs query -> SELECT production_line"],
    })

    result = build_graph.get_graph_app().invoke(
        {
            "question": "Line 2 motor failure trendini ve benzer kayıtları açıkla",
            "thread_id": "t-3",
        }
    )

    assert result["query_type"] == "hybrid"
    assert "Analytical sonuç" in result["answer"]
    assert "Semantic olarak" in result["answer"]
    assert result["node_trace"] == ["classify", "hybrid", "analytical", "semantic", "answer", "compose"]
    assert len(result["citations"]) == 2


def test_metadata_filter_extraction() -> None:
    filters = nodes.build_filters("Line 3 MC-101 yüksek motor failure kayıtları")

    assert filters == {
        "fault_category": "motor failure",
        "production_line": "Line 3",
        "machine_id": "MC-101",
        "severity": "high",
    }
