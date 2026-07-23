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
from graph.classifier import RouteDecision


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
    monkeypatch.setattr(nodes, "classify_question", lambda question, conversation_history=None: RouteDecision(
        intent="analytical",
        confidence=0.91,
        reasoning="test route",
        filters={"fault_category": "motor failure"},
    ))
    monkeypatch.setattr(nodes, "run_sql_agent", lambda question, filters, conversation_history=None, **kwargs: {
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
    assert result["filters"] == {"fault_category": "motor failure"}
    assert result["router_confidence"] == 0.91
    assert "Line 1" in result["answer"] and "Line 2" in result["answer"]
    assert result["node_trace"] == [
        "input_guardrail",
        "classify",
        "analytical",
        "answer (attempt 1)",
        "evaluate_answer (sufficient)",
        "compose",
    ]
    assert result["is_sufficient"] is True
    assert result["retry_count"] == 0
    assert result["evidence_retry_count"] == 0
    assert result["citations"]


def test_graph_semantic_branch(monkeypatch) -> None:
    monkeypatch.setattr(nodes, "classify_question", lambda question, conversation_history=None: RouteDecision(
        intent="semantic",
        confidence=0.88,
        reasoning="test route",
        filters={},
    ))
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
    assert result["node_trace"] == [
        "input_guardrail",
        "classify",
        "semantic",
        "answer (attempt 1)",
        "evaluate_answer (sufficient)",
        "compose",
    ]
    assert result["citations"]


def test_graph_hybrid_branch_combines_sql_and_retrieval(monkeypatch) -> None:
    monkeypatch.setattr(nodes, "classify_question", lambda question, conversation_history=None: RouteDecision(
        intent="hybrid",
        confidence=0.93,
        reasoning="needs both sql and retrieval",
        filters={"production_line": "Line 2", "fault_category": "motor failure"},
    ))
    monkeypatch.setattr(nodes, "_make_hybrid_retriever", _FakeRetriever)
    monkeypatch.setattr(nodes, "_make_reranker", _FakeReranker)
    monkeypatch.setattr(nodes, "run_sql_agent", lambda question, filters, conversation_history=None, **kwargs: {
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
    assert result["node_trace"] == [
        "input_guardrail",
        "classify",
        "hybrid",
        "analytical",
        "semantic",
        "answer (attempt 1)",
        "evaluate_answer (sufficient)",
        "compose",
    ]
    assert len(result["citations"]) == 2


def test_metadata_filter_extraction() -> None:
    filters = nodes.build_filters("Line 3 MC-101 yüksek motor failure kayıtları")

    assert filters == {
        "fault_category": "motor failure",
        "production_line": "Line 3",
        "machine_id": "MC-101",
        "severity": "high",
    }


def test_answer_reflection_retries_with_targeted_feedback(monkeypatch) -> None:
    monkeypatch.setattr(nodes, "classify_question", lambda question, conversation_history=None: RouteDecision(
        intent="analytical",
        confidence=0.9,
        reasoning="test route",
        filters={"production_line": "Line 2"},
    ))
    monkeypatch.setattr(nodes, "run_sql_agent", lambda question, filters, conversation_history=None, **kwargs: {
        "sql_query": "SELECT production_line, COUNT(*) AS issue_count FROM maintenance_logs WHERE production_line = 'Line 2' LIMIT 10",
        "reasoning": "stubbed query",
        "rows": [{"production_line": "Line 2", "issue_count": 4}],
        "citations": ["SQLite: maintenance_logs query -> SELECT production_line"],
    })

    generated_answers = iter([
        {"answer": "Line 2 için sayı verilmedi.", "citations": ["SQLite: maintenance_logs query -> SELECT production_line"]},
        {"answer": "Line 2 için 4 kayıt bulundu.", "citations": ["SQLite: maintenance_logs query -> SELECT production_line"]},
    ])
    evaluations = iter([
        {
            "is_sufficient": False,
            "missing_aspects": ["include the issue count"],
            "reasoning": "The count is missing.",
            "retry_target": "answer",
        },
        {"is_sufficient": True, "missing_aspects": [], "reasoning": "The answer includes the count.", "retry_target": None},
    ])
    seen_missing_aspects = []

    def fake_generate_answer(state):
        seen_missing_aspects.append(list(state.get("missing_aspects", []) or []))
        return next(generated_answers)

    monkeypatch.setattr(nodes, "generate_answer", fake_generate_answer)
    monkeypatch.setattr(nodes, "evaluate_draft_answer", lambda state: next(evaluations))

    result = build_graph.get_graph_app().invoke(
        {
            "question": "Line 2'de kaç kayıt var?",
            "thread_id": "retry-thread",
        }
    )

    assert result["answer"] == "Line 2 için 4 kayıt bulundu."
    assert result["retry_count"] == 1
    assert result["evidence_retry_count"] == 0
    assert result["hit_retry_cap"] is False
    assert seen_missing_aspects == [[], ["include the issue count"]]
    assert result["node_trace"] == [
        "input_guardrail",
        "classify",
        "analytical",
        "answer (attempt 1)",
        "evaluate_answer (insufficient: missing include the issue count)",
        "answer (attempt 2)",
        "evaluate_answer (sufficient)",
        "compose",
    ]


def test_answer_reflection_caps_retries_and_softens_answer(monkeypatch) -> None:
    monkeypatch.setattr(nodes, "classify_question", lambda question, conversation_history=None: RouteDecision(
        intent="semantic",
        confidence=0.8,
        reasoning="test route",
        filters={},
    ))
    monkeypatch.setattr(nodes, "_make_hybrid_retriever", _FakeRetriever)
    monkeypatch.setattr(nodes, "_make_reranker", _FakeReranker)
    monkeypatch.setattr(
        nodes,
        "evaluate_draft_answer",
        lambda state: {
            "is_sufficient": False,
            "missing_aspects": ["cite the retrieved record"],
            "reasoning": "Citation is not specific enough.",
            "retry_target": "answer",
        },
    )

    result = build_graph.get_graph_app().invoke(
        {
            "question": "benzer bakım kayıtlarını göster",
            "thread_id": "cap-thread",
        }
    )

    assert result["retry_count"] == 2
    assert result["evidence_retry_count"] == 0
    assert result["hit_retry_cap"] is True
    assert "kısmen eksik olabilir" in result["answer"]
    assert result["node_trace"] == [
        "input_guardrail",
        "classify",
        "semantic",
        "answer (attempt 1)",
        "evaluate_answer (insufficient: missing cite the retrieved record)",
        "answer (attempt 2)",
        "evaluate_answer (insufficient: missing cite the retrieved record)",
        "answer (attempt 3)",
        "evaluate_answer (insufficient: missing cite the retrieved record)",
        "compose",
    ]


def test_evaluator_can_retry_semantic_evidence_with_wider_search(monkeypatch) -> None:
    class TrackingRetriever:
        def __init__(self) -> None:
            self.calls: list[dict[str, object]] = []

        def search(self, query: str, top_k_each: int = 15, filters: dict[str, Any] | None = None) -> list[dict[str, Any]]:
            self.calls.append({"top_k_each": top_k_each, "filters": filters})
            return [
                {
                    "id": str(len(self.calls)),
                    "document": "Line 2 motor failure retry evidence",
                    "metadata": {"production_line": "Line 2", "fault_category": "motor failure"},
                    "rrf_score": 0.04,
                }
            ]

    class TrackingReranker:
        def __init__(self) -> None:
            self.top_k_values: list[int] = []

        def rerank(self, query: str, candidates: list[dict[str, Any]], top_k: int = 5) -> list[dict[str, Any]]:
            self.top_k_values.append(top_k)
            return [dict(candidate, rerank_score=1.0) for candidate in candidates[:top_k]]

    retriever = TrackingRetriever()
    reranker = TrackingReranker()
    evaluations = iter([
        {
            "is_sufficient": False,
            "missing_aspects": ["retrieved chunks do not cover similar incidents"],
            "reasoning": "Retrieval evidence is too narrow.",
            "retry_target": "semantic",
        },
        {"is_sufficient": True, "missing_aspects": [], "reasoning": "Wider retrieval covers the question.", "retry_target": None},
    ])

    monkeypatch.setattr(nodes, "classify_question", lambda question, conversation_history=None: RouteDecision(
        intent="semantic",
        confidence=0.86,
        reasoning="test route",
        filters={"production_line": "Line 2"},
    ))
    monkeypatch.setattr(nodes, "_make_hybrid_retriever", lambda: retriever)
    monkeypatch.setattr(nodes, "_make_reranker", lambda: reranker)
    monkeypatch.setattr(nodes, "evaluate_draft_answer", lambda state: next(evaluations))

    result = build_graph.get_graph_app().invoke(
        {
            "question": "Line 2'deki benzer motor failure kayıtlarını göster",
            "thread_id": "semantic-evidence-retry",
        }
    )

    assert result["retry_count"] == 0
    assert result["evidence_retry_count"] == 1
    assert result["hit_retry_cap"] is False
    assert retriever.calls == [
        {"top_k_each": 15, "filters": {"production_line": "Line 2"}},
        {"top_k_each": 30, "filters": {"production_line": "Line 2"}},
    ]
    assert reranker.top_k_values == [5, 10]
    assert result["node_trace"] == [
        "input_guardrail",
        "classify",
        "semantic",
        "answer (attempt 1)",
        "evaluate_answer (insufficient: missing retrieved chunks do not cover similar incidents)",
        "semantic (evidence retry: widening search)",
        "answer (attempt 2)",
        "evaluate_answer (sufficient)",
        "compose",
    ]
