"""Graph nodes and helper functions for Maintenance Copilot."""

from __future__ import annotations

import sqlite3
from pathlib import Path
import re
from typing import Any

from graph.answering import generate_answer
from graph.classifier import classify_intent
from graph.sql_agent import run_sql_agent
from graph.state import MaintenanceState


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DB_PATH = PROJECT_ROOT / "data" / "maintenance_logs.db"

FAULT_CATEGORY_ALIASES = {
    "motor failure": ["motor failure", "motor", "motor arız", "motor ariza"],
    "sensor error": ["sensor error", "sensör", "sensor", "sensör hatası", "sensor hatası"],
    "belt misalignment": ["belt misalignment", "belt", "kayış", "hiza", "misalignment"],
    "overheating": ["overheating", "ısın", "sicak", "sıcak", "heat", "overheat"],
    "electrical fault": ["electrical fault", "electrical", "elektrik", "voltaj", "power", "wiring"],
}


def _append_trace(state: MaintenanceState, node_name: str) -> MaintenanceState:
    state = dict(state)
    trace = list(state.get("node_trace", []))
    trace.append(node_name)
    state["node_trace"] = trace
    return state


def _normalize_question(question: str) -> str:
    return question.lower().strip()


def _detect_fault_category(question: str) -> str | None:
    lowered = _normalize_question(question)
    for category, aliases in FAULT_CATEGORY_ALIASES.items():
        if any(alias in lowered for alias in aliases):
            return category
    return None


def _detect_line(question: str) -> str | None:
    lowered = _normalize_question(question)
    match = re.search(r"\b(?:line|hat)\s*([123])\b", lowered)
    if match:
        return f"Line {match.group(1)}"
    return None


def _detect_machine_id(question: str) -> str | None:
    match = re.search(r"\b(?:mc|machine|makine)[-\s]?(\d{2,4})\b", question, flags=re.IGNORECASE)
    if not match:
        return None
    return f"MC-{match.group(1)}"


def _detect_severity(question: str) -> str | None:
    lowered = _normalize_question(question)
    aliases = {
        "high": ["high", "yüksek", "yuksek", "kritik", "critical"],
        "medium": ["medium", "orta"],
        "low": ["low", "düşük", "dusuk"],
    }
    for severity, terms in aliases.items():
        if any(term in lowered for term in terms):
            return severity
    return None


def build_filters(question: str) -> dict[str, Any]:
    filters: dict[str, Any] = {}
    fault_category = _detect_fault_category(question)
    line = _detect_line(question)
    machine_id = _detect_machine_id(question)
    severity = _detect_severity(question)
    if fault_category:
        filters["fault_category"] = fault_category
    if line:
        filters["production_line"] = line
    if machine_id:
        filters["machine_id"] = machine_id
    if severity:
        filters["severity"] = severity
    return filters


def load_rows() -> list[sqlite3.Row]:
    if not DB_PATH.exists():
        raise FileNotFoundError(
            f"SQLite database not found at {DB_PATH}. Run data/generate_synthetic_data.py first."
        )

    connection = sqlite3.connect(DB_PATH)
    connection.row_factory = sqlite3.Row
    try:
        cursor = connection.execute(
            """
            SELECT id, timestamp, production_line, machine_id, fault_category,
                   description, severity, resolution_time_minutes, resolved_by
            FROM maintenance_logs
            ORDER BY id
            """
        )
        return cursor.fetchall()
    finally:
        connection.close()


def classify_node(state: MaintenanceState) -> MaintenanceState:
    state = _append_trace(state, "classify")
    question = state["question"]
    state["query_type"] = classify_intent(question)
    state["filters"] = build_filters(question)
    return state


def analytical_node(state: MaintenanceState) -> MaintenanceState:
    state = _append_trace(state, "analytical")
    question = state["question"]
    filters = state.get("filters", {})
    sql_result = run_sql_agent(question, filters, state.get("conversation_history", []))
    rows = sql_result["rows"]
    state["sql_rows"] = rows
    state["sql_query"] = str(sql_result["sql_query"])
    state["citations"] = list(sql_result.get("citations", []))
    state["evidence"] = [
        {
            "kind": "sql",
            "sql_query": sql_result["sql_query"],
            "rows": rows,
            "reasoning": sql_result.get("reasoning", ""),
        }
    ]

    if not rows:
        state["evidence"] = [
            {
                "kind": "sql",
                "sql_query": sql_result["sql_query"],
                "rows": [],
                "reasoning": sql_result.get("reasoning", ""),
            }
        ]
    return state


def semantic_node(state: MaintenanceState) -> MaintenanceState:
    state = _append_trace(state, "semantic")

    retriever = _make_hybrid_retriever()
    reranker = _make_reranker()
    filters = state.get("filters", {})
    candidates = retriever.search(state["question"], top_k_each=15, filters=filters)
    reranked = reranker.rerank(state["question"], candidates, top_k=5)

    state["candidates"] = candidates
    state["results"] = reranked
    state["evidence"] = [
        {
            "kind": "retrieval",
            "items": reranked,
        }
    ]
    state["citations"] = [
        f"Record #{item.get('id', '?')}: {item.get('metadata', {}).get('production_line', '?')} / {item.get('metadata', {}).get('fault_category', '?')}"
        for item in reranked[:5]
    ]

    if not reranked:
        return state
    return state


def hybrid_node(state: MaintenanceState) -> MaintenanceState:
    state = _append_trace(state, "hybrid")
    analytical_state = analytical_node(dict(state))
    semantic_state = semantic_node(dict(state))

    state["node_trace"] = list(dict.fromkeys(list(state.get("node_trace", [])) + ["analytical", "semantic"]))
    state["sql_rows"] = analytical_state.get("sql_rows", [])
    state["sql_query"] = analytical_state.get("sql_query", "")
    state["candidates"] = semantic_state.get("candidates", [])
    state["results"] = semantic_state.get("results", [])
    state["evidence"] = list(analytical_state.get("evidence", [])) + list(semantic_state.get("evidence", []))
    state["citations"] = list(analytical_state.get("citations", [])) + list(semantic_state.get("citations", []))
    return state


def answer_node(state: MaintenanceState) -> MaintenanceState:
    state = _append_trace(state, "answer")
    generated = generate_answer(state)
    state["answer"] = str(generated.get("answer", state.get("answer", "")))
    citations = generated.get("citations", state.get("citations", []))
    if isinstance(citations, list):
        state["citations"] = [str(item) for item in citations]
    return state


def compose_node(state: MaintenanceState) -> MaintenanceState:
    state = _append_trace(state, "compose")
    if not state.get("answer"):
        state["answer"] = "Soru işlendi ama yanıt üretilemedi."
    if not state.get("citations"):
        state["citations"] = []
    return state


def _make_hybrid_retriever():
    from retrieval.hybrid_search import HybridRetriever

    return HybridRetriever()


def _make_reranker():
    from retrieval.reranker import Reranker

    return Reranker()
