"""Graph nodes and helper functions for Maintenance Copilot."""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Any

from graph.answering import generate_answer
from graph.classifier import classify_question, extract_filters_fallback
from graph.llm_client import build_chat_client, chat_model
from graph.sql_agent import run_sql_agent
from graph.state import MaintenanceState


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DB_PATH = PROJECT_ROOT / "data" / "maintenance_logs.db"
MAX_ANSWER_RETRIES = 2

def _append_trace(state: MaintenanceState, node_name: str) -> MaintenanceState:
    state = dict(state)
    trace = list(state.get("node_trace", []))
    trace.append(node_name)
    state["node_trace"] = trace
    return state


def _evaluation_summary(is_sufficient: bool, missing_aspects: list[str]) -> str:
    if is_sufficient:
        return "evaluate_answer (sufficient)"
    if missing_aspects:
        return f"evaluate_answer (insufficient: missing {', '.join(missing_aspects[:3])})"
    return "evaluate_answer (insufficient)"


def build_filters(question: str) -> dict[str, Any]:
    return extract_filters_fallback(question)


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
    decision = classify_question(question, state.get("conversation_history", []))
    state["query_type"] = decision.intent
    state["filters"] = decision.filters
    state["router_confidence"] = decision.confidence
    state["router_reasoning"] = decision.reasoning
    state["retry_count"] = int(state.get("retry_count", 0))
    state["is_sufficient"] = None
    state["evaluation_reasoning"] = None
    state["missing_aspects"] = None
    state["hit_retry_cap"] = False
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
    attempt = int(state.get("retry_count", 0)) + 1
    state = _append_trace(state, f"answer (attempt {attempt})")
    generated = generate_answer(state)
    state["answer"] = str(generated.get("answer", state.get("answer", "")))
    citations = generated.get("citations", state.get("citations", []))
    if isinstance(citations, list):
        state["citations"] = [str(item) for item in citations]
    return state


def _deterministic_evaluate_answer(state: MaintenanceState) -> dict[str, object]:
    answer = str(state.get("answer", "")).strip()
    evidence = state.get("evidence", [])
    citations = state.get("citations", [])
    missing_aspects: list[str] = []
    if not answer:
        missing_aspects.append("draft answer is empty")
    if evidence and not citations:
        missing_aspects.append("evidence is available but citations are missing")
    if not evidence:
        missing_aspects.append("no evidence was available for verification")

    return {
        "is_sufficient": bool(answer) and not missing_aspects,
        "missing_aspects": missing_aspects,
        "reasoning": "Deterministic evaluator fallback checked answer presence, evidence, and citations.",
    }


def evaluate_draft_answer(state: MaintenanceState) -> dict[str, object]:
    """Evaluate a draft answer against the graph evidence.

    This is a self-correction/reflection pattern, not a multi-agent handoff:
    one controlled evaluator checks one generator's draft inside the same graph.
    There are no independent agents negotiating ownership or passing tasks around.
    """

    client = build_chat_client()
    if client is None:
        return _deterministic_evaluate_answer(state)

    response = client.chat.completions.create(
        model=chat_model("AZURE_OPENAI_EVALUATOR_DEPLOYMENT_NAME", "OPENAI_EVALUATOR_MODEL"),
        temperature=0,
        response_format={"type": "json_object"},
        messages=[
            {
                "role": "system",
                "content": (
                    "You are an evaluator for an industrial maintenance copilot. "
                    "Judge only whether the draft answer is grounded in the provided evidence. "
                    "Return JSON with keys is_sufficient, missing_aspects, and reasoning."
                ),
            },
            {
                "role": "user",
                "content": json.dumps(
                    {
                        "question": state.get("question"),
                        "draft_answer": state.get("answer"),
                        "query_type": state.get("query_type"),
                        "router_filters": state.get("filters", {}),
                        "router_reasoning": state.get("router_reasoning"),
                        "sql_rows": state.get("sql_rows", []),
                        "sql_query": state.get("sql_query"),
                        "retrieved_chunks": state.get("results", []),
                        "evidence": state.get("evidence", []),
                        "citations": state.get("citations", []),
                        "checks": [
                            "Does the draft answer actually address the user's question?",
                            "Is every substantive claim supported by SQL rows or retrieved chunks?",
                            "Does it cite the evidence it should cite?",
                        ],
                    },
                    ensure_ascii=False,
                ),
            },
        ],
    )

    try:
        payload = json.loads(response.choices[0].message.content or "{}")
    except json.JSONDecodeError:
        return _deterministic_evaluate_answer(state)
    if not isinstance(payload, dict):
        return _deterministic_evaluate_answer(state)

    missing_aspects = payload.get("missing_aspects", [])
    if not isinstance(missing_aspects, list):
        missing_aspects = [str(missing_aspects)]

    return {
        "is_sufficient": bool(payload.get("is_sufficient", False)),
        "missing_aspects": [str(item) for item in missing_aspects],
        "reasoning": str(payload.get("reasoning", "Evaluator returned structured feedback.")),
    }


def evaluate_answer_node(state: MaintenanceState) -> MaintenanceState:
    evaluation = evaluate_draft_answer(state)
    retry_count = int(state.get("retry_count", 0)) + 1
    missing_aspects = [str(item) for item in evaluation.get("missing_aspects", [])]
    is_sufficient = bool(evaluation.get("is_sufficient", False))
    hit_retry_cap = (not is_sufficient) and retry_count >= MAX_ANSWER_RETRIES

    state = _append_trace(state, _evaluation_summary(is_sufficient, missing_aspects))
    state["is_sufficient"] = is_sufficient
    state["evaluation_reasoning"] = str(evaluation.get("reasoning", ""))
    state["missing_aspects"] = missing_aspects
    state["retry_count"] = retry_count
    state["hit_retry_cap"] = hit_retry_cap
    return state


def compose_node(state: MaintenanceState) -> MaintenanceState:
    state = _append_trace(state, "compose")
    if not state.get("answer"):
        state["answer"] = "Soru işlendi ama yanıt üretilemedi."
    if state.get("hit_retry_cap"):
        reasoning = state.get("evaluation_reasoning") or "Evaluator was not fully satisfied after retries."
        state["answer"] = (
            "Not: Bu cevap otomatik değerlendirme döngüsünde tam yeterli bulunmadı; "
            f"kısmen eksik olabilir. Değerlendirme: {reasoning}\n\n{state['answer']}"
        )
    if not state.get("citations"):
        state["citations"] = []
    return state


def _make_hybrid_retriever():
    from retrieval.hybrid_search import HybridRetriever

    return HybridRetriever()


def _make_reranker():
    from retrieval.reranker import Reranker

    return Reranker()
