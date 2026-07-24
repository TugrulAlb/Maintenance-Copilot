"""Graph nodes and helper functions for Maintenance Copilot."""

from __future__ import annotations

import json
import re
import sqlite3
from pathlib import Path
from typing import Any

from graph.answering import generate_answer
from graph.classifier import classify_question, extract_filters_fallback
from graph.guardrails import apply_output_guardrails, evaluate_input_guardrails
from graph.llm_client import build_chat_client, chat_model
from graph.sql_agent import question_requires_aggregate, run_sql_agent, sql_uses_aggregate
from graph.state import MaintenanceState


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DB_PATH = PROJECT_ROOT / "data" / "maintenance_logs.db"
MAX_ANSWER_RETRIES = 2
MAX_EVIDENCE_RETRIES = 1
EVIDENCE_RETRY_TARGETS = {"analytical", "semantic"}

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


def _is_evidence_retry(state: MaintenanceState, target: str) -> bool:
    retry_target = state.get("retry_target")
    retry_targets = state.get("retry_targets") or []
    return int(state.get("evidence_retry_count", 0)) > 0 and (
        retry_target == target or target in retry_targets
    )


def _evidence_retry_label(target: str) -> str:
    if target == "semantic":
        return "semantic (evidence retry: widening search)"
    if target == "analytical":
        return "analytical (evidence retry: regenerating SQL)"
    return f"{target} (evidence retry)"


def _evaluation_feedback(state: MaintenanceState) -> dict[str, object]:
    return {
        "missing_aspects": state.get("missing_aspects", []),
        "evaluation_reasoning": state.get("evaluation_reasoning"),
        "retry_target": state.get("retry_target"),
        "retry_targets": state.get("retry_targets", []),
    }


def _replace_evidence(state: MaintenanceState, kind: str, replacement: dict[str, object]) -> list[dict[str, object]]:
    return [item for item in state.get("evidence", []) if item.get("kind") != kind] + [replacement]


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


def _tokenize_for_fallback(text: str) -> set[str]:
    """Tiny lexical tokenizer used only when the vector stack is unavailable."""

    stopwords = {
        "a",
        "an",
        "and",
        "about",
        "the",
        "for",
        "find",
        "show",
        "me",
        "incidents",
        "incident",
        "maintenance",
        "kaydi",
        "kaydı",
        "olan",
        "ile",
        "ilgili",
    }
    tokens = {token for token in re.split(r"[^a-z0-9ğüşöçıİĞÜŞÖÇ]+", text.lower()) if token and token not in stopwords}
    synonyms = {
        "kayış": {"belt"},
        "kayis": {"belt"},
        "hizalama": {"alignment", "tracking"},
        "hizalaması": {"alignment", "tracking"},
        "hizalamasi": {"alignment", "tracking"},
        "kaydı": {"drifted"},
        "kaydi": {"drifted"},
        "kaydıği": {"drifted"},
        "kaydığı": {"drifted"},
        "kaydigi": {"drifted"},
        "sola": {"left"},
        "gerginlik": {"tension"},
        "ayar": {"adjustment"},
        "ayarı": {"adjustment"},
        "ayari": {"adjustment"},
        "gereken": {"required"},
        "gerektiren": {"required"},
        "motor": {"motor"},
        "akım": {"current"},
        "akim": {"current"},
        "anormal": {"abnormal"},
        "durma": {"stall"},
        "sıkışma": {"stall"},
        "sicaklik": {"temperature"},
        "sıcaklık": {"temperature"},
        "sensör": {"sensor"},
        "sensor": {"sensor"},
    }
    expanded = set(tokens)
    for token in tokens:
        expanded.update(synonyms.get(token, set()))
    return expanded


def _fallback_semantic_retrieval(question: str, filters: dict[str, Any], top_k: int = 5) -> list[dict[str, Any]]:
    """Return approximate semantic evidence from SQLite when Chroma is not available.

    This keeps the demo path useful in lightweight Docker images. It is not a
    replacement for the real dense+BM25 retriever; it is a graceful fallback that
    scores overlap against the same chunk text the vector index would contain.
    """

    query_tokens = _tokenize_for_fallback(question)
    if not query_tokens:
        return []

    scored: list[tuple[float, dict[str, Any]]] = []
    for row in load_rows():
        metadata = {
            "production_line": row["production_line"],
            "machine_id": row["machine_id"],
            "fault_category": row["fault_category"],
            "severity": row["severity"],
            "timestamp": row["timestamp"],
        }
        if filters and any(not isinstance(value, dict) and metadata.get(key) != value for key, value in filters.items()):
            continue

        document = (
            f"Record #{row['id']} | {row['timestamp']} | {row['production_line']} | "
            f"{row['machine_id']} | {row['fault_category']} | {row['severity']} | "
            f"{row['description']} | resolution_time_minutes={row['resolution_time_minutes']}"
        )
        doc_tokens = _tokenize_for_fallback(document)
        overlap = query_tokens & doc_tokens
        if not overlap:
            continue
        score = len(overlap) / max(1, len(query_tokens))
        scored.append(
            (
                score,
                {
                    "id": str(row["id"]),
                    "document": document,
                    "metadata": metadata,
                    "rrf_score": score,
                    "rerank_score": score,
                    "sources": [{"rank": 1, "score": score, "source": "sqlite_lexical_fallback"}],
                },
            )
        )

    scored.sort(key=lambda item: item[0], reverse=True)
    return [item for _, item in scored[:top_k]]


def input_guardrail_node(state: MaintenanceState) -> MaintenanceState:
    state = _append_trace(state, "input_guardrail")
    result = evaluate_input_guardrails(state["question"])
    state["input_blocked"] = result.blocked
    state["block_reason"] = result.reason
    if result.blocked:
        state["answer"] = result.safe_response or "Bu isteği güvenli şekilde işleyemem."
        state["query_type"] = "blocked"
        state["filters"] = {}
        state["citations"] = []
        state["is_sufficient"] = False
        state["router_confidence"] = None
        state["router_reasoning"] = result.reason
        state["missing_aspects"] = [result.reason] if result.reason else []
        state["retry_count"] = int(state.get("retry_count", 0))
        state["evidence_retry_count"] = int(state.get("evidence_retry_count", 0))
        state["hit_retry_cap"] = False
    return state


def classify_node(state: MaintenanceState) -> MaintenanceState:
    state = _append_trace(state, "classify")
    question = state["question"]
    decision = classify_question(question, state.get("conversation_history", []))
    state["query_type"] = decision.intent
    state["filters"] = decision.filters
    state["router_confidence"] = decision.confidence
    state["router_reasoning"] = decision.reasoning
    state["retry_count"] = int(state.get("retry_count", 0))
    state["evidence_retry_count"] = int(state.get("evidence_retry_count", 0))
    state["is_sufficient"] = None
    state["evaluation_reasoning"] = None
    state["missing_aspects"] = None
    state["retry_target"] = None
    state["retry_targets"] = None
    state["hit_retry_cap"] = False
    return state


def analytical_node(state: MaintenanceState) -> MaintenanceState:
    is_retry = _is_evidence_retry(state, "analytical")
    state = _append_trace(state, _evidence_retry_label("analytical") if is_retry else "analytical")
    question = state["question"]
    filters = state.get("filters", {})
    # On analytical evidence retries, widening means regenerating SQL with the
    # previous query and evaluator feedback in prompt context, so the SQL agent
    # can loosen an overly narrow WHERE clause or choose a more useful aggregate.
    sql_result = run_sql_agent(
        question,
        filters,
        state.get("conversation_history", []),
        previous_sql_query=state.get("sql_query") if is_retry else None,
        evaluation_feedback=_evaluation_feedback(state) if is_retry else None,
    )
    rows = sql_result["rows"]
    state["sql_rows"] = rows
    state["sql_query"] = str(sql_result["sql_query"])
    state["sql_generation_issue"] = bool(sql_result.get("sql_generation_issue", False))
    state["aggregate_required"] = bool(sql_result.get("aggregate_required", False))
    sql_citations = list(sql_result.get("citations", []))
    if is_retry:
        state["citations"] = [item for item in state.get("citations", []) if not str(item).startswith("SQLite:")] + sql_citations
    else:
        state["citations"] = sql_citations

    sql_evidence = {
        "kind": "sql",
        "sql_query": sql_result["sql_query"],
        "rows": rows,
        "reasoning": sql_result.get("reasoning", ""),
        "aggregate_required": sql_result.get("aggregate_required", False),
        "sql_generation_issue": sql_result.get("sql_generation_issue", False),
    }
    state["evidence"] = _replace_evidence(state, "sql", sql_evidence) if is_retry else [sql_evidence]
    return state


def semantic_node(state: MaintenanceState) -> MaintenanceState:
    is_retry = _is_evidence_retry(state, "semantic")
    state = _append_trace(state, _evidence_retry_label("semantic") if is_retry else "semantic")

    try:
        retriever = _make_hybrid_retriever()
        reranker = _make_reranker()
    except Exception as exc:
        # Container demos or locked-down environments may intentionally omit
        # heavy retrieval dependencies or embedding credentials. The graph should
        # degrade gracefully instead of turning a semantic/hybrid question into a
        # 500 response. We still run a small SQLite lexical fallback so the
        # semantic path can return cited evidence in local demos while the full
        # dense+BM25 index is unavailable.
        state["semantic_error"] = f"Semantic retrieval unavailable: {exc}"
        top_k = 10 if is_retry else 5
        fallback_results = _fallback_semantic_retrieval(state["question"], state.get("filters", {}), top_k=top_k)
        if not fallback_results and state.get("filters"):
            fallback_results = _fallback_semantic_retrieval(state["question"], {}, top_k=top_k)
            state["loosened_filters_on_retry"] = True
        state["candidates"] = fallback_results
        state["results"] = fallback_results
        state["evidence"] = _replace_evidence(
            state,
            "retrieval",
            {"kind": "retrieval", "items": fallback_results, "fallback": "sqlite_lexical"},
        )
        fallback_citations = [
            f"Record #{item.get('id', '?')}: {item.get('metadata', {}).get('production_line', '?')} / {item.get('metadata', {}).get('fault_category', '?')}"
            for item in fallback_results[:5]
        ]
        state["citations"] = [item for item in state.get("citations", []) if not str(item).startswith("Record #")] + fallback_citations
        return state
    filters = state.get("filters", {})
    # Widening makes evidence retries meaningfully different from the first pass:
    # we ask each retriever for a larger pool and rerank more final chunks. If
    # strict router filters produce no candidates, we retry once without filters
    # because the original filters may have been too narrow.
    top_k_each = 30 if is_retry else 15
    rerank_top_k = 10 if is_retry else 5
    candidates = retriever.search(state["question"], top_k_each=top_k_each, filters=filters)
    if is_retry and not candidates and filters:
        candidates = retriever.search(state["question"], top_k_each=top_k_each, filters=None)
        state["loosened_filters_on_retry"] = True
    reranked = reranker.rerank(state["question"], candidates, top_k=rerank_top_k)

    state["candidates"] = candidates
    state["results"] = reranked
    retrieval_evidence = {
        "kind": "retrieval",
        "items": reranked,
    }
    state["evidence"] = _replace_evidence(state, "retrieval", retrieval_evidence) if is_retry else [retrieval_evidence]
    retrieval_citations = [
        f"Record #{item.get('id', '?')}: {item.get('metadata', {}).get('production_line', '?')} / {item.get('metadata', {}).get('fault_category', '?')}"
        for item in reranked[:5]
    ]
    if is_retry:
        state["citations"] = [item for item in state.get("citations", []) if not str(item).startswith("Record #")] + retrieval_citations
    else:
        state["citations"] = retrieval_citations

    if not reranked:
        return state
    return state


def hybrid_node(state: MaintenanceState) -> MaintenanceState:
    state = _append_trace(state, "hybrid (evidence retry)" if int(state.get("evidence_retry_count", 0)) > 0 else "hybrid")
    analytical_state = analytical_node(dict(state))
    semantic_state = semantic_node(dict(state))

    state["node_trace"] = list(
        dict.fromkeys(
            list(state.get("node_trace", []))
            + list(analytical_state.get("node_trace", []))[len(state.get("node_trace", [])) :]
            + list(semantic_state.get("node_trace", []))[len(state.get("node_trace", [])) :]
        )
    )
    state["sql_rows"] = analytical_state.get("sql_rows", [])
    state["sql_query"] = analytical_state.get("sql_query", "")
    state["candidates"] = semantic_state.get("candidates", [])
    state["results"] = semantic_state.get("results", [])
    state["evidence"] = list(analytical_state.get("evidence", [])) + list(semantic_state.get("evidence", []))
    state["citations"] = list(analytical_state.get("citations", [])) + list(semantic_state.get("citations", []))
    return state


def answer_node(state: MaintenanceState) -> MaintenanceState:
    attempt = sum(1 for item in state.get("node_trace", []) if str(item).startswith("answer (attempt")) + 1
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
    aggregate_required = bool(state.get("aggregate_required")) or question_requires_aggregate(str(state.get("question", "")))
    sql_query = str(state.get("sql_query", ""))
    if aggregate_required and sql_query and not sql_uses_aggregate(sql_query):
        missing_aspects.append("SQL query type does not match aggregation question")
    retry_target = "answer"
    retry_targets: list[str] = []
    if "no evidence was available for verification" in missing_aspects:
        query_type = state.get("query_type")
        if query_type == "analytical":
            retry_target = "analytical"
        elif query_type == "semantic":
            retry_target = "semantic"
        elif query_type == "hybrid":
            retry_target = "answer"
            retry_targets = ["analytical", "semantic"]
    elif "SQL query type does not match aggregation question" in missing_aspects:
        retry_target = "analytical"

    return {
        "is_sufficient": bool(answer) and not missing_aspects,
        "missing_aspects": missing_aspects,
        "reasoning": "Deterministic evaluator fallback checked answer presence, evidence, and citations.",
        "retry_target": None if bool(answer) and not missing_aspects else retry_target,
        "retry_targets": retry_targets or None,
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
                    "If insufficient, decide whether the root cause is weak evidence or weak answer composition. "
                    "Return JSON with keys is_sufficient, missing_aspects, reasoning, retry_target, and retry_targets."
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
                            "If the user asks for a count, average, top-N, distribution, or trend, does the SQL query use aggregate SQL such as COUNT, AVG, SUM, or GROUP BY instead of a plain SELECT ... LIMIT listing?",
                            "If the evidence itself is missing or insufficient, choose retry_target analytical or semantic.",
                            "If this is a hybrid question and both SQL and retrieval evidence need repair, use retry_targets.",
                            "If the evidence is enough but the answer failed to use it well, choose retry_target answer.",
                        ],
                        "retry_target_rules": {
                            "answer": "Use when evidence is sufficient but the draft answer is incomplete, poorly framed, or missing citations.",
                            "analytical": "Use when SQL evidence is missing, too narrow, or does not answer the analytical part.",
                            "semantic": "Use when retrieved chunks are missing, too narrow, or do not answer the semantic part.",
                            "retry_targets": "For hybrid intent only, list analytical and/or semantic when more than one evidence source needs re-acquisition.",
                        },
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
        "retry_target": payload.get("retry_target"),
        "retry_targets": payload.get("retry_targets"),
    }


def _sanitize_retry_target(value: object) -> str | None:
    if value in {"answer", "analytical", "semantic"}:
        return str(value)
    return None


def _sanitize_retry_targets(value: object) -> list[str]:
    if not isinstance(value, list):
        return []
    return [str(item) for item in value if item in EVIDENCE_RETRY_TARGETS]


def evaluate_answer_node(state: MaintenanceState) -> MaintenanceState:
    evaluation = evaluate_draft_answer(state)
    missing_aspects = [str(item) for item in evaluation.get("missing_aspects", [])]
    is_sufficient = bool(evaluation.get("is_sufficient", False))
    retry_count = int(state.get("retry_count", 0))
    evidence_retry_count = int(state.get("evidence_retry_count", 0))
    retry_target = None if is_sufficient else _sanitize_retry_target(evaluation.get("retry_target"))
    retry_targets = [] if is_sufficient else _sanitize_retry_targets(evaluation.get("retry_targets"))

    if not is_sufficient and retry_target is None and not retry_targets:
        retry_target = "answer"

    evidence_retry_requested = bool(retry_targets) or retry_target in EVIDENCE_RETRY_TARGETS
    hit_retry_cap = False
    # Answer retries and evidence retries have different cost profiles. Rewriting
    # a final response is cheap; re-running retrieval or regenerating SQL can be
    # slower and more expensive, so evidence re-acquisition gets its own tighter
    # budget instead of sharing the answer-only retry budget.
    if not is_sufficient and evidence_retry_requested:
        if evidence_retry_count < MAX_EVIDENCE_RETRIES:
            evidence_retry_count += 1
        else:
            hit_retry_cap = True
    elif not is_sufficient and retry_target == "answer":
        if retry_count < MAX_ANSWER_RETRIES:
            retry_count += 1
        else:
            hit_retry_cap = True

    state = _append_trace(state, _evaluation_summary(is_sufficient, missing_aspects))
    state["is_sufficient"] = is_sufficient
    state["evaluation_reasoning"] = str(evaluation.get("reasoning", ""))
    state["missing_aspects"] = missing_aspects
    state["retry_target"] = retry_target
    state["retry_targets"] = retry_targets or None
    state["retry_count"] = retry_count
    state["evidence_retry_count"] = evidence_retry_count
    state["hit_retry_cap"] = hit_retry_cap
    return state


def compose_node(state: MaintenanceState) -> MaintenanceState:
    state = _append_trace(state, "compose")
    guarded = apply_output_guardrails(
        str(state.get("answer", "")),
        is_sufficient=state.get("is_sufficient"),
        hit_retry_cap=bool(state.get("hit_retry_cap", False)),
        evaluation_reasoning=state.get("evaluation_reasoning"),
        input_blocked=bool(state.get("input_blocked", False)),
    )
    state["answer"] = guarded.answer
    state["output_redactions"] = guarded.redactions
    if not state.get("citations"):
        state["citations"] = []
    return state


def _make_hybrid_retriever():
    from retrieval.hybrid_search import HybridRetriever

    return HybridRetriever()


def _make_reranker():
    from retrieval.reranker import Reranker

    return Reranker()
