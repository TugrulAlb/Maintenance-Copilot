"""Workflow assembly for the Maintenance Copilot LangGraph app."""

from __future__ import annotations

from langgraph.graph import END, START, StateGraph

from graph.nodes import (
    MAX_ANSWER_RETRIES,
    MAX_EVIDENCE_RETRIES,
    analytical_node,
    answer_node,
    classify_node,
    compose_node,
    evaluate_answer_node,
    hybrid_node,
    input_guardrail_node,
    semantic_node,
)
from graph.state import MaintenanceState


def _route_node(state: MaintenanceState) -> str:
    query_type = state.get("query_type")
    if query_type in {"analytical", "semantic", "hybrid"}:
        return query_type
    return "hybrid"


def route_after_input_guardrail(state: MaintenanceState) -> str:
    if state.get("input_blocked"):
        return "compose"
    return "classify"


def route_after_evaluation(state: MaintenanceState) -> str:
    if state.get("is_sufficient") is True:
        return "compose"

    if state.get("hit_retry_cap"):
        # The caps guarantee graceful degradation: the graph surfaces a caveat
        # in compose instead of looping forever or silently pretending certainty.
        return "compose"

    retry_targets = state.get("retry_targets") or []
    retry_target = state.get("retry_target")

    if retry_targets and int(state.get("evidence_retry_count", 0)) <= MAX_EVIDENCE_RETRIES:
        # Hybrid questions can legitimately need both SQL and retrieval evidence
        # refreshed, so retry_targets is plural. Routing to hybrid re-runs both
        # evidence paths with the evaluator feedback in state.
        return "hybrid" if set(retry_targets) == {"analytical", "semantic"} else retry_targets[0]

    if retry_target in {"analytical", "semantic"} and int(state.get("evidence_retry_count", 0)) <= MAX_EVIDENCE_RETRIES:
        return retry_target

    if retry_target == "answer" and int(state.get("retry_count", 0)) <= MAX_ANSWER_RETRIES:
        return "generate_answer"

    return "compose"


def _build_graph():
    graph = StateGraph(MaintenanceState)
    graph.add_node("input_guardrail", input_guardrail_node)
    graph.add_node("classify", classify_node)
    graph.add_node("analytical", analytical_node)
    graph.add_node("semantic", semantic_node)
    graph.add_node("hybrid", hybrid_node)
    graph.add_node("generate_answer", answer_node)
    graph.add_node("evaluate_answer", evaluate_answer_node)
    graph.add_node("compose", compose_node)

    graph.add_edge(START, "input_guardrail")
    graph.add_conditional_edges(
        "input_guardrail",
        route_after_input_guardrail,
        {"classify": "classify", "compose": "compose"},
    )
    graph.add_conditional_edges(
        "classify",
        _route_node,
        {"analytical": "analytical", "semantic": "semantic", "hybrid": "hybrid"},
    )
    graph.add_edge("analytical", "generate_answer")
    graph.add_edge("semantic", "generate_answer")
    graph.add_edge("hybrid", "generate_answer")
    graph.add_edge("generate_answer", "evaluate_answer")
    graph.add_conditional_edges(
        "evaluate_answer",
        route_after_evaluation,
        {
            "generate_answer": "generate_answer",
            "analytical": "analytical",
            "semantic": "semantic",
            "hybrid": "hybrid",
            "compose": "compose",
        },
    )
    graph.add_edge("compose", END)
    return graph.compile()


_GRAPH_APP = _build_graph()


def get_graph_app():
    """Return the compiled LangGraph app used by the API layer."""

    return _GRAPH_APP
