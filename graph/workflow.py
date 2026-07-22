"""Workflow assembly for the Maintenance Copilot LangGraph app."""

from __future__ import annotations

from langgraph.graph import END, START, StateGraph

from graph.nodes import analytical_node, answer_node, classify_node, compose_node, hybrid_node, semantic_node
from graph.state import MaintenanceState


def _route_node(state: MaintenanceState) -> str:
    query_type = state.get("query_type")
    if query_type in {"analytical", "semantic", "hybrid"}:
        return query_type
    return "hybrid"


def _build_graph():
    graph = StateGraph(MaintenanceState)
    graph.add_node("classify", classify_node)
    graph.add_node("analytical", analytical_node)
    graph.add_node("semantic", semantic_node)
    graph.add_node("hybrid", hybrid_node)
    graph.add_node("answer", answer_node)
    graph.add_node("compose", compose_node)

    graph.add_edge(START, "classify")
    graph.add_conditional_edges(
        "classify",
        _route_node,
        {"analytical": "analytical", "semantic": "semantic", "hybrid": "hybrid"},
    )
    graph.add_edge("analytical", "answer")
    graph.add_edge("semantic", "answer")
    graph.add_edge("hybrid", "answer")
    graph.add_edge("answer", "compose")
    graph.add_edge("compose", END)
    return graph.compile()


_GRAPH_APP = _build_graph()


def get_graph_app():
    """Return the compiled LangGraph app used by the API layer."""

    return _GRAPH_APP
