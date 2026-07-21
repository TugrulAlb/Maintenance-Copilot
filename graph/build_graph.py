"""Minimal LangGraph app surface for the API layer.

This file provides the same import target the FastAPI app expects. The real graph
can be expanded later without changing the API contract or the tests.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass
class _CompiledGraphApp:
    """Tiny stand-in that matches the compiled LangGraph invoke interface."""

    def invoke(self, payload: dict[str, Any], config: dict[str, Any] | None = None) -> dict[str, Any]:
        question = payload.get("question", "")
        thread_id = payload.get("thread_id", "")
        query_type = payload.get("query_type", "hybrid")
        return {
            "answer": f"Stub answer for: {question}",
            "query_type": query_type,
            "thread_id": thread_id,
            "node_trace": ["ingest", "retrieve", "rerank", "answer"],
        }


def get_graph_app() -> _CompiledGraphApp:
    """Return the compiled graph app object expected by the API."""

    return _CompiledGraphApp()