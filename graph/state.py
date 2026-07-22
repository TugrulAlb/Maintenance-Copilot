"""State definitions for the Maintenance Copilot graph."""

from __future__ import annotations

from typing import Any, TypedDict


class MaintenanceState(TypedDict, total=False):
    """State passed between graph nodes."""

    question: str
    thread_id: str
    query_type: str
    answer: str
    node_trace: list[str]
    filters: dict[str, Any]
    conversation_history: list[dict[str, object]]
    sql_rows: list[dict[str, object]]
    sql_query: str
    evidence: list[dict[str, object]]
    candidates: list[dict[str, object]]
    results: list[dict[str, object]]
    citations: list[str]
