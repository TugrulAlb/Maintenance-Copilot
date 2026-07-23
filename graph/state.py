"""State definitions for the Maintenance Copilot graph."""

from __future__ import annotations

from typing import Any, Literal, TypedDict


class MaintenanceState(TypedDict, total=False):
    """State passed between graph nodes."""

    question: str
    thread_id: str
    query_type: str
    router_confidence: float
    router_reasoning: str
    answer: str
    node_trace: list[str]
    filters: dict[str, Any]
    is_sufficient: bool | None
    evaluation_reasoning: str | None
    missing_aspects: list[str] | None
    retry_target: Literal["answer", "analytical", "semantic"] | None
    retry_targets: list[Literal["analytical", "semantic"]] | None
    retry_count: int
    evidence_retry_count: int
    hit_retry_cap: bool
    conversation_history: list[dict[str, object]]
    sql_rows: list[dict[str, object]]
    sql_query: str
    evidence: list[dict[str, object]]
    candidates: list[dict[str, object]]
    results: list[dict[str, object]]
    citations: list[str]
