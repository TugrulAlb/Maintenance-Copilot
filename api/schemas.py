"""Pydantic models for the Maintenance Copilot API."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field


class QuestionRequest(BaseModel):
    """Incoming user question and optional conversation thread id."""

    question: str
    thread_id: str | None = None


class AnswerResponse(BaseModel):
    """Structured API response returned by the LangGraph-backed assistant."""

    answer: str
    query_type: str
    thread_id: str
    node_trace: list[str]
    citations: list[str] = Field(default_factory=list)
    filters: dict[str, Any] = Field(default_factory=dict)
    router_confidence: float | None = None
    router_reasoning: str | None = None
    is_sufficient: bool | None = None
    evaluation_reasoning: str | None = None
    missing_aspects: list[str] | None = None
    retry_target: Literal["answer", "analytical", "semantic"] | None = None
    retry_targets: list[Literal["analytical", "semantic"]] | None = None
    retry_count: int = 0
    evidence_retry_count: int = 0
    hit_retry_cap: bool = False
