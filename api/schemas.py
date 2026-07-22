"""Pydantic models for the Maintenance Copilot API."""

from __future__ import annotations

from pydantic import BaseModel


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
    citations: list[str] = []