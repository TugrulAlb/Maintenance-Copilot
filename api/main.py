"""FastAPI app for Maintenance Copilot."""

from __future__ import annotations

import logging
import time
from typing import Any
from uuid import uuid4

from dotenv import load_dotenv
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from api.schemas import AnswerResponse, QuestionRequest
from graph.build_graph import get_graph_app


load_dotenv()

logger = logging.getLogger("maintenance_copilot.api")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

app = FastAPI(title="Maintenance Copilot API", version="0.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def classify_query_type(question: str) -> str:
    """Classify the question for logging and response metadata.

    This is intentionally lightweight so the API can log a stable query type even
    before the graph logic becomes more sophisticated.
    """

    lowered = question.lower()
    analytical_keywords = ["kaç", "how many", "trend", "ortalama", "rate", "en çok", "distribution"]
    semantic_keywords = ["neden", "why", "ne oldu", "what happened", "arıza", "fault", "error"]

    if any(keyword in lowered for keyword in analytical_keywords):
        return "analytical"
    if any(keyword in lowered for keyword in semantic_keywords):
        return "semantic"
    return "hybrid"


def _extract_response(payload: Any, fallback_thread_id: str, fallback_query_type: str) -> AnswerResponse:
    """Normalize graph output into the API response schema."""

    if isinstance(payload, dict):
        answer = str(payload.get("answer", payload.get("response", "")))
        query_type = str(payload.get("query_type", fallback_query_type))
        thread_id = str(payload.get("thread_id", fallback_thread_id))
        node_trace = payload.get("node_trace", payload.get("trace", []))
        if not isinstance(node_trace, list):
            node_trace = [str(node_trace)]
        return AnswerResponse(
            answer=answer,
            query_type=query_type,
            thread_id=thread_id,
            node_trace=[str(node) for node in node_trace],
        )

    return AnswerResponse(
        answer=str(payload),
        query_type=fallback_query_type,
        thread_id=fallback_thread_id,
        node_trace=[],
    )


@app.middleware("http")
async def request_logging_middleware(request: Request, call_next):
    """Log request timing and query classification for observability.

    Request logging matters in production AI systems because it helps debug which
    questions get misclassified, track latency regressions, and later feed real
    traffic into an evaluation pipeline.
    """

    start = time.perf_counter()
    response = await call_next(request)
    duration_ms = (time.perf_counter() - start) * 1000
    question = getattr(request.state, "question", None)
    query_type = getattr(request.state, "query_type", None)
    if question is not None:
        logger.info(
            "request path=%s query_type=%s duration_ms=%.2f question=%s",
            request.url.path,
            query_type,
            duration_ms,
            question,
        )
    return response


@app.get("/health")
def health() -> dict[str, str]:
    """Basic liveness/readiness endpoint for container orchestration."""

    return {"status": "ok"}


@app.post("/ask", response_model=AnswerResponse)
def ask(http_request: Request, request: QuestionRequest) -> AnswerResponse:
    """Invoke the compiled LangGraph app and return the assistant answer."""

    thread_id = request.thread_id or str(uuid4())
    query_type = classify_query_type(request.question)
    http_request.state.question = request.question
    http_request.state.query_type = query_type

    graph_app = get_graph_app()
    result = graph_app.invoke(
        {
            "question": request.question,
            "thread_id": thread_id,
            "query_type": query_type,
        },
        config={"configurable": {"thread_id": thread_id}},
    )

    return _extract_response(result, fallback_thread_id=thread_id, fallback_query_type=query_type)


@app.exception_handler(Exception)
def unexpected_error_handler(_: Request, exc: Exception) -> JSONResponse:
    """Convert unexpected failures into a JSON error response."""

    logger.exception("Unhandled API error: %s", exc)
    return JSONResponse(status_code=500, content={"detail": "Internal server error"})