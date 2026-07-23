"""FastAPI app for Maintenance Copilot."""

from __future__ import annotations

import json
import logging
import os
import time
from collections import defaultdict
from threading import Lock
from typing import Any
from uuid import uuid4

from dotenv import load_dotenv
from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse, JSONResponse, Response
from fastapi.middleware.cors import CORSMiddleware

from api.auth import AccessContext, require_roles
from api.conversation_store import CONVERSATIONS
from api.rate_limit import enforce_rate_limit
from api.ui import render_index_html
from api.schemas import AnswerResponse, QuestionRequest
from graph.build_graph import get_graph_app


load_dotenv()

logger = logging.getLogger("maintenance_copilot.api")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

app = FastAPI(title="Maintenance Copilot API", version="0.1.0")

_METRICS_LOCK = Lock()
_REQUEST_COUNTS: dict[tuple[str, str, str, str], int] = defaultdict(int)
_REQUEST_DURATION_MS_SUM: dict[tuple[str, str, str, str], float] = defaultdict(float)
_REQUEST_DURATION_MS_COUNT: dict[tuple[str, str, str, str], int] = defaultdict(int)

def _cors_origins() -> list[str]:
    origins = [
        origin.strip()
        for origin in os.getenv(
            "MAINTENANCE_COPILOT_CORS_ORIGINS",
            "http://localhost:3000,http://127.0.0.1:3000",
        ).split(",")
        if origin.strip()
    ]
    if "*" in origins:
        raise RuntimeError("Wildcard CORS is not allowed. Set explicit MAINTENANCE_COPILOT_CORS_ORIGINS.")
    return origins


app.add_middleware(
    CORSMiddleware,
    allow_origins=_cors_origins(),
    allow_credentials=False,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["Content-Type", "Authorization", "X-API-Key", "X-Request-ID", "X-User-Role"],
)


def _extract_response(payload: Any, fallback_thread_id: str, fallback_query_type: str) -> AnswerResponse:
    """Normalize graph output into the API response schema."""

    if isinstance(payload, dict):
        answer = str(payload.get("answer", payload.get("response", "")))
        query_type = str(payload.get("query_type", fallback_query_type))
        thread_id = str(payload.get("thread_id", fallback_thread_id))
        node_trace = payload.get("node_trace", payload.get("trace", []))
        if not isinstance(node_trace, list):
            node_trace = [str(node_trace)]
        filters = payload.get("filters", {})
        if not isinstance(filters, dict):
            filters = {}
        return AnswerResponse(
            answer=answer,
            query_type=query_type,
            thread_id=thread_id,
            node_trace=[str(node) for node in node_trace],
            citations=[str(item) for item in payload.get("citations", [])] if isinstance(payload.get("citations", []), list) else [],
            filters=filters,
            router_confidence=payload.get("router_confidence") if isinstance(payload.get("router_confidence"), (float, int)) else None,
            router_reasoning=str(payload.get("router_reasoning")) if payload.get("router_reasoning") else None,
            is_sufficient=payload.get("is_sufficient") if isinstance(payload.get("is_sufficient"), bool) else None,
            evaluation_reasoning=str(payload.get("evaluation_reasoning")) if payload.get("evaluation_reasoning") else None,
            missing_aspects=[str(item) for item in payload.get("missing_aspects", [])] if isinstance(payload.get("missing_aspects"), list) else None,
            retry_target=payload.get("retry_target") if payload.get("retry_target") in {"answer", "analytical", "semantic"} else None,
            retry_targets=[str(item) for item in payload.get("retry_targets", [])] if isinstance(payload.get("retry_targets"), list) else None,
            retry_count=int(payload.get("retry_count", 0)) if isinstance(payload.get("retry_count", 0), int) else 0,
            evidence_retry_count=int(payload.get("evidence_retry_count", 0)) if isinstance(payload.get("evidence_retry_count", 0), int) else 0,
            hit_retry_cap=bool(payload.get("hit_retry_cap", False)),
        )

    return AnswerResponse(
        answer=str(payload),
        query_type=fallback_query_type,
        thread_id=fallback_thread_id,
        node_trace=[],
        citations=[],
    )


def _record_request_metric(path: str, method: str, status_code: int, query_type: str, duration_ms: float) -> None:
    """Record request metrics in memory for later /metrics scraping."""

    key = (path, method, str(status_code), query_type)
    with _METRICS_LOCK:
        _REQUEST_COUNTS[key] += 1
        _REQUEST_DURATION_MS_SUM[key] += duration_ms
        _REQUEST_DURATION_MS_COUNT[key] += 1


def _render_metrics() -> str:
    """Render Prometheus-style metrics text without an extra dependency."""

    lines = [
        "# HELP maintenance_requests_total Total number of API requests.",
        "# TYPE maintenance_requests_total counter",
        "# HELP maintenance_request_duration_ms Average request duration in milliseconds.",
        "# TYPE maintenance_request_duration_ms gauge",
    ]
    with _METRICS_LOCK:
        for (path, method, status_code, query_type), count in sorted(_REQUEST_COUNTS.items()):
            labels = f'path="{path}",method="{method}",status_code="{status_code}",query_type="{query_type}"'
            avg = _REQUEST_DURATION_MS_SUM[(path, method, status_code, query_type)] / max(
                1, _REQUEST_DURATION_MS_COUNT[(path, method, status_code, query_type)]
            )
            lines.append(f"maintenance_requests_total{{{labels}}} {count}")
            lines.append(f"maintenance_request_duration_ms{{{labels}}} {avg:.3f}")
    return "\n".join(lines) + "\n"


@app.middleware("http")
async def request_logging_middleware(request: Request, call_next):
    """Log request timing and query classification for observability.

    Request logging matters in production AI systems because it helps debug which
    questions get misclassified, track latency regressions, and later feed real
    traffic into an evaluation pipeline.
    """

    request_id = str(uuid4())
    request.state.request_id = request_id
    start = time.perf_counter()
    query_type = "unknown"
    status_code = 500
    try:
        enforce_rate_limit(request)
        response = await call_next(request)
        status_code = getattr(response, "status_code", 500)
    except HTTPException as exc:
        status_code = exc.status_code
        response = JSONResponse(status_code=exc.status_code, content={"detail": exc.detail}, headers=exc.headers)
    duration_ms = (time.perf_counter() - start) * 1000
    question = getattr(request.state, "question", None)
    query_type = str(getattr(request.state, "query_type", None) or query_type)
    response.headers["X-Request-ID"] = request_id

    _record_request_metric(request.url.path, request.method, status_code, query_type, duration_ms)

    logger.info(
        json.dumps(
            {
                "event": "request",
                "request_id": request_id,
                "path": request.url.path,
                "method": request.method,
                "status_code": status_code,
                "duration_ms": round(duration_ms, 2),
                "query_type": query_type,
                "question": question,
            },
            ensure_ascii=False,
        )
    )
    return response


@app.get("/health")
def health() -> dict[str, str]:
    """Basic liveness/readiness endpoint for container orchestration."""

    return {"status": "ok"}


@app.get("/", response_class=HTMLResponse)
def index() -> str:
    """Serve a minimal browser UI for demo and portfolio use."""

    return render_index_html()


@app.get("/metrics")
def metrics(_: AccessContext = Depends(require_roles("admin"))) -> Response:
    """Expose lightweight Prometheus-style metrics for scraping."""

    return Response(content=_render_metrics(), media_type="text/plain; version=0.0.4")


@app.post("/ask", response_model=AnswerResponse)
def ask(
    http_request: Request,
    request: QuestionRequest,
    _: AccessContext = Depends(require_roles("user", "admin")),
) -> AnswerResponse:
    """Invoke the compiled LangGraph app and return the assistant answer."""

    thread_id = request.thread_id or str(uuid4())
    http_request.state.question = request.question
    conversation_history = CONVERSATIONS.get_history(thread_id)

    graph_app = get_graph_app()
    result = graph_app.invoke(
        {
            "question": request.question,
            "thread_id": thread_id,
            "conversation_history": conversation_history,
        },
        config={"configurable": {"thread_id": thread_id}},
    )

    response = _extract_response(result, fallback_thread_id=thread_id, fallback_query_type="hybrid")
    http_request.state.query_type = response.query_type
    CONVERSATIONS.append_turn(thread_id, request.question, response.answer, response.query_type, response.citations)
    return response


@app.exception_handler(Exception)
def unexpected_error_handler(_: Request, exc: Exception) -> JSONResponse:
    """Convert unexpected failures into a JSON error response."""

    logger.exception("Unhandled API error: %s", exc)
    return JSONResponse(status_code=500, content={"detail": "Internal server error"})
