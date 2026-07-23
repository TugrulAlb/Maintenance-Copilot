"""FastAPI app for Maintenance Copilot.

Security posture: API-key auth, RBAC, rate limiting, and explicit CORS provide
reasonable production hygiene for this portfolio-scale internal tool. A real
Siemens-scale deployment would likely add OAuth2/OIDC through an enterprise
identity provider, mTLS between services, and a proper API gateway in front of
this service.
"""

import json
import logging
import os
import time
from typing import Any
from uuid import uuid4

from dotenv import load_dotenv
from fastapi import Body, Depends, FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse, Response
from slowapi.errors import RateLimitExceeded

from api.auth import AccessContext, require_role
from api.conversation_store import CONVERSATIONS
from api.metrics import record_graph_response, record_http_request, render_metrics
from api.rate_limit import ask_rate_limit_rule, limiter
from api.schemas import AnswerResponse, QuestionRequest
from api.ui import render_index_html
from graph.build_graph import get_graph_app


load_dotenv()

logger = logging.getLogger("maintenance_copilot.api")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

app = FastAPI(title="Maintenance Copilot API", version="0.1.0")
app.state.limiter = limiter


def _cors_origins() -> list[str]:
    # Demo-stage wildcard CORS is convenient but unacceptable once API keys or
    # credentials exist: a malicious origin could trick a browser into sending
    # authenticated requests. Production CORS must be an explicit allow-list.
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


@app.exception_handler(RateLimitExceeded)
def rate_limit_exceeded_handler(_: Request, exc: RateLimitExceeded) -> JSONResponse:
    return JSONResponse(
        status_code=429,
        content={"detail": "Rate limit exceeded"},
        headers={"Retry-After": "60", "X-RateLimit-Limit": str(exc.limit.limit)},
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
            input_blocked=bool(payload.get("input_blocked", False)),
            block_reason=str(payload.get("block_reason")) if payload.get("block_reason") else None,
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
            output_redactions=[str(item) for item in payload.get("output_redactions", [])] if isinstance(payload.get("output_redactions", []), list) else [],
        )

    return AnswerResponse(
        answer=str(payload),
        query_type=fallback_query_type,
        thread_id=fallback_thread_id,
        node_trace=[],
        citations=[],
    )


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
        response = await call_next(request)
        status_code = getattr(response, "status_code", 500)
    except HTTPException as exc:
        status_code = exc.status_code
        response = JSONResponse(status_code=exc.status_code, content={"detail": exc.detail}, headers=exc.headers)
    duration_seconds = time.perf_counter() - start
    question = getattr(request.state, "question", None)
    query_type = str(getattr(request.state, "query_type", None) or query_type)
    response.headers["X-Request-ID"] = request_id

    record_http_request(request.url.path, request.method, status_code, query_type, duration_seconds)

    logger.info(
        json.dumps(
            {
                "event": "request",
                "request_id": request_id,
                "path": request.url.path,
                "method": request.method,
                "status_code": status_code,
                "duration_ms": round(duration_seconds * 1000, 2),
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
def metrics(_: AccessContext = Depends(require_role(["admin"]))) -> Response:
    """Expose Prometheus-style metrics for scraping."""

    return Response(content=render_metrics(), media_type="text/plain; version=0.0.4")


@app.post("/ask", response_model=AnswerResponse)
@limiter.limit(ask_rate_limit_rule)
def ask(
    request: Request,
    question_request: QuestionRequest = Body(...),
    _: AccessContext = Depends(require_role(["viewer", "admin"], allow_public=True)),
) -> AnswerResponse:
    """Invoke the compiled LangGraph app and return the assistant answer."""

    thread_id = question_request.thread_id or str(uuid4())
    request.state.question = question_request.question
    conversation_history = CONVERSATIONS.get_history(thread_id)

    graph_app = get_graph_app()
    result = graph_app.invoke(
        {
            "question": question_request.question,
            "thread_id": thread_id,
            "conversation_history": conversation_history,
        },
        config={"configurable": {"thread_id": thread_id}},
    )

    response = _extract_response(result, fallback_thread_id=thread_id, fallback_query_type="hybrid")
    request.state.query_type = response.query_type
    record_graph_response(response)
    CONVERSATIONS.append_turn(thread_id, question_request.question, response.answer, response.query_type, response.citations)
    return response


@app.exception_handler(Exception)
def unexpected_error_handler(_: Request, exc: Exception) -> JSONResponse:
    """Convert unexpected failures into a JSON error response."""

    logger.exception("Unhandled API error: %s", exc)
    return JSONResponse(status_code=500, content={"detail": "Internal server error"})
