"""Prometheus metrics for the Maintenance Copilot API."""

from __future__ import annotations

from prometheus_client import CollectorRegistry, Counter, Histogram, generate_latest

from api.schemas import AnswerResponse


REGISTRY = CollectorRegistry()

REQUEST_COUNT = Counter(
    "maintenance_requests_total",
    "Total number of API requests.",
    ["path", "method", "status_code", "query_type"],
    registry=REGISTRY,
)
REQUEST_LATENCY = Histogram(
    "maintenance_request_latency_seconds",
    "Request latency in seconds.",
    ["path", "method", "status_code", "query_type"],
    registry=REGISTRY,
)
GRAPH_NODE_VISITS = Counter(
    "maintenance_graph_node_visits_total",
    "LangGraph node visits by node name and query intent.",
    ["node_name", "query_type"],
    registry=REGISTRY,
)
GUARDRAIL_BLOCKS = Counter(
    "maintenance_guardrail_blocks_total",
    "Guardrail blocks by reason.",
    ["reason"],
    registry=REGISTRY,
)
GRAPH_RETRIES = Counter(
    "maintenance_graph_retries_total",
    "Retry occurrences by retry type.",
    ["retry_type", "query_type"],
    registry=REGISTRY,
)


def record_http_request(path: str, method: str, status_code: int, query_type: str, duration_seconds: float) -> None:
    REQUEST_COUNT.labels(path=path, method=method, status_code=str(status_code), query_type=query_type).inc()
    REQUEST_LATENCY.labels(path=path, method=method, status_code=str(status_code), query_type=query_type).observe(duration_seconds)


def record_graph_response(response: AnswerResponse) -> None:
    query_type = response.query_type or "unknown"
    for node_name in response.node_trace:
        GRAPH_NODE_VISITS.labels(node_name=node_name, query_type=query_type).inc()
    if response.input_blocked:
        GUARDRAIL_BLOCKS.labels(reason=response.block_reason or "unknown").inc()
    if response.retry_count:
        GRAPH_RETRIES.labels(retry_type="answer", query_type=query_type).inc(response.retry_count)
    if response.evidence_retry_count:
        GRAPH_RETRIES.labels(retry_type="evidence", query_type=query_type).inc(response.evidence_retry_count)


def render_metrics() -> str:
    """Return Prometheus exposition text.

    In production, Prometheus would scrape this service's /metrics endpoint and
    Grafana would read those series for panels such as request latency, guardrail
    blocks by reason, and retry counts by graph intent.
    """

    return generate_latest(REGISTRY).decode("utf-8")
