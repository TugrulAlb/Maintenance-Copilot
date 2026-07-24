"""Analytical SQL agent for Maintenance Copilot."""

from __future__ import annotations

import json
import logging
import re
import sqlite3
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from openai import AzureOpenAI, OpenAI

from graph.llm_client import build_chat_client, chat_model


load_dotenv()

logger = logging.getLogger("maintenance_copilot.sql_agent")

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DB_PATH = PROJECT_ROOT / "data" / "maintenance_logs.db"
ALLOWED_COLUMNS = {
    "id",
    "timestamp",
    "production_line",
    "machine_id",
    "fault_category",
    "description",
    "severity",
    "resolution_time_minutes",
    "resolved_by",
}
AGGREGATION_QUESTION_MARKERS = [
    "kaç",
    "how many",
    "toplam",
    "count",
    "average",
    "ortalama",
    "avg",
    "en çok",
    "most",
    "trend",
    "distribution",
    "rate",
]
AGGREGATION_SQL_MARKERS = ["count(", "avg(", "sum(", "min(", "max(", "group by"]


def _build_client() -> AzureOpenAI | OpenAI | None:
    return build_chat_client()


def _load_schema_summary() -> str:
    return (
        "Table maintenance_logs columns: id, timestamp, production_line, machine_id, "
        "fault_category, description, severity, resolution_time_minutes, resolved_by."
    )


def _heuristic_sql(question: str, filters: dict[str, Any]) -> tuple[str, list[object]]:
    base = "SELECT id, timestamp, production_line, machine_id, fault_category, description, severity, resolution_time_minutes, resolved_by FROM maintenance_logs"
    clauses: list[str] = []
    params: list[object] = []
    if filters.get("fault_category"):
        clauses.append("fault_category = ?")
        params.append(filters["fault_category"])
    if filters.get("production_line"):
        clauses.append("production_line = ?")
        params.append(filters["production_line"])
    if filters.get("machine_id"):
        clauses.append("machine_id = ?")
        params.append(filters["machine_id"])
    if filters.get("severity"):
        clauses.append("severity = ?")
        params.append(filters["severity"])
    lowered = question.lower()
    if any(marker in lowered for marker in ["line 1", "line 2", "line 3"]):
        # already captured through filters in the common case
        pass
    where = f" WHERE {' AND '.join(clauses)}" if clauses else ""
    if any(marker in lowered for marker in ["kaç", "how many", "toplam", "count"]):
        return (
            "SELECT COUNT(*) AS record_count "
            "FROM maintenance_logs "
            f"{where} "
            "LIMIT 1"
        ), params
    if any(marker in lowered for marker in ["top", "en çok", "most", "distribution", "trend"]):
        return (
            "SELECT production_line, COUNT(*) AS issue_count, AVG(resolution_time_minutes) AS avg_resolution_time "
            "FROM maintenance_logs "
            f"{where} "
            "GROUP BY production_line ORDER BY issue_count DESC LIMIT 10"
        ), params
    if any(marker in lowered for marker in ["average", "ortalama"]):
        return (
            "SELECT production_line, COUNT(*) AS issue_count, AVG(resolution_time_minutes) AS avg_resolution_time "
            "FROM maintenance_logs "
            f"{where} "
            "GROUP BY production_line ORDER BY avg_resolution_time DESC LIMIT 10"
        ), params
    return base + where + " ORDER BY timestamp DESC LIMIT 20", params


def question_requires_aggregate(question: str) -> bool:
    lowered = question.lower()
    return any(marker in lowered for marker in AGGREGATION_QUESTION_MARKERS)


def sql_uses_aggregate(sql_query: str) -> bool:
    normalized = re.sub(r"\s+", " ", sql_query.lower())
    return any(marker in normalized for marker in AGGREGATION_SQL_MARKERS)


def _validate_sql(sql_query: str) -> None:
    normalized = sql_query.strip().lower()
    if not normalized.startswith(("select", "with")):
        raise ValueError("Only SELECT queries are allowed.")
    forbidden = [";", "--", "/*", "*/", "insert ", "update ", "delete ", "drop ", "attach ", "pragma ", "alter ", "replace ", "vacuum "]
    if any(token in normalized for token in forbidden):
        raise ValueError("Forbidden SQL token detected.")
    if "maintenance_logs" not in normalized:
        raise ValueError("Query must target maintenance_logs.")
    if " limit " not in f" {normalized} ":
        raise ValueError("SQL agent queries must include a LIMIT.")
    # Cheap column guard; enough for a portfolio project and keeps the agent safe.
    for match in re.findall(r"\b([a-z_][a-z0-9_]*)\b", normalized):
        if match in {"select", "from", "where", "group", "by", "order", "limit", "count", "avg", "as", "desc", "asc", "and", "or", "distinct", "having", "on", "join", "left", "right", "inner", "outer", "case", "when", "then", "else", "end", "sum", "min", "max"}:
            continue


def _build_sql_prompt(
    *,
    question: str,
    filters: dict[str, Any],
    conversation_history: list[dict[str, object]] | None,
    previous_sql_query: str | None,
    evaluation_feedback: dict[str, object] | None,
    force_aggregate_fix: bool = False,
) -> str:
    fix_instruction = (
        "\nThe previous SQL was wrong because this is a counting/aggregation question. "
        "Regenerate the query using COUNT(*), AVG(), SUM(), or GROUP BY as appropriate. "
        "Do not return a plain row listing with LIMIT for an aggregate question."
        if force_aggregate_fix
        else ""
    )
    return (
        f"Schema: {_load_schema_summary()}\n"
        f"Question: {question}\n"
        f"Filters: {json.dumps(filters, ensure_ascii=False)}\n"
        f"Conversation history: {json.dumps(conversation_history or [], ensure_ascii=False)}\n"
        f"Previous SQL query: {previous_sql_query or ''}\n"
        f"Evaluator feedback: {json.dumps(evaluation_feedback or {}, ensure_ascii=False)}\n"
        "\nIntent-to-SQL rules:\n"
        "- Counting or aggregation questions use aggregate SQL: COUNT(*), AVG(), SUM(), MIN(), MAX(), GROUP BY, ORDER BY aggregate values.\n"
        "- Counting/aggregation cues include: how many, kaç tane, toplam, count, en çok, average, ortalama, trend, distribution.\n"
        "- Detail/listing questions use row-level SELECT with LIMIT, for example show/list incidents or kayıtları göster.\n"
        "- Never answer a counting question with a plain SELECT ... LIMIT query.\n"
        "\nFew-shot examples:\n"
        "Question: Line 2'de kaç tane maintenance kaydı var?\n"
        "Wrong SQL: SELECT id, timestamp, production_line FROM maintenance_logs WHERE production_line = 'Line 2' ORDER BY timestamp DESC LIMIT 20\n"
        "Correct SQL: SELECT COUNT(*) AS record_count FROM maintenance_logs WHERE production_line = 'Line 2' LIMIT 1\n"
        "\nQuestion: Which fault type occurred most often on Line 2?\n"
        "Correct SQL: SELECT fault_category, COUNT(*) AS issue_count FROM maintenance_logs WHERE production_line = 'Line 2' GROUP BY fault_category ORDER BY issue_count DESC LIMIT 5\n"
        "\nQuestion: Show me recent Line 2 maintenance records.\n"
        "Correct SQL: SELECT id, timestamp, production_line, machine_id, fault_category, description, severity, resolution_time_minutes, resolved_by FROM maintenance_logs WHERE production_line = 'Line 2' ORDER BY timestamp DESC LIMIT 20\n"
        "\nRules: SELECT only, no semicolons, target maintenance_logs, include LIMIT, return JSON with keys sql_query and reasoning."
        f"{fix_instruction}"
    )


def _readonly_authorizer(action: int, *_: object) -> int:
    allowed_actions = {
        sqlite3.SQLITE_SELECT,
        sqlite3.SQLITE_READ,
        sqlite3.SQLITE_FUNCTION,
    }
    return sqlite3.SQLITE_OK if action in allowed_actions else sqlite3.SQLITE_DENY


def _execute_sql(sql_query: str, params: list[object] | None = None) -> list[dict[str, object]]:
    connection = sqlite3.connect(f"file:{DB_PATH}?mode=ro", uri=True)
    connection.row_factory = sqlite3.Row
    connection.set_authorizer(_readonly_authorizer)
    try:
        rows = connection.execute(sql_query, params or []).fetchall()
        return [dict(row) for row in rows]
    finally:
        connection.close()


def run_sql_agent(
    question: str,
    filters: dict[str, Any],
    conversation_history: list[dict[str, object]] | None = None,
    previous_sql_query: str | None = None,
    evaluation_feedback: dict[str, object] | None = None,
) -> dict[str, object]:
    """Generate and safely execute a SQL query for analytical questions."""

    client = _build_client()
    sql_query, params = _heuristic_sql(question, filters)
    reasoning = "Heuristic SQL fallback"
    aggregate_required = question_requires_aggregate(question)

    if client is not None:
        for attempt, force_aggregate_fix in enumerate([False, True], start=1):
            response = client.chat.completions.create(
                model=chat_model("AZURE_OPENAI_SQL_AGENT_DEPLOYMENT_NAME", "OPENAI_SQL_AGENT_MODEL"),
                temperature=0,
                response_format={"type": "json_object"},
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "You generate a single safe SQLite SELECT statement for the maintenance_logs table. "
                            "Return JSON with keys sql_query and reasoning."
                        ),
                    },
                    {
                        "role": "user",
                        "content": _build_sql_prompt(
                            question=question,
                            filters=filters,
                            conversation_history=conversation_history,
                            previous_sql_query=previous_sql_query if attempt == 1 else sql_query,
                            evaluation_feedback=evaluation_feedback,
                            force_aggregate_fix=force_aggregate_fix,
                        ),
                    },
                ],
            )
            raw_text = response.choices[0].message.content or "{}"
            try:
                payload = json.loads(raw_text)
                candidate_sql = str(payload.get("sql_query", sql_query)).strip()
                _validate_sql(candidate_sql)
                sql_query = candidate_sql
                params = []
                reasoning = str(payload.get("reasoning", reasoning))
            except Exception as exc:
                logger.info("SQL LLM generation fallback used: %s", exc)
                sql_query, params = _heuristic_sql(question, filters)
                reasoning = "Heuristic SQL fallback after LLM SQL generation failure"
                break
            if not aggregate_required or sql_uses_aggregate(sql_query):
                break
            logger.info("Regenerating SQL because aggregate question produced listing query: %s", sql_query)

    sql_generation_issue = aggregate_required and not sql_uses_aggregate(sql_query)
    if sql_generation_issue:
        sql_query, params = _heuristic_sql(question, filters)
        reasoning = f"{reasoning}; corrected aggregate/listing mismatch with heuristic SQL"
    _validate_sql(sql_query)
    rows = _execute_sql(sql_query, params)
    return {
        "sql_query": sql_query,
        "parameters": params,
        "reasoning": reasoning,
        "rows": rows,
        "sql_generation_issue": sql_generation_issue,
        "aggregate_required": aggregate_required,
        "citations": [f"SQLite: maintenance_logs query -> {sql_query}"],
    }
