"""Final answer generation for Maintenance Copilot."""

from __future__ import annotations

import json

from dotenv import load_dotenv
from openai import AzureOpenAI, OpenAI

from graph.llm_client import build_chat_client, chat_model


load_dotenv()


def _build_client() -> AzureOpenAI | OpenAI | None:
    return build_chat_client()


def _deterministic_answer(state: dict[str, object]) -> tuple[str, list[str]]:
    query_type = state.get("query_type", "hybrid")
    citations = list(state.get("citations", []))
    history = state.get("conversation_history", [])
    if query_type in {"analytical", "hybrid"}:
        sql_rows = state.get("sql_rows", [])
        if sql_rows:
            preview = sql_rows[:3]
            parts = []
            for row in preview:
                parts.append(
                    f"{row.get('production_line', '?')} / {row.get('fault_category', '?')} / {row.get('machine_id', '?')}"
                )
            answer = f"Analytical sonuç: {len(sql_rows)} kayıt bulundu. İlk örnekler: {', '.join(parts)}."
        else:
            answer = "Analytical sorgu çalıştı ancak sonuç döndürmedi."
        if query_type == "hybrid" and state.get("results"):
            top = state["results"][0]
            metadata = top.get("metadata", {})
            answer += (
                f" Semantic olarak en yakın kayıt {metadata.get('production_line', '?')} hattında, "
                f"{metadata.get('fault_category', '?')} ile ilgili."
            )
    else:
        results = state.get("results", [])
        if results:
            top = results[0]
            metadata = top.get("metadata", {})
            answer = (
                f"En ilgili kayıt {metadata.get('production_line', '?')} hattında, "
                f"{metadata.get('fault_category', '?')} ile ilgili."
            )
            if history:
                last_turn = history[-1]
                answer += f" Önceki konuşma: '{last_turn.get('question', '')}'."
        else:
            answer = "İlgili bakım kaydı bulunamadı."
    return answer, citations


def generate_answer(state: dict[str, object]) -> dict[str, object]:
    """Generate the final user-facing answer, preferably via an LLM."""

    client = _build_client()
    if client is None:
        answer, citations = _deterministic_answer(state)
        return {"answer": answer, "citations": citations}

    model = chat_model("AZURE_OPENAI_ANSWER_DEPLOYMENT_NAME", "OPENAI_ANSWER_MODEL")
    response = client.chat.completions.create(
        model=model,
        temperature=0.2,
        response_format={"type": "json_object"},
        messages=[
            {
                "role": "system",
                "content": (
                    "You are an assistant for industrial maintenance analytics. "
                    "Answer using the provided evidence and include citations from the citations array. "
                    "If evaluator feedback is present, fix those specific gaps while staying grounded in evidence. "
                    "Return JSON with keys answer and citations."
                ),
            },
            {
                "role": "user",
                "content": json.dumps(
                    {
                        "question": state.get("question"),
                        "query_type": state.get("query_type"),
                        "router_confidence": state.get("router_confidence"),
                        "router_reasoning": state.get("router_reasoning"),
                        "filters": state.get("filters", {}),
                        # Passing targeted reflection feedback prevents a naive
                        # retry from simply producing the same flawed answer again.
                        "previous_evaluation_reasoning": state.get("evaluation_reasoning"),
                        "previous_missing_aspects": state.get("missing_aspects", []),
                        "retry_target": state.get("retry_target"),
                        "retry_targets": state.get("retry_targets", []),
                        "retry_count": state.get("retry_count", 0),
                        "evidence_retry_count": state.get("evidence_retry_count", 0),
                        "conversation_history": state.get("conversation_history", []),
                        "evidence": state.get("evidence", []),
                        "sql_rows": state.get("sql_rows", []),
                        "candidates": state.get("results", []),
                        "citations": state.get("citations", []),
                    },
                    ensure_ascii=False,
                ),
            },
        ],
    )

    raw_text = response.choices[0].message.content or "{}"
    try:
        payload = json.loads(raw_text)
        answer = str(payload.get("answer", "")).strip()
        citations = payload.get("citations", state.get("citations", []))
        if not isinstance(citations, list):
            citations = list(state.get("citations", []))
        if not answer:
            raise ValueError("Empty answer")
        return {"answer": answer, "citations": [str(item) for item in citations]}
    except Exception:
        answer, citations = _deterministic_answer(state)
        return {"answer": answer, "citations": citations}
