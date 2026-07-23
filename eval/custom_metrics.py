"""Small custom evaluation metrics for Maintenance Copilot."""

from __future__ import annotations

import json
from typing import Any

from graph.llm_client import build_chat_client, chat_model


def _flatten_evidence(evidence: list[dict[str, Any]]) -> str:
    chunks: list[str] = []
    for item in evidence:
        if item.get("kind") == "sql":
            chunks.append(
                json.dumps(
                    {
                        "kind": "sql",
                        "sql_query": item.get("sql_query"),
                        "rows": item.get("rows", []),
                    },
                    ensure_ascii=False,
                )
            )
        elif item.get("kind") == "retrieval":
            for result in item.get("items", []):
                chunks.append(str(result.get("document", "")))
    return "\n\n".join(chunk for chunk in chunks if chunk)


def judge_hallucination(question: str, answer: str, evidence: list[dict[str, Any]]) -> dict[str, Any]:
    """Return whether the answer contains claims unsupported by evidence."""

    evidence_text = _flatten_evidence(evidence)
    if not answer.strip():
        return {"hallucination_flag": True, "reasoning": "Answer is empty."}
    if not evidence_text.strip():
        return {"hallucination_flag": True, "reasoning": "No evidence was available to support the answer."}

    client = build_chat_client()
    if client is None:
        return {
            "hallucination_flag": False,
            "reasoning": "Skipped LLM judge because no OpenAI/Azure credentials are configured.",
        }

    response = client.chat.completions.create(
        model=chat_model("AZURE_OPENAI_EVALUATOR_DEPLOYMENT_NAME", "OPENAI_EVALUATOR_MODEL"),
        temperature=0,
        response_format={"type": "json_object"},
        messages=[
            {
                "role": "system",
                "content": (
                    "You are a strict hallucination judge. Decide whether the answer contains any substantive "
                    "claim that is not present in or directly supported by the evidence. Return JSON with "
                    "hallucination_flag and reasoning."
                ),
            },
            {
                "role": "user",
                "content": json.dumps(
                    {
                        "question": question,
                        "answer": answer,
                        "evidence": evidence_text,
                    },
                    ensure_ascii=False,
                ),
            },
        ],
    )

    try:
        payload = json.loads(response.choices[0].message.content or "{}")
    except json.JSONDecodeError:
        return {"hallucination_flag": True, "reasoning": "Judge returned invalid JSON."}

    return {
        "hallucination_flag": bool(payload.get("hallucination_flag", False)),
        "reasoning": str(payload.get("reasoning", "")),
    }


def hallucination_flag_rate(rows: list[dict[str, Any]]) -> float:
    """Return percentage of rows where the custom judge flagged hallucination."""

    if not rows:
        return 0.0
    flagged = sum(1 for row in rows if row.get("hallucination_flag"))
    return round(flagged / len(rows), 4)
