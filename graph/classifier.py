"""Intent classification for Maintenance Copilot.

The preferred path is an LLM-based classifier so the system does not rely on
hard-coded keyword lists for intent routing. If no model credentials are
available, we fall back to a small heuristic classifier so local development
and tests still work offline.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from functools import lru_cache
from typing import Any

from dotenv import load_dotenv
from openai import AzureOpenAI, OpenAI

from graph.llm_client import build_chat_client, chat_model


load_dotenv()


VALID_INTENTS = {"analytical", "semantic", "hybrid"}
VALID_FILTERS = {"production_line", "machine_id", "fault_category", "severity"}
FAULT_CATEGORY_ALIASES = {
    "motor failure": ["motor failure", "motor", "motor arız", "motor ariza"],
    "sensor error": ["sensor error", "sensör", "sensor", "sensör hatası", "sensor hatası"],
    "belt misalignment": ["belt misalignment", "belt", "kayış", "hiza", "misalignment"],
    "overheating": ["overheating", "ısın", "sicak", "sıcak", "heat", "overheat"],
    "electrical fault": ["electrical fault", "electrical", "elektrik", "voltaj", "power", "wiring"],
}


@dataclass(frozen=True)
class RouteDecision:
    """Structured routing output consumed by the LangGraph classify node."""

    intent: str
    confidence: float
    reasoning: str
    filters: dict[str, Any] = field(default_factory=dict)


def _normalize_question(question: str) -> str:
    return question.lower().strip()


def _detect_fault_category(question: str) -> str | None:
    lowered = _normalize_question(question)
    for category, aliases in FAULT_CATEGORY_ALIASES.items():
        if any(alias in lowered for alias in aliases):
            return category
    return None


def _detect_line(question: str) -> str | None:
    match = re.search(r"\b(?:line|hat)\s*([123])\b", _normalize_question(question))
    if match:
        return f"Line {match.group(1)}"
    return None


def _detect_machine_id(question: str) -> str | None:
    match = re.search(r"\b(?:mc|machine|makine)[-\s]?(\d{2,4})\b", question, flags=re.IGNORECASE)
    if not match:
        return None
    return f"MC-{match.group(1)}"


def _detect_severity(question: str) -> str | None:
    lowered = _normalize_question(question)
    aliases = {
        "high": ["high", "yüksek", "yuksek", "kritik", "critical"],
        "medium": ["medium", "orta"],
        "low": ["low", "düşük", "dusuk"],
    }
    for severity, terms in aliases.items():
        if any(term in lowered for term in terms):
            return severity
    return None


def extract_filters_fallback(question: str) -> dict[str, Any]:
    """Extract safe metadata filters locally when the LLM is unavailable."""

    filters: dict[str, Any] = {}
    fault_category = _detect_fault_category(question)
    production_line = _detect_line(question)
    machine_id = _detect_machine_id(question)
    severity = _detect_severity(question)
    if fault_category:
        filters["fault_category"] = fault_category
    if production_line:
        filters["production_line"] = production_line
    if machine_id:
        filters["machine_id"] = machine_id
    if severity:
        filters["severity"] = severity
    return filters


def _heuristic_classify(question: str) -> RouteDecision:
    lowered = question.lower().strip()
    analytical_keywords = ["kaç", "how many", "trend", "ortalama", "rate", "en çok", "distribution", "count", "which"]
    semantic_keywords = ["neden", "why", "ne oldu", "what happened", "arıza", "fault", "error", "sensör", "sensor", "benzer", "similar"]
    strong_semantic_keywords = ["neden", "why", "ne oldu", "what happened", "benzer", "similar", "açıkla", "explain"]
    has_analytical = any(keyword in lowered for keyword in analytical_keywords)
    has_semantic = any(keyword in lowered for keyword in semantic_keywords)
    has_strong_semantic = any(keyword in lowered for keyword in strong_semantic_keywords)
    filters = extract_filters_fallback(question)

    if has_analytical and has_strong_semantic:
        return RouteDecision("hybrid", 0.65, "Fallback matched analytical and semantic cues.", filters)
    if has_analytical:
        return RouteDecision("analytical", 0.7, "Fallback matched aggregation or trend cues.", filters)
    if has_semantic:
        return RouteDecision("semantic", 0.7, "Fallback matched fault or similarity cues.", filters)
    return RouteDecision("hybrid", 0.45, "Fallback defaulted to hybrid for an ambiguous question.", filters)


@lru_cache(maxsize=1)
def _build_client() -> AzureOpenAI | OpenAI | None:
    return build_chat_client()


def _coerce_confidence(value: object) -> float:
    try:
        confidence = float(value)
    except (TypeError, ValueError):
        return 0.5
    return max(0.0, min(1.0, confidence))


def _sanitize_filters(raw_filters: object) -> dict[str, Any]:
    if not isinstance(raw_filters, dict):
        return {}

    filters: dict[str, Any] = {}
    for key, value in raw_filters.items():
        if key not in VALID_FILTERS or value is None or value == "":
            continue
        if key == "production_line":
            match = re.search(r"\b([123])\b", str(value))
            if match:
                filters[key] = f"Line {match.group(1)}"
        elif key == "machine_id":
            match = re.search(r"\b(?:MC-?)?(\d{2,4})\b", str(value), flags=re.IGNORECASE)
            if match:
                filters[key] = f"MC-{match.group(1)}"
        elif key == "severity":
            severity = str(value).strip().lower()
            if severity in {"low", "medium", "high"}:
                filters[key] = severity
        elif key == "fault_category":
            category = str(value).strip().lower()
            if category in FAULT_CATEGORY_ALIASES:
                filters[key] = category
    return filters


def _route_from_payload(payload: dict[str, object], question: str) -> RouteDecision:
    intent = str(payload.get("intent", "")).strip().lower()
    if intent not in VALID_INTENTS:
        return _heuristic_classify(question)

    llm_filters = _sanitize_filters(payload.get("filters", {}))
    fallback_filters = extract_filters_fallback(question)
    filters = {**fallback_filters, **llm_filters}
    reasoning = str(payload.get("reasoning", "")).strip() or "Structured LLM router decision."
    confidence = _coerce_confidence(payload.get("confidence", 0.75))
    return RouteDecision(intent=intent, confidence=confidence, reasoning=reasoning, filters=filters)


def classify_question(question: str, conversation_history: list[dict[str, object]] | None = None) -> RouteDecision:
    """Classify the question and extract metadata filters.

    The model is used when available; otherwise we fall back to a deterministic
    local router with the same output shape.
    """

    client = _build_client()
    if client is None:
        return _heuristic_classify(question)

    model = chat_model("AZURE_OPENAI_CLASSIFIER_DEPLOYMENT_NAME", "OPENAI_CLASSIFIER_MODEL")

    try:
        response = client.chat.completions.create(
            model=model,
            temperature=0,
            response_format={"type": "json_object"},
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You route industrial maintenance questions and extract safe metadata filters. "
                        "Return JSON only. Do not invent filters that are not stated or strongly implied."
                    ),
                },
                {
                    "role": "user",
                    "content": json.dumps(
                        {
                            "question": question,
                            "conversation_history": conversation_history or [],
                            "valid_intents": sorted(VALID_INTENTS),
                            "intent_rules": {
                                "analytical": "Counts, averages, trends, top-N, distributions, SQL-style aggregations.",
                                "semantic": "Specific incidents, explanations, similar cases, troubleshooting context.",
                                "hybrid": "Needs both analytical aggregation and semantic incident evidence.",
                            },
                            "valid_filters": {
                                "production_line": ["Line 1", "Line 2", "Line 3"],
                                "fault_category": sorted(FAULT_CATEGORY_ALIASES),
                                "severity": ["low", "medium", "high"],
                                "machine_id": "Canonical form like MC-101.",
                            },
                            "output_schema": {
                                "intent": "analytical | semantic | hybrid",
                                "confidence": "number between 0 and 1",
                                "reasoning": "short reason",
                                "filters": {
                                    "production_line": "optional",
                                    "machine_id": "optional",
                                    "fault_category": "optional",
                                    "severity": "optional",
                                },
                            },
                        },
                        ensure_ascii=False,
                    ),
                },
            ],
        )
    except Exception:
        return _heuristic_classify(question)

    raw_text = response.choices[0].message.content or "{}"
    try:
        payload = json.loads(raw_text)
    except json.JSONDecodeError:
        return _heuristic_classify(question)

    if not isinstance(payload, dict):
        return _heuristic_classify(question)
    return _route_from_payload(payload, question)


def classify_intent(question: str) -> str:
    """Backward-compatible helper for callers that only need the route name."""

    return classify_question(question).intent
