"""Tests for the structured LangGraph router."""

from __future__ import annotations

import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import graph.classifier as classifier


class _FakeMessage:
    def __init__(self, content: str) -> None:
        self.content = content


class _FakeChoice:
    def __init__(self, content: str) -> None:
        self.message = _FakeMessage(content)


class _FakeResponse:
    def __init__(self, content: str) -> None:
        self.choices = [_FakeChoice(content)]


class _FakeCompletions:
    def __init__(self, payload: dict[str, object]) -> None:
        self.payload = payload

    def create(self, **kwargs):
        assert kwargs["response_format"] == {"type": "json_object"}
        return _FakeResponse(json.dumps(self.payload))


class _FakeChat:
    def __init__(self, payload: dict[str, object]) -> None:
        self.completions = _FakeCompletions(payload)


class _FakeClient:
    def __init__(self, payload: dict[str, object]) -> None:
        self.chat = _FakeChat(payload)


def test_structured_router_returns_intent_confidence_reasoning_and_filters(monkeypatch) -> None:
    classifier._build_client.cache_clear()
    monkeypatch.setattr(
        classifier,
        "_build_client",
        lambda: _FakeClient(
            {
                "intent": "hybrid",
                "confidence": 0.92,
                "reasoning": "The question asks for trend and similar incidents.",
                "filters": {
                    "production_line": "2",
                    "machine_id": "101",
                    "fault_category": "motor failure",
                    "severity": "high",
                    "unsupported": "ignored",
                },
            }
        ),
    )

    decision = classifier.classify_question("Line 2 MC-101 yüksek motor failure trendini ve benzer kayıtları özetle")

    assert decision.intent == "hybrid"
    assert decision.confidence == 0.92
    assert decision.reasoning == "The question asks for trend and similar incidents."
    assert decision.filters == {
        "production_line": "Line 2",
        "machine_id": "MC-101",
        "fault_category": "motor failure",
        "severity": "high",
    }


def test_classify_intent_remains_backward_compatible(monkeypatch) -> None:
    classifier._build_client.cache_clear()
    monkeypatch.setattr(classifier, "_build_client", lambda: None)

    assert classifier.classify_intent("Line 3'te en çok hangi arızalar var?") == "analytical"
