"""Tests for shared LLM client environment handling."""

from __future__ import annotations

import sys
from pathlib import Path

from openai import OpenAI

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from graph.llm_client import build_chat_client, chat_model


def test_openai_compatible_azure_v1_endpoint(monkeypatch) -> None:
    monkeypatch.setenv("AZURE_OPENAI_API_KEY", "test-key")
    monkeypatch.setenv("AZURE_OPENAI_ENDPOINT", "https://topuz-openai.openai.azure.com/openai/v1")
    monkeypatch.setenv("AZURE_OPENAI_DEPLOYMENT_NAME", "gpt-5.4-mini")
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)

    client = build_chat_client()

    assert isinstance(client, OpenAI)
    assert str(client.base_url) == "https://topuz-openai.openai.azure.com/openai/v1/"
    assert chat_model() == "gpt-5.4-mini"


def test_specific_model_override_wins(monkeypatch) -> None:
    monkeypatch.setenv("AZURE_OPENAI_DEPLOYMENT_NAME", "gpt-5.4-mini")
    monkeypatch.setenv("AZURE_OPENAI_ANSWER_DEPLOYMENT_NAME", "answer-model")

    assert chat_model("AZURE_OPENAI_ANSWER_DEPLOYMENT_NAME") == "answer-model"
