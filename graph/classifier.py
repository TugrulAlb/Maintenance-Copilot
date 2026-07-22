"""Intent classification for Maintenance Copilot.

The preferred path is an LLM-based classifier so the system does not rely on
hard-coded keyword lists for intent routing. If no model credentials are
available, we fall back to a small heuristic classifier so local development
and tests still work offline.
"""

from __future__ import annotations

import json
from functools import lru_cache

from dotenv import load_dotenv
from openai import AzureOpenAI, OpenAI

from graph.llm_client import build_chat_client, chat_model


load_dotenv()


def _heuristic_classify(question: str) -> str:
    lowered = question.lower().strip()
    analytical_keywords = ["kaç", "how many", "trend", "ortalama", "rate", "en çok", "distribution", "count", "which"]
    semantic_keywords = ["neden", "why", "ne oldu", "what happened", "arıza", "fault", "error", "sensör", "sensor", "benzer", "similar"]
    has_analytical = any(keyword in lowered for keyword in analytical_keywords)
    has_semantic = any(keyword in lowered for keyword in semantic_keywords)

    if has_analytical and has_semantic:
        return "hybrid"
    if has_analytical:
        return "analytical"
    if has_semantic:
        return "semantic"
    return "hybrid"


@lru_cache(maxsize=1)
def _build_client() -> AzureOpenAI | OpenAI | None:
    return build_chat_client()


def classify_intent(question: str) -> str:
    """Classify the question as analytical, semantic, or hybrid.

    The model is used when available; otherwise we fall back to a deterministic
    local heuristic.
    """

    client = _build_client()
    if client is None:
        return _heuristic_classify(question)

    model = chat_model("AZURE_OPENAI_CLASSIFIER_DEPLOYMENT_NAME", "OPENAI_CLASSIFIER_MODEL")

    response = client.chat.completions.create(
        model=model,
        temperature=0,
        response_format={"type": "json_object"},
        messages=[
            {
                "role": "system",
                "content": (
                    "You classify industrial maintenance questions into exactly one of: "
                    "analytical, semantic, hybrid. Return JSON with a single key 'intent'."
                ),
            },
            {
                "role": "user",
                "content": (
                    "Question: "
                    f"{question}\n\n"
                    "Rules:\n"
                    "- analytical: asks for counts, averages, trends, top-N, filters, or aggregations\n"
                    "- semantic: asks about a specific issue, fault, explanation, or similar cases\n"
                    "- hybrid: could reasonably need both retrieval and analysis\n"
                    "Return only JSON like {\"intent\": \"analytical\"}."
                ),
            },
        ],
    )

    raw_text = response.choices[0].message.content or "{}"
    try:
        payload = json.loads(raw_text)
        intent = str(payload.get("intent", "")).strip().lower()
    except json.JSONDecodeError:
        return _heuristic_classify(question)

    if intent in {"analytical", "semantic", "hybrid"}:
        return intent
    return _heuristic_classify(question)
