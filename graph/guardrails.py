"""Lightweight guardrails for Maintenance Copilot.

The project keeps these rails dependency-free on purpose, but the shape mirrors
larger guardrail frameworks: input rails decide whether work should start at
all, while output rails sanitize and normalize what leaves the system.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field


@dataclass(frozen=True)
class InputGuardrailResult:
    blocked: bool
    reason: str | None = None
    safe_response: str | None = None


@dataclass(frozen=True)
class OutputGuardrailResult:
    answer: str
    redactions: list[str] = field(default_factory=list)


MAINTENANCE_TERMS = {
    "bakım",
    "arıza",
    "makine",
    "motor",
    "hat",
    "sensör",
    "sensor",
    "kayış",
    "line",
    "machine",
    "equipment",
    "maintenance",
    "fault",
    "failure",
    "incident",
    "overheat",
    "overheating",
    "noise",
    "vibration",
    "downtime",
    "severity",
    "production",
    "resolved",
    "technician",
    "sql",
    "trend",
    "distribution",
}

PROMPT_INJECTION_PATTERNS = [
    re.compile(pattern, re.IGNORECASE)
    for pattern in [
        r"\bignore (all )?(previous|prior|above) instructions\b",
        r"\bdisregard (all )?(previous|prior|above) instructions\b",
        r"\byou are now\b",
        r"\bact as\b.*\b(system|developer|admin)\b",
        r"\breveal\b.*\b(system prompt|developer message|hidden prompt)\b",
        r"\bshow\b.*\b(system prompt|developer message|hidden prompt)\b",
        r"\bbypass\b.*\bguardrail|safety|policy\b",
        r"\bjailbreak\b",
    ]
]

OFF_TOPIC_PATTERNS = [
    re.compile(pattern, re.IGNORECASE)
    for pattern in [
        r"\bwrite (me )?(a )?(poem|song|story)\b",
        r"\b(recipe|weather|horoscope|football|movie|dating advice)\b",
        r"\bwhat is the capital of\b",
        r"\bcrypto|stock price|lottery\b",
        r"\bşiir|tarif|hava durumu|film öner\b",
    ]
]

PII_PATTERNS: list[tuple[str, re.Pattern[str], str]] = [
    ("email", re.compile(r"\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b", re.IGNORECASE), "[REDACTED_EMAIL]"),
    ("ssn", re.compile(r"\b\d{3}-\d{2}-\d{4}\b"), "[REDACTED_ID]"),
    ("national_id", re.compile(r"\b\d{11}\b"), "[REDACTED_ID]"),
    (
        "phone",
        re.compile(r"(?<!\w)(?:\+?\d{1,3}[\s.-]?)?(?:\(?\d{3}\)?[\s.-]?)\d{3}[\s.-]?\d{2,4}[\s.-]?\d{0,4}(?!\w)"),
        "[REDACTED_PHONE]",
    ),
]


def _safe_refusal(reason: str) -> str:
    if reason == "prompt_injection":
        return (
            "Bu isteği güvenli şekilde işleyemem. Maintenance Copilot yalnızca "
            "endüstriyel bakım kayıtlarıyla ilgili soruları yanıtlar."
        )
    return (
        "Bu soru bakım ve ekipman arızaları alanının dışında görünüyor. "
        "Maintenance Copilot'a üretim hattı, makine, arıza, bakım kaydı veya trendlerle ilgili bir soru sorabilirsin."
    )


def evaluate_input_guardrails(question: str) -> InputGuardrailResult:
    """Return whether a question should be blocked before expensive graph work.

    Input guardrails run as the first graph node for cost and safety reasons:
    rejecting an off-topic or injection-like request here avoids paying for the
    router, retrieval, SQL generation, answer generation, and evaluator when the
    question was never going to produce a useful maintenance answer.
    """

    normalized = question.strip().lower()
    if any(pattern.search(normalized) for pattern in PROMPT_INJECTION_PATTERNS):
        return InputGuardrailResult(True, "prompt_injection", _safe_refusal("prompt_injection"))

    if any(pattern.search(normalized) for pattern in OFF_TOPIC_PATTERNS):
        return InputGuardrailResult(True, "off_topic", _safe_refusal("off_topic"))

    has_maintenance_signal = any(term in normalized for term in MAINTENANCE_TERMS)
    if not has_maintenance_signal and len(normalized.split()) >= 6:
        return InputGuardrailResult(True, "off_topic", _safe_refusal("off_topic"))

    return InputGuardrailResult(False)


def redact_pii(answer: str) -> OutputGuardrailResult:
    """Redact obvious personal identifiers if generation accidentally leaks one."""

    redactions: list[str] = []
    redacted = answer
    for label, pattern, replacement in PII_PATTERNS:
        redacted, count = pattern.subn(replacement, redacted)
        if count:
            redactions.extend([label] * count)
    return OutputGuardrailResult(answer=redacted, redactions=redactions)


def apply_output_guardrails(
    answer: str,
    *,
    is_sufficient: bool | None,
    hit_retry_cap: bool,
    evaluation_reasoning: str | None,
    input_blocked: bool,
) -> OutputGuardrailResult:
    """Apply final response guardrails before the API schema is built."""

    trimmed = answer.strip()
    if not trimmed:
        trimmed = "Elimde bu soruyu güvenle yanıtlamak için yeterli bilgi yok."

    if hit_retry_cap and is_sufficient is not True and not input_blocked:
        reason = evaluation_reasoning or "kanıtlar otomatik kontrol adımında yeterli bulunmadı"
        trimmed = (
            "Elimde bu soruyu tamamen güvenle yanıtlamak için yeterli bilgi yok; "
            f"cevabın kısmen eksik olabilir. Değerlendirme: {reason}\n\n{trimmed}"
        )

    return redact_pii(trimmed)
