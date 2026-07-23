"""Curated evaluation questions for Maintenance Copilot.

The expected values intentionally avoid exact answer strings. For analytical
cases we describe the SQL result shape. For semantic/hybrid cases, known
relevant record ids can be filled after generating the synthetic database; until
then the runner can derive reference contexts from these filters and keywords.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal


Intent = Literal["analytical", "semantic", "hybrid"]


@dataclass(frozen=True)
class EvalQuestion:
    id: str
    question: str
    expected_intent: Intent
    expected_answer_characteristics: list[str]
    expected_sql_shape: str | None = None
    expected_relevant_ids: list[int] = field(default_factory=list)
    reference_filters: dict[str, str] = field(default_factory=dict)
    reference_keywords: list[str] = field(default_factory=list)


TEST_QUESTIONS: list[EvalQuestion] = [
    EvalQuestion(
        id="analytical_line2_top_fault_last_month",
        question="Which fault type occurred most often on Line 2 last month?",
        expected_intent="analytical",
        expected_sql_shape="Grouped counts by fault_category filtered to production_line='Line 2' and a recent timestamp window.",
        expected_answer_characteristics=["names the top fault category", "mentions Line 2", "uses count or ranking"],
        reference_filters={"production_line": "Line 2"},
    ),
    EvalQuestion(
        id="analytical_avg_resolution_motor",
        question="What is the average resolution time for motor failure incidents?",
        expected_intent="analytical",
        expected_sql_shape="AVG(resolution_time_minutes) filtered by fault_category='motor failure'.",
        expected_answer_characteristics=["average resolution time", "motor failure"],
        reference_filters={"fault_category": "motor failure"},
    ),
    EvalQuestion(
        id="analytical_high_severity_by_line",
        question="How many high severity issues happened on each production line?",
        expected_intent="analytical",
        expected_sql_shape="COUNT(*) grouped by production_line filtered by severity='high'.",
        expected_answer_characteristics=["per-line counts", "high severity"],
        reference_filters={"severity": "high"},
    ),
    EvalQuestion(
        id="analytical_top_machines_sensor",
        question="Which machines have the most sensor error tickets?",
        expected_intent="analytical",
        expected_sql_shape="COUNT(*) grouped by machine_id filtered by fault_category='sensor error', ordered descending.",
        expected_answer_characteristics=["machine ids", "sensor error", "ranking or counts"],
        reference_filters={"fault_category": "sensor error"},
    ),
    EvalQuestion(
        id="analytical_overheating_distribution",
        question="Show the distribution of overheating incidents across Line 1, Line 2, and Line 3.",
        expected_intent="analytical",
        expected_sql_shape="COUNT(*) grouped by production_line filtered by fault_category='overheating'.",
        expected_answer_characteristics=["overheating", "Line 1", "Line 2", "Line 3"],
        reference_filters={"fault_category": "overheating"},
    ),
    EvalQuestion(
        id="analytical_longest_repairs_electrical",
        question="Which electrical fault records took the longest to resolve?",
        expected_intent="analytical",
        expected_sql_shape="Rows filtered by fault_category='electrical fault', ordered by resolution_time_minutes descending.",
        expected_answer_characteristics=["electrical fault", "resolution time", "record or machine"],
        reference_filters={"fault_category": "electrical fault"},
    ),
    EvalQuestion(
        id="semantic_motor_noise",
        question="Find incidents describing unusual motor noise or rattling.",
        expected_intent="semantic",
        expected_relevant_ids=[],
        expected_answer_characteristics=["motor-related incident", "noise or rattling context"],
        reference_filters={"fault_category": "motor failure"},
        reference_keywords=["noise", "rattle", "rattling", "motor"],
    ),
    EvalQuestion(
        id="semantic_sensor_flaky",
        question="Find records where a sensor looked flaky or gave inconsistent readings.",
        expected_intent="semantic",
        expected_relevant_ids=[],
        expected_answer_characteristics=["sensor error", "inconsistent or flaky readings"],
        reference_filters={"fault_category": "sensor error"},
        reference_keywords=["sensor", "reading", "flaky", "intermittent", "inconsistent"],
    ),
    EvalQuestion(
        id="semantic_belt_alignment",
        question="Which incidents sound like belt alignment or tracking problems?",
        expected_intent="semantic",
        expected_relevant_ids=[],
        expected_answer_characteristics=["belt misalignment", "alignment or tracking"],
        reference_filters={"fault_category": "belt misalignment"},
        reference_keywords=["belt", "alignment", "tracking", "misalignment"],
    ),
    EvalQuestion(
        id="semantic_burning_smell",
        question="Find tickets mentioning a burning smell, wiring concern, or power issue.",
        expected_intent="semantic",
        expected_relevant_ids=[],
        expected_answer_characteristics=["electrical fault", "burning smell or power issue"],
        reference_filters={"fault_category": "electrical fault"},
        reference_keywords=["burn", "burning", "smell", "power", "wiring", "electrical"],
    ),
    EvalQuestion(
        id="semantic_hot_shutdown",
        question="Find incidents where the machine got hot and shut down.",
        expected_intent="semantic",
        expected_relevant_ids=[],
        expected_answer_characteristics=["overheating", "shutdown or high temperature"],
        reference_filters={"fault_category": "overheating"},
        reference_keywords=["hot", "heat", "overheat", "temperature", "shutdown"],
    ),
    EvalQuestion(
        id="semantic_line3_critical",
        question="Show me critical-looking Line 3 maintenance notes.",
        expected_intent="semantic",
        expected_relevant_ids=[],
        expected_answer_characteristics=["Line 3", "high severity or critical issue"],
        reference_filters={"production_line": "Line 3", "severity": "high"},
        reference_keywords=["critical", "urgent", "stopped", "shutdown", "high"],
    ),
    EvalQuestion(
        id="hybrid_line2_motor_trend_similar",
        question="Summarize the Line 2 motor failure trend and include similar incident examples.",
        expected_intent="hybrid",
        expected_sql_shape="Aggregation or count/trend for production_line='Line 2' and fault_category='motor failure'.",
        expected_relevant_ids=[],
        expected_answer_characteristics=["Line 2", "motor failure", "trend or count", "similar incidents"],
        reference_filters={"production_line": "Line 2", "fault_category": "motor failure"},
        reference_keywords=["motor", "failure", "overheat", "noise", "stopped"],
    ),
    EvalQuestion(
        id="hybrid_overheating_by_line_examples",
        question="Which line has the most overheating issues, and what do the related notes say?",
        expected_intent="hybrid",
        expected_sql_shape="Grouped count by production_line filtered by fault_category='overheating'.",
        expected_relevant_ids=[],
        expected_answer_characteristics=["overheating", "line with most issues", "example notes"],
        reference_filters={"fault_category": "overheating"},
        reference_keywords=["overheat", "hot", "temperature", "shutdown"],
    ),
    EvalQuestion(
        id="hybrid_sensor_avg_and_examples",
        question="What is the average repair time for sensor errors and show representative sensor tickets?",
        expected_intent="hybrid",
        expected_sql_shape="AVG(resolution_time_minutes) filtered by fault_category='sensor error'.",
        expected_relevant_ids=[],
        expected_answer_characteristics=["average repair time", "sensor error", "representative tickets"],
        reference_filters={"fault_category": "sensor error"},
        reference_keywords=["sensor", "reading", "signal", "calibration"],
    ),
    EvalQuestion(
        id="hybrid_electrical_long_repairs_notes",
        question="Find electrical faults with long resolution times and summarize the note patterns.",
        expected_intent="hybrid",
        expected_sql_shape="Rows or aggregate filtered by fault_category='electrical fault', ordered by resolution_time_minutes.",
        expected_relevant_ids=[],
        expected_answer_characteristics=["electrical fault", "long resolution", "note patterns"],
        reference_filters={"fault_category": "electrical fault"},
        reference_keywords=["electrical", "power", "wiring", "voltage", "burning"],
    ),
]
