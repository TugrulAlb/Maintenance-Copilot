"""Simple evaluation runner for Maintenance Copilot."""

from __future__ import annotations

import json
from pathlib import Path

from graph.build_graph import get_graph_app


CASES_PATH = Path(__file__).resolve().parent / "cases.json"


def run_eval() -> dict[str, object]:
    """Run a light-weight evaluation over a small question set."""

    cases = json.loads(CASES_PATH.read_text(encoding="utf-8"))
    graph_app = get_graph_app()
    results = []
    passed = 0

    for index, case in enumerate(cases, start=1):
        output = graph_app.invoke(
            {
                "question": case["question"],
                "thread_id": f"eval-{index}",
            }
        )
        actual = output.get("query_type")
        answer = str(output.get("answer", ""))
        citations = output.get("citations", [])
        node_trace = output.get("node_trace", [])
        expected_terms = case.get("expected_answer_terms", [])
        checks = {
            "query_type": actual == case["expected_query_type"],
            "citations": bool(citations) if case.get("require_citations", True) else True,
            "node_trace": all(node in node_trace for node in case.get("required_nodes", [])),
            "answer_terms": all(term.lower() in answer.lower() for term in expected_terms),
        }
        success = all(checks.values())
        passed += int(success)
        results.append({
            "question": case["question"],
            "expected": case["expected_query_type"],
            "actual": actual,
            "success": success,
            "checks": checks,
            "citations": citations,
            "node_trace": node_trace,
        })

    summary = {
        "total": len(cases),
        "passed": passed,
        "failed": len(cases) - passed,
        "accuracy": round(passed / max(1, len(cases)), 3),
        "results": results,
    }
    return summary


if __name__ == "__main__":
    report = run_eval()
    print(json.dumps(report, ensure_ascii=False, indent=2))
