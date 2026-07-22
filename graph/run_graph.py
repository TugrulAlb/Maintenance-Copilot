"""Run the Maintenance Copilot graph from the command line.

This is a debugging helper for visualizing the self-correction loop. It is not
part of the API server path.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from uuid import uuid4

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from graph.build_graph import get_graph_app


def format_node_trace(trace: list[object]) -> str:
    """Render node trace with retry iterations visible in one line."""

    return " -> ".join(str(item) for item in trace)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run one question through the Maintenance Copilot LangGraph app.")
    parser.add_argument("question", help="Natural-language maintenance question to ask.")
    parser.add_argument("--thread-id", default=None, help="Optional conversation thread id.")
    args = parser.parse_args()

    output = get_graph_app().invoke(
        {
            "question": args.question,
            "thread_id": args.thread_id or str(uuid4()),
            "conversation_history": [],
        }
    )

    print("Node trace:")
    print(format_node_trace(output.get("node_trace", [])))
    print("\nEvaluation:")
    print(
        json.dumps(
            {
                "is_sufficient": output.get("is_sufficient"),
                "retry_count": output.get("retry_count", 0),
                "hit_retry_cap": output.get("hit_retry_cap", False),
                "missing_aspects": output.get("missing_aspects", []),
                "evaluation_reasoning": output.get("evaluation_reasoning"),
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    print("\nAnswer:")
    print(output.get("answer", ""))


if __name__ == "__main__":
    main()
