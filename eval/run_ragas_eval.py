"""Run RAGAS and custom evaluations for the full Maintenance Copilot graph."""

from __future__ import annotations

import json
import os
import sqlite3
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from graph.build_graph import get_graph_app
from graph.llm_client import chat_model
from ingestion.chunker import chunk_record
from retrieval.bm25_index import BM25Index
from eval.custom_metrics import hallucination_flag_rate, judge_hallucination
from eval.test_questions import TEST_QUESTIONS, EvalQuestion


RESULTS_PATH = Path(__file__).resolve().parent / "results.csv"
DB_PATH = PROJECT_ROOT / "data" / "maintenance_logs.db"
CHROMA_PATH = PROJECT_ROOT / "chroma_data"
BM25_PATH = PROJECT_ROOT / "retrieval" / "bm25_index.pkl"


def _load_rows() -> list[dict[str, Any]]:
    if not DB_PATH.exists() or DB_PATH.stat().st_size == 0:
        return []
    connection = sqlite3.connect(DB_PATH)
    connection.row_factory = sqlite3.Row
    try:
        rows = connection.execute(
            """
            SELECT id, timestamp, production_line, machine_id, fault_category,
                   description, severity, resolution_time_minutes, resolved_by
            FROM maintenance_logs
            ORDER BY id
            """
        ).fetchall()
        return [dict(row) for row in rows]
    except sqlite3.Error:
        return []
    finally:
        connection.close()


def _matches_reference(row: dict[str, Any], case: EvalQuestion) -> bool:
    for key, expected in case.reference_filters.items():
        if str(row.get(key, "")) != expected:
            return False
    if not case.reference_keywords:
        return True
    haystack = f"{row.get('fault_category', '')} {row.get('description', '')}".lower()
    return any(keyword.lower() in haystack for keyword in case.reference_keywords)


def reference_contexts_for(case: EvalQuestion, rows: list[dict[str, Any]], max_contexts: int = 5) -> list[str]:
    """Build ground-truth contexts for context_recall.

    Context recall is the one RAGAS metric here that needs a reference: it asks
    whether the retriever found the contexts that should have been found. Without
    expected record ids or a curated reference set, there is no denominator for
    recall. Faithfulness and answer relevancy are reference-free because they can
    judge answer-vs-context and answer-vs-question directly.
    """

    if not rows:
        return []

    if case.expected_relevant_ids:
        wanted = {int(record_id) for record_id in case.expected_relevant_ids}
        selected = [row for row in rows if int(row["id"]) in wanted]
    else:
        selected = [row for row in rows if _matches_reference(row, case)]

    return [chunk_record(row) for row in selected[:max_contexts]]


def ensure_bm25_index(rows: list[dict[str, Any]]) -> None:
    if BM25_PATH.exists():
        return
    bm25 = BM25Index(persist_path=BM25_PATH)
    bm25.build_index((str(row["id"]), chunk_record(row)) for row in rows)


def retrieved_contexts_from_output(output: dict[str, Any]) -> list[str]:
    contexts: list[str] = []
    for item in output.get("results", []) or []:
        document = str(item.get("document", "")).strip()
        if document:
            contexts.append(document)
    return contexts


def evidence_for_custom_judge(output: dict[str, Any]) -> list[dict[str, Any]]:
    evidence = output.get("evidence", [])
    return evidence if isinstance(evidence, list) else []


def sql_shape_ok(case: EvalQuestion, output: dict[str, Any]) -> bool | None:
    if case.expected_intent == "semantic":
        return None
    sql_query = str(output.get("sql_query", "")).lower()
    rows = output.get("sql_rows", [])
    if not sql_query or not isinstance(rows, list):
        return False

    shape = (case.expected_sql_shape or "").lower()
    if "grouped" in shape and "group by" not in sql_query:
        return False
    if "avg(" in shape and "avg(" not in sql_query:
        return False
    if "count" in shape and "count" not in sql_query and "rows or aggregate" not in shape:
        return False
    if "ordered" in shape and "order by" not in sql_query:
        return False
    return True


def _import_ragas():
    try:
        from datasets import Dataset
        from ragas import evaluate
        try:
            from ragas.metrics import answer_relevancy
        except ImportError:
            from ragas.metrics import response_relevancy as answer_relevancy
        from ragas.metrics import context_precision, context_recall, faithfulness
    except ImportError as exc:
        raise RuntimeError(
            "RAGAS evaluation dependencies are missing. Run `pip install -r requirements.txt` first."
        ) from exc
    return Dataset, evaluate, faithfulness, answer_relevancy, context_precision, context_recall


def _build_ragas_models():
    try:
        from langchain_openai import ChatOpenAI, OpenAIEmbeddings
        from ragas.embeddings import LangchainEmbeddingsWrapper
        from ragas.llms import LangchainLLMWrapper
    except ImportError:
        return None, None

    api_key = os.getenv("OPENAI_API_KEY") or os.getenv("AZURE_OPENAI_API_KEY")
    base_url = os.getenv("OPENAI_BASE_URL") or os.getenv("AZURE_OPENAI_ENDPOINT")
    if not api_key:
        return None, None

    llm_kwargs: dict[str, Any] = {
        "model": chat_model("AZURE_OPENAI_EVALUATOR_DEPLOYMENT_NAME", "OPENAI_EVALUATOR_MODEL"),
        "api_key": api_key,
        "temperature": 0,
    }
    embedding_kwargs: dict[str, Any] = {
        "model": os.getenv("AZURE_OPENAI_EMBEDDING_DEPLOYMENT_NAME")
        or os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small"),
        "api_key": api_key,
    }
    if base_url:
        llm_kwargs["base_url"] = base_url.rstrip("/")
        embedding_kwargs["base_url"] = base_url.rstrip("/")

    return LangchainLLMWrapper(ChatOpenAI(**llm_kwargs)), LangchainEmbeddingsWrapper(
        OpenAIEmbeddings(**embedding_kwargs)
    )


def _score_with_ragas(sample: dict[str, Any], metrics: list[Any]) -> dict[str, float | None]:
    Dataset, evaluate, *_ = _import_ragas()
    dataset = Dataset.from_list([sample])
    llm, embeddings = _build_ragas_models()
    result = evaluate(
        dataset,
        metrics=metrics,
        llm=llm,
        embeddings=embeddings,
        raise_exceptions=False,
        show_progress=False,
    )
    if hasattr(result, "to_pandas"):
        row = result.to_pandas().iloc[0].to_dict()
    else:
        row = dict(result)

    scores: dict[str, float | None] = {}
    for name in ["faithfulness", "answer_relevancy", "response_relevancy", "context_precision", "context_recall"]:
        if name in row:
            try:
                scores[name] = round(float(row[name]), 4)
            except (TypeError, ValueError):
                scores[name] = None
    if "response_relevancy" in scores and "answer_relevancy" not in scores:
        scores["answer_relevancy"] = scores["response_relevancy"]
    return scores


def run_eval() -> list[dict[str, Any]]:
    import pandas as pd

    _, _, faithfulness, answer_relevancy, context_precision, context_recall = _import_ragas()
    graph = get_graph_app()
    rows = _load_rows()
    if not rows:
        raise RuntimeError(
            "No maintenance rows found. Run `python data/generate_synthetic_data.py --records 500` first."
        )
    ensure_bm25_index(rows)
    if not CHROMA_PATH.exists():
        raise RuntimeError("Retrieval indexes are missing. Run `python ingestion/run_ingestion.py` first.")
    results: list[dict[str, Any]] = []

    for index, case in enumerate(TEST_QUESTIONS, start=1):
        output = graph.invoke(
            {
                "question": case.question,
                "thread_id": f"ragas-eval-{index}",
                "conversation_history": [],
            }
        )
        contexts = retrieved_contexts_from_output(output)
        reference_contexts = reference_contexts_for(case, rows)
        reference = "\n\n".join(reference_contexts)

        sample = {
            "user_input": case.question,
            "response": str(output.get("answer", "")),
            "retrieved_contexts": contexts or [json.dumps(output.get("sql_rows", []), ensure_ascii=False)],
            "reference": reference or "; ".join(case.expected_answer_characteristics),
            # Compatibility aliases for older examples/docs and quick inspection.
            "question": case.question,
            "answer": str(output.get("answer", "")),
            "contexts": contexts or [json.dumps(output.get("sql_rows", []), ensure_ascii=False)],
            "ground_truth": reference or "; ".join(case.expected_answer_characteristics),
        }

        metrics = [faithfulness, answer_relevancy]
        if case.expected_intent in {"semantic", "hybrid"} and reference_contexts:
            metrics.extend([context_precision, context_recall])

        ragas_scores = _score_with_ragas(sample, metrics)
        judge_result = judge_hallucination(case.question, sample["response"], evidence_for_custom_judge(output))

        results.append(
            {
                "id": case.id,
                "question": case.question,
                "expected_intent": case.expected_intent,
                "actual_intent": output.get("query_type"),
                "sql_shape_ok": sql_shape_ok(case, output),
                "retrieved_context_count": len(contexts),
                "reference_context_count": len(reference_contexts),
                "faithfulness": ragas_scores.get("faithfulness"),
                "answer_relevancy": ragas_scores.get("answer_relevancy"),
                "context_precision": ragas_scores.get("context_precision"),
                "context_recall": ragas_scores.get("context_recall"),
                "hallucination_flag": judge_result["hallucination_flag"],
                "hallucination_reasoning": judge_result["reasoning"],
                "node_trace": " -> ".join(str(item) for item in output.get("node_trace", [])),
            }
        )

    frame = pd.DataFrame(results)
    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(RESULTS_PATH, index=False)
    print(frame.to_string(index=False))
    metric_columns = ["faithfulness", "answer_relevancy", "context_precision", "context_recall"]
    print("\nOverall metric averages:")
    print(frame[metric_columns].mean(numeric_only=True).round(4).to_string())
    print(f"\nCustom hallucination_flag_rate: {hallucination_flag_rate(results):.4f}")
    print(f"Saved results to {RESULTS_PATH}")
    return results


if __name__ == "__main__":
    run_eval()
