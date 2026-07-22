"""Manual retrieval sanity-checks for Maintenance Copilot.

These tests are intentionally integration-style. They print ranked results so a
developer can visually inspect whether hybrid retrieval and reranking surface the
expected maintenance tickets.
"""

from __future__ import annotations

import os
import sqlite3
import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ingestion.chunker import chunk_record
from retrieval.bm25_index import BM25Index
from retrieval.hybrid_search import HybridRetriever
from retrieval.reranker import Reranker


class _FakeVectorStore:
    def __init__(self) -> None:
        self.filters = None
        self.collection = self

    def query(self, query_embedding, top_k, filters=None):
        self.filters = filters
        return [
            {
                "id": "1",
                "document": "Line 2 motor failure",
                "metadata": {"production_line": "Line 2", "fault_category": "motor failure"},
                "score": 0.9,
            }
        ]

    def get(self, ids, include):
        return {
            "ids": ["1", "2"],
            "documents": ["Line 2 motor failure", "Line 1 sensor error"],
            "metadatas": [
                {"production_line": "Line 2", "fault_category": "motor failure"},
                {"production_line": "Line 1", "fault_category": "sensor error"},
            ],
        }


class _FakeBM25Index:
    def search(self, query, top_k):
        return [("1", 3.0), ("2", 2.0)]


class _FakeEmbedder:
    def embed(self, texts):
        return [[0.1, 0.2, 0.3] for _ in texts]


def test_hybrid_search_applies_metadata_filters_to_vector_and_bm25() -> None:
    vector_store = _FakeVectorStore()
    retriever = HybridRetriever(
        vector_store=vector_store,
        bm25_index=_FakeBM25Index(),
        embedder=_FakeEmbedder(),
    )

    results = retriever.search(
        "Line 2 motor failure",
        top_k_each=5,
        filters={"production_line": "Line 2", "fault_category": "motor failure"},
    )

    assert vector_store.filters == {"production_line": "Line 2", "fault_category": "motor failure"}
    assert [item["id"] for item in results] == ["1"]


def _project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _require_prerequisites() -> None:
    project_root = _project_root()
    if not (project_root / "data" / "maintenance_logs.db").exists():
        pytest.skip("Synthetic SQLite database is missing; run data generation first.")
    if not (project_root / "chroma_data").exists():
        pytest.skip("ChromaDB store is missing; run ingestion first.")


def _build_bm25_if_needed() -> None:
    project_root = _project_root()
    index_path = project_root / "retrieval" / "bm25_index.pkl"
    if index_path.exists():
        return

    connection = sqlite3.connect(project_root / "data" / "maintenance_logs.db")
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
    finally:
        connection.close()

    bm25 = BM25Index(persist_path=index_path)
    bm25.build_index((str(row["id"]), chunk_record(row)) for row in rows)


@pytest.mark.parametrize(
    "query",
    [
        "motor arızası nedeniyle duran hatlar",
        "sensör hatası olan makineler",
    ],
)
def test_hybrid_retrieval_and_rerank(query: str) -> None:
    """Run a query through hybrid retrieval and reranking and print the top results."""

    _require_prerequisites()
    _build_bm25_if_needed()

    if not (os.getenv("AZURE_OPENAI_API_KEY") or os.getenv("OPENAI_API_KEY")):
        pytest.skip("Embedding calls require Azure OpenAI or OpenAI credentials.")

    retriever = HybridRetriever()
    reranker = Reranker()

    candidates = retriever.search(query=query, top_k_each=15)
    results = reranker.rerank(query=query, candidates=candidates, top_k=5)

    print(f"\nQuery: {query}")
    for index, result in enumerate(results, start=1):
        metadata = result.get("metadata", {})
        print(
            f"{index}. id={result.get('id')} rerank_score={result.get('rerank_score'):.4f} "
            f"rrf_score={result.get('rrf_score'):.4f} line={metadata.get('production_line')} "
            f"fault={metadata.get('fault_category')}"
        )

    assert len(results) > 0
