"""Persistent ChromaDB access for maintenance log embeddings.

Design note:
- We store metadata next to each embedding so later hybrid queries can filter by
  dimensions like line, fault category, or timestamp without re-reading SQLite.
- That matters for operational questions such as "show me Line 3 overheating
  incidents" where the text match alone is not enough.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import chromadb


class VectorStore:
    """Thin persistent wrapper around a Chroma collection."""

    def __init__(self, persist_dir: str | Path | None = None, collection_name: str = "maintenance_logs") -> None:
        project_root = Path(__file__).resolve().parents[1]
        self.persist_dir = Path(persist_dir) if persist_dir is not None else project_root / "chroma_data"
        self.persist_dir.mkdir(parents=True, exist_ok=True)
        self.client = chromadb.PersistentClient(path=str(self.persist_dir))
        self.collection = self.client.get_or_create_collection(name=collection_name)

    def upsert(self, id: str, text: str, embedding: list[float], metadata: dict[str, Any]) -> None:
        """Insert or update a single record by id so ingestion stays idempotent."""

        self.collection.upsert(
            ids=[str(id)],
            documents=[text],
            embeddings=[embedding],
            metadatas=[metadata],
        )

    def query(self, query_embedding: list[float], top_k: int, filters: dict[str, Any] | None = None) -> list[dict[str, Any]]:
        """Run a nearest-neighbor search with optional metadata filtering."""

        query_kwargs: dict[str, Any] = {
            "query_embeddings": [query_embedding],
            "n_results": top_k,
        }
        if filters:
            query_kwargs["where"] = filters
        result = self.collection.query(**query_kwargs)

        ids = result.get("ids", [[]])[0]
        documents = result.get("documents", [[]])[0]
        metadatas = result.get("metadatas", [[]])[0]
        distances = result.get("distances", [[]])[0]

        items: list[dict[str, Any]] = []
        for doc_id, document, metadata, distance in zip(ids, documents, metadatas, distances):
            items.append(
                {
                    "id": doc_id,
                    "document": document,
                    "metadata": metadata,
                    # Convert distance into a higher-is-better similarity score for easier ranking fusion.
                    "score": 1.0 / (1.0 + float(distance)),
                }
            )
        return items
