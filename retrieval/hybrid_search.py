"""Hybrid retrieval over vector and lexical indexes.

Design note:
- Reciprocal Rank Fusion (RRF) is a strong default because it combines rank
  lists without forcing cosine similarity and BM25 scores onto the same scale.
- That avoids a common bug where mixed retrieval signals are normalized poorly
  and one modality accidentally dominates the other.
"""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Any

from ingestion.embedder import EmbeddingClient
from ingestion.vector_store import VectorStore
from .bm25_index import BM25Index


class HybridRetriever:
    """Combine vector search and BM25 search using Reciprocal Rank Fusion."""

    def __init__(
        self,
        vector_store: VectorStore | None = None,
        bm25_index: BM25Index | None = None,
        embedder: EmbeddingClient | None = None,
        rrf_k: int = 60,
    ) -> None:
        self.vector_store = vector_store or VectorStore()
        self.bm25_index = bm25_index or BM25Index()
        self.embedder = embedder or EmbeddingClient()
        self.rrf_k = rrf_k

    def _rrf_merge(self, ranked_lists: list[list[dict[str, Any]]]) -> list[dict[str, Any]]:
        """Fuse multiple ranking lists with Reciprocal Rank Fusion."""

        fused: dict[str, dict[str, Any]] = {}
        for source_ranked in ranked_lists:
            for rank, item in enumerate(source_ranked, start=1):
                doc_id = str(item["id"])
                if doc_id not in fused:
                    fused[doc_id] = {
                        "id": doc_id,
                        "document": item.get("document", ""),
                        "metadata": item.get("metadata", {}),
                        "rrf_score": 0.0,
                        "sources": [],
                    }
                fused[doc_id]["rrf_score"] += 1.0 / (self.rrf_k + rank)
                fused[doc_id]["sources"].append(
                    {
                        "rank": rank,
                        "score": item.get("score"),
                    }
                )

        merged = sorted(fused.values(), key=lambda item: item["rrf_score"], reverse=True)
        return merged

    @staticmethod
    def _matches_filters(metadata: dict[str, Any], filters: dict[str, Any] | None) -> bool:
        if not filters:
            return True
        for key, expected in filters.items():
            if isinstance(expected, dict):
                continue
            if metadata.get(key) != expected:
                return False
        return True

    def search(self, query: str, top_k_each: int = 15, filters: dict[str, Any] | None = None) -> list[dict[str, Any]]:
        """Search both retrievers independently and fuse the rankings with RRF."""

        # We intentionally fetch a wider candidate pool first, then re-rank later.
        # This "retrieve wide, rerank narrow" pattern keeps recall high while
        # reserving expensive cross-encoder scoring for only the best candidates.
        query_embedding = self.embedder.embed([query])[0]
        vector_results = self.vector_store.query(query_embedding=query_embedding, top_k=top_k_each, filters=filters)
        bm25_results = self.bm25_index.search(query=query, top_k=top_k_each * 3 if filters else top_k_each)

        bm25_ranked: list[dict[str, Any]] = []
        if bm25_results:
            # BM25 only knows doc ids and scores, so we fetch the stored text from Chroma
            # for downstream reranking and inspection.
            bm25_ids = [doc_id for doc_id, _ in bm25_results]
            documents = self.vector_store.collection.get(ids=bm25_ids, include=["documents", "metadatas"])
            doc_lookup = {
                str(doc_id): {
                    "document": document,
                    "metadata": metadata or {},
                }
                for doc_id, document, metadata in zip(
                    documents.get("ids", []),
                    documents.get("documents", []),
                    documents.get("metadatas", []),
                )
            }
            for doc_id, score in bm25_results:
                payload = doc_lookup.get(str(doc_id), {"document": "", "metadata": {}})
                if not self._matches_filters(payload["metadata"], filters):
                    continue
                bm25_ranked.append(
                    {
                        "id": str(doc_id),
                        "document": payload["document"],
                        "metadata": payload["metadata"],
                        "score": float(score),
                    }
                )

        return self._rrf_merge([vector_results, bm25_ranked])
