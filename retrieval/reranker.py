"""Cross-encoder reranking for a smaller candidate set.

Design note:
- This is the second stage in a two-stage retrieval pipeline.
- We first retrieve a wider candidate set, then rerank narrowly with a more
  expensive but more accurate model before passing the final results to the LLM.
"""

from __future__ import annotations

from typing import Any

from sentence_transformers import CrossEncoder


class Reranker:
    """Cross-encoder reranker for query-candidate relevance scoring."""

    def __init__(self, model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2") -> None:
        self.model = CrossEncoder(model_name)

    def rerank(self, query: str, candidates: list[dict[str, Any]], top_k: int = 5) -> list[dict[str, Any]]:
        """Score candidate texts against the query and return the best matches."""

        if not candidates:
            return []

        pairs = [(query, candidate.get("document", "")) for candidate in candidates]
        scores = self.model.predict(pairs)

        ranked: list[dict[str, Any]] = []
        for candidate, score in zip(candidates, scores):
            enriched = dict(candidate)
            enriched["rerank_score"] = float(score)
            ranked.append(enriched)

        ranked.sort(key=lambda item: item["rerank_score"], reverse=True)
        return ranked[:top_k]