"""BM25 retrieval over the same maintenance-log corpus used for embeddings."""

from __future__ import annotations

import pickle
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from rank_bm25 import BM25Okapi


def tokenize(text: str) -> list[str]:
    """Lowercase and split on punctuation/whitespace for a simple lexical index."""

    return [token for token in re.split(r"[^a-z0-9]+", text.lower()) if token]


@dataclass
class _PersistedIndex:
    doc_ids: list[str]
    tokenized_corpus: list[list[str]]


class BM25Index:
    """Persistent BM25 index for hybrid retrieval."""

    def __init__(self, persist_path: str | Path | None = None) -> None:
        project_root = Path(__file__).resolve().parents[1]
        self.persist_path = Path(persist_path) if persist_path is not None else project_root / "retrieval" / "bm25_index.pkl"
        self.doc_ids: list[str] = []
        self.tokenized_corpus: list[list[str]] = []
        self._bm25: BM25Okapi | None = None

    def build_index(self, documents: Iterable[tuple[str, str]]) -> None:
        """Build the BM25 model and persist the tokenized corpus to disk."""

        self.doc_ids = []
        self.tokenized_corpus = []

        for doc_id, text in documents:
            self.doc_ids.append(str(doc_id))
            self.tokenized_corpus.append(tokenize(text))

        self._bm25 = BM25Okapi(self.tokenized_corpus)
        self.persist_path.parent.mkdir(parents=True, exist_ok=True)
        with self.persist_path.open("wb") as handle:
            pickle.dump(
                _PersistedIndex(doc_ids=self.doc_ids, tokenized_corpus=self.tokenized_corpus),
                handle,
            )

    def _load(self) -> None:
        """Load a previously persisted corpus if the in-memory index is empty."""

        if self._bm25 is not None:
            return
        with self.persist_path.open("rb") as handle:
            persisted: _PersistedIndex = pickle.load(handle)
        self.doc_ids = persisted.doc_ids
        self.tokenized_corpus = persisted.tokenized_corpus
        self._bm25 = BM25Okapi(self.tokenized_corpus)

    def search(self, query: str, top_k: int) -> list[tuple[str, float]]:
        """Return document ids ranked by BM25 score."""

        self._load()
        assert self._bm25 is not None
        scores = self._bm25.get_scores(tokenize(query))
        ranked = sorted(zip(self.doc_ids, scores), key=lambda item: item[1], reverse=True)
        return ranked[:top_k]