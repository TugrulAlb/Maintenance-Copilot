"""BM25 retrieval over the same maintenance-log corpus used for embeddings."""

from __future__ import annotations

import pickle
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

try:
    from rank_bm25 import BM25Okapi as ExternalBM25Okapi
except ImportError:  # pragma: no cover - fallback for minimal environments
    ExternalBM25Okapi = None


def tokenize(text: str) -> list[str]:
    """Lowercase and split on punctuation/whitespace for a simple lexical index."""

    return [token for token in re.split(r"[^a-z0-9]+", text.lower()) if token]


@dataclass
class _PersistedIndex:
    doc_ids: list[str]
    tokenized_corpus: list[list[str]]


class _FallbackBM25Okapi:
    """Tiny BM25 implementation used when the external package is unavailable."""

    def __init__(self, tokenized_corpus: list[list[str]]) -> None:
        self.tokenized_corpus = tokenized_corpus
        self.doc_freqs: list[dict[str, int]] = []
        self.doc_lengths: list[int] = []
        self.doc_freq: dict[str, int] = {}
        self.corpus_size = len(tokenized_corpus)
        self.avgdl = sum(len(doc) for doc in tokenized_corpus) / max(1, self.corpus_size)
        for doc in tokenized_corpus:
            freq: dict[str, int] = {}
            for token in doc:
                freq[token] = freq.get(token, 0) + 1
            self.doc_freqs.append(freq)
            self.doc_lengths.append(len(doc))
            for token in freq:
                self.doc_freq[token] = self.doc_freq.get(token, 0) + 1

    def get_scores(self, query_tokens: list[str]) -> list[float]:
        scores: list[float] = []
        k1 = 1.5
        b = 0.75
        for index, freq in enumerate(self.doc_freqs):
            score = 0.0
            doc_len = self.doc_lengths[index] or 1
            for token in query_tokens:
                if token not in freq:
                    continue
                df = self.doc_freq.get(token, 0)
                idf = max(0.0, ((self.corpus_size - df + 0.5) / (df + 0.5)))
                term_freq = freq[token]
                denom = term_freq + k1 * (1 - b + b * (doc_len / max(1e-9, self.avgdl)))
                score += idf * ((term_freq * (k1 + 1)) / denom)
            scores.append(score)
        return scores


class BM25Index:
    """Persistent BM25 index for hybrid retrieval."""

    def __init__(self, persist_path: str | Path | None = None) -> None:
        project_root = Path(__file__).resolve().parents[1]
        self.persist_path = Path(persist_path) if persist_path is not None else project_root / "retrieval" / "bm25_index.pkl"
        self.doc_ids: list[str] = []
        self.tokenized_corpus: list[list[str]] = []
        self._bm25: ExternalBM25Okapi | _FallbackBM25Okapi | None = None

    def build_index(self, documents: Iterable[tuple[str, str]]) -> None:
        """Build the BM25 model and persist the tokenized corpus to disk."""

        self.doc_ids = []
        self.tokenized_corpus = []

        for doc_id, text in documents:
            self.doc_ids.append(str(doc_id))
            self.tokenized_corpus.append(tokenize(text))

        self._bm25 = ExternalBM25Okapi(self.tokenized_corpus) if ExternalBM25Okapi else _FallbackBM25Okapi(self.tokenized_corpus)
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
        self._bm25 = ExternalBM25Okapi(self.tokenized_corpus) if ExternalBM25Okapi else _FallbackBM25Okapi(self.tokenized_corpus)

    def search(self, query: str, top_k: int) -> list[tuple[str, float]]:
        """Return document ids ranked by BM25 score."""

        self._load()
        assert self._bm25 is not None
        scores = self._bm25.get_scores(tokenize(query))
        ranked = sorted(zip(self.doc_ids, scores), key=lambda item: item[1], reverse=True)
        return ranked[:top_k]