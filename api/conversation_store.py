"""In-memory conversation history store keyed by thread id."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from threading import Lock


@dataclass
class ConversationTurn:
    question: str
    answer: str
    query_type: str
    citations: list[str] = field(default_factory=list)


class ConversationStore:
    """Persist conversation context for the lifetime of the process."""

    def __init__(self) -> None:
        self._lock = Lock()
        self._history: dict[str, list[ConversationTurn]] = defaultdict(list)

    def get_history(self, thread_id: str) -> list[dict[str, object]]:
        with self._lock:
            return [
                {
                    "question": turn.question,
                    "answer": turn.answer,
                    "query_type": turn.query_type,
                    "citations": list(turn.citations),
                }
                for turn in self._history.get(thread_id, [])
            ]

    def append_turn(self, thread_id: str, question: str, answer: str, query_type: str, citations: list[str]) -> None:
        with self._lock:
            self._history[thread_id].append(
                ConversationTurn(
                    question=question,
                    answer=answer,
                    query_type=query_type,
                    citations=list(citations),
                )
            )
            self._history[thread_id] = self._history[thread_id][-10:]

    def clear(self) -> None:
        with self._lock:
            self._history.clear()


CONVERSATIONS = ConversationStore()
