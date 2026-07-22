"""OpenAI-compatible embedding wrapper for batched vector generation.

Design note:
- Embeddings are requested in batches to reduce latency and API overhead.
- The wrapper hides provider-specific details so later retrieval code can stay
  focused on data flow rather than client configuration.
"""

from __future__ import annotations

import os
from typing import Sequence

from openai import AzureOpenAI, OpenAI

from graph.llm_client import build_chat_client


class EmbeddingClient:
    """Batch embedding client using OpenAI-compatible APIs."""

    def __init__(self, model: str | None = None, batch_size: int = 64) -> None:
        self.model = model or os.getenv("AZURE_OPENAI_EMBEDDING_DEPLOYMENT_NAME") or os.getenv(
            "OPENAI_EMBEDDING_MODEL", "text-embedding-3-small"
        )
        self.batch_size = batch_size

        client = build_chat_client()
        if client is None:
            self._client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        else:
            self._client: AzureOpenAI | OpenAI = client

    def embed(self, texts: list[str]) -> list[list[float]]:
        """Embed texts in batches and preserve input order."""

        embeddings: list[list[float]] = []
        for index in range(0, len(texts), self.batch_size):
            batch = texts[index : index + self.batch_size]
            response = self._client.embeddings.create(model=self.model, input=batch)
            batch_embeddings = [item.embedding for item in response.data]
            embeddings.extend(batch_embeddings)
        return embeddings
