"""Read maintenance logs from SQLite, embed them, and persist them in ChromaDB."""

from __future__ import annotations

import sqlite3
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    # Allow direct execution with `python ingestion/run_ingestion.py` from the project root.
    sys.path.insert(0, str(PROJECT_ROOT))

from ingestion.chunker import chunk_record
from ingestion.embedder import EmbeddingClient
from ingestion.vector_store import VectorStore


def load_records(database_path: Path) -> list[sqlite3.Row]:
    """Load all maintenance records from SQLite as row objects."""

    connection = sqlite3.connect(database_path)
    connection.row_factory = sqlite3.Row
    try:
        cursor = connection.execute(
            """
            SELECT id, timestamp, production_line, machine_id, fault_category,
                   description, severity, resolution_time_minutes, resolved_by
            FROM maintenance_logs
            ORDER BY id
            """
        )
        return cursor.fetchall()
    finally:
        connection.close()


def main() -> None:
    """End-to-end ingestion entry point."""

    project_root = PROJECT_ROOT
    database_path = project_root / "data" / "maintenance_logs.db"
    records = load_records(database_path)

    embedder = EmbeddingClient()
    vector_store = VectorStore(persist_dir=project_root / "chroma_data")

    batch_size = embedder.batch_size
    for batch_start in range(0, len(records), batch_size):
        batch_records = records[batch_start : batch_start + batch_size]
        texts = [chunk_record(record) for record in batch_records]
        embeddings = embedder.embed(texts)

        for record, text, embedding in zip(batch_records, texts, embeddings):
            vector_store.upsert(
                id=str(record["id"]),
                text=text,
                embedding=embedding,
                metadata={
                    "production_line": record["production_line"],
                    "machine_id": record["machine_id"],
                    "fault_category": record["fault_category"],
                    "severity": record["severity"],
                    "timestamp": record["timestamp"],
                },
            )

        processed = min(batch_start + len(batch_records), len(records))
        print(f"Ingested {processed}/{len(records)} records")


if __name__ == "__main__":
    main()