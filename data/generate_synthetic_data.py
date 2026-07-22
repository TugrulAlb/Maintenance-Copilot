"""Generate synthetic industrial maintenance logs for semantic and analytical testing.

Design notes:
- The LLM is only responsible for the free-text field and the fault label.
- Structured fields are generated locally so the dataset stays reproducible, cheap,
  and easy to reason about during interviews.
- The script writes a single SQLite database file that can later be reused by the
  ingestion, retrieval, and graph layers.
"""

from __future__ import annotations

import argparse
import json
import random
import sqlite3
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Iterable

from dotenv import load_dotenv
from openai import AzureOpenAI, OpenAI

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from graph.llm_client import build_chat_client, chat_model


FAULT_CATEGORIES = [
    "motor failure",
    "sensor error",
    "belt misalignment",
    "overheating",
    "electrical fault",
]

SEVERITIES = ["low", "medium", "high"]
PRODUCTION_LINES = ["Line 1", "Line 2", "Line 3"]
RESOLVED_BY = ["Alice", "Ben", "Carlos", "Diana", "Ethan", "Fatima", "Grace"]


@dataclass(frozen=True)
class GeneratedDescription:
    """A single LLM-produced maintenance note and its coarse label."""

    fault_category: str
    description: str


def build_client() -> AzureOpenAI | OpenAI:
    """Create an OpenAI-compatible client from environment variables.

    Azure OpenAI is the preferred path for this project because the interview
    portfolio can later show cloud readiness without changing the script logic.
    """

    client = build_chat_client()
    if client is not None:
        return client

    raise RuntimeError(
        "Set Azure OpenAI env vars or OPENAI_API_KEY before generating data."
    )


def create_schema(connection: sqlite3.Connection) -> None:
    """Create the database table with a narrow schema that matches the use case."""

    connection.execute(
        """
        CREATE TABLE IF NOT EXISTS maintenance_logs (
            id INTEGER PRIMARY KEY,
            timestamp TEXT NOT NULL,
            production_line TEXT NOT NULL,
            machine_id TEXT NOT NULL,
            fault_category TEXT NOT NULL,
            description TEXT NOT NULL,
            severity TEXT NOT NULL,
            resolution_time_minutes INTEGER NOT NULL,
            resolved_by TEXT NOT NULL
        )
        """
    )
    connection.commit()


def generate_prompt(batch_size: int, seed_hint: int) -> str:
    """Ask the model for varied technician notes, not templated prose.

    The prompt intentionally requests short, messy, operator-style language so
    semantic search and embedding tests face realistic variation.
    """

    return f"""
You are generating industrial maintenance ticket text for a factory log.

    Return one JSON object with a top-level `items` array containing exactly {batch_size} objects.
    Each object in `items` must have:
    - fault_category: one of {FAULT_CATEGORIES}
    - description: a realistic technician note in first person or shorthand

Rules:
- Write like a busy technician entering a fault ticket.
- Use varied phrasing, abbreviations, partial sentences, and occasional minor typos.
- Keep each description short to medium length, about 12-35 words.
- Avoid duplicate wording inside the batch.
- Mix all categories naturally.
- Do not add any other fields.

Seed hint for variation: {seed_hint}
""".strip()


def generate_description_batch(client: AzureOpenAI | OpenAI, batch_size: int, seed_hint: int) -> list[GeneratedDescription]:
    """Generate one batch of LLM-produced tickets.

    A moderate batch size reduces API overhead while still giving the model
    enough room to vary its phrasing within each request.
    """

    response = client.chat.completions.create(
        model=chat_model(),
        temperature=1.0,
        response_format={"type": "json_object"},
        messages=[
            {"role": "system", "content": "You write realistic maintenance log entries."},
            {"role": "user", "content": generate_prompt(batch_size=batch_size, seed_hint=seed_hint)},
        ],
    )

    raw_text = response.choices[0].message.content or "[]"
    parsed = json.loads(raw_text)
    if isinstance(parsed, dict):
        parsed = parsed.get("items", parsed.get("records", []))

    return [
        GeneratedDescription(
            fault_category=str(item["fault_category"]).strip(),
            description=str(item["description"]).strip(),
        )
        for item in parsed
    ]


def build_record(record_id: int, description: GeneratedDescription, rng: random.Random) -> dict[str, object]:
    """Create a full synthetic row around the LLM text.

    The local metadata keeps the dataset grounded in realistic operational patterns
    while the language model handles the expensive part: natural language variety.
    """

    now = datetime.now(timezone.utc)
    timestamp = now - timedelta(
        days=rng.randint(0, 365),
        hours=rng.randint(0, 23),
        minutes=rng.randint(0, 59),
    )

    severity_bias = {
        "motor failure": ["medium", "high"],
        "sensor error": ["low", "medium"],
        "belt misalignment": ["low", "medium"],
        "overheating": ["medium", "high"],
        "electrical fault": ["medium", "high"],
    }
    severity = rng.choice(severity_bias.get(description.fault_category, SEVERITIES))
    resolution_time = {
        "low": rng.randint(10, 45),
        "medium": rng.randint(35, 180),
        "high": rng.randint(120, 720),
    }[severity]

    return {
        "id": record_id,
        "timestamp": timestamp.isoformat(),
        "production_line": rng.choice(PRODUCTION_LINES),
        "machine_id": f"MC-{rng.randint(100, 599)}",
        "fault_category": description.fault_category,
        "description": description.description,
        "severity": severity,
        "resolution_time_minutes": resolution_time,
        "resolved_by": rng.choice(RESOLVED_BY),
    }


def insert_rows(connection: sqlite3.Connection, rows: list[dict[str, object]]) -> None:
    """Insert rows with a single transaction for speed and consistency."""

    connection.executemany(
        """
        INSERT INTO maintenance_logs (
            id, timestamp, production_line, machine_id, fault_category,
            description, severity, resolution_time_minutes, resolved_by
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        [
            (
                row["id"],
                row["timestamp"],
                row["production_line"],
                row["machine_id"],
                row["fault_category"],
                row["description"],
                row["severity"],
                row["resolution_time_minutes"],
                row["resolved_by"],
            )
            for row in rows
        ],
    )
    connection.commit()


def generate_dataset(record_count: int, batch_size: int, max_workers: int, seed: int) -> list[dict[str, object]]:
    """Generate a full dataset using concurrent LLM batches.

    Parallel batches keep the wall-clock time reasonable for a 500-row dataset
    without making the implementation hard to follow.
    """

    client = build_client()
    rng = random.Random(seed)
    batch_count = (record_count + batch_size - 1) // batch_size
    batch_seeds = [rng.randint(0, 1_000_000) for _ in range(batch_count)]

    batches: list[list[GeneratedDescription]] = []
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(generate_description_batch, client, min(batch_size, record_count - index * batch_size), batch_seed): index
            for index, batch_seed in enumerate(batch_seeds)
        }
        ordered_results: list[list[GeneratedDescription] | None] = [None] * batch_count
        for future in as_completed(futures):
            batch_index = futures[future]
            ordered_results[batch_index] = future.result()

    for batch in ordered_results:
        if batch is None:
            continue
        batches.append(batch)

    flat_descriptions = [description for batch in batches for description in batch]
    if len(flat_descriptions) < record_count:
        raise RuntimeError(
            f"LLM returned only {len(flat_descriptions)} descriptions for {record_count} records."
        )

    records: list[dict[str, object]] = []
    for record_id, description in enumerate(flat_descriptions[:record_count], start=1):
        records.append(build_record(record_id, description, rng))
    return records


def main() -> None:
    """CLI entry point for generating the SQLite dataset."""

    load_dotenv()

    parser = argparse.ArgumentParser(description="Generate synthetic maintenance logs into SQLite.")
    parser.add_argument("--records", type=int, default=500, help="Number of records to generate.")
    parser.add_argument("--batch-size", type=int, default=50, help="LLM descriptions per request.")
    parser.add_argument("--max-workers", type=int, default=4, help="Parallel LLM request count.")
    parser.add_argument("--seed", type=int, default=42, help="Seed for local synthetic metadata.")
    args = parser.parse_args()

    output_path = Path(__file__).resolve().parent / "maintenance_logs.db"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    connection = sqlite3.connect(output_path)
    try:
        create_schema(connection)
        connection.execute("DELETE FROM maintenance_logs")
        rows = generate_dataset(
            record_count=args.records,
            batch_size=args.batch_size,
            max_workers=args.max_workers,
            seed=args.seed,
        )
        insert_rows(connection, rows)
    finally:
        connection.close()

    print(f"Saved {args.records} synthetic maintenance records to {output_path}")


if __name__ == "__main__":
    main()
