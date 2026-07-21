"""Chunk maintenance records into retrievable text blocks.

Design note:
- Each synthetic maintenance record is already short, so we keep it as a single
  chunk today.
- The function is intentionally generic so the same interface can later support
  longer documents such as manuals or work instructions.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any


def _get_value(record: Any, key: str, default: str = "") -> str:
    """Read a field from either a mapping or a row-like object."""

    if isinstance(record, dict):
        value = record.get(key, default)
    else:
        value = record[key] if key in record.keys() else default
    return str(value)


def chunk_record(record: Any) -> str:
    """Convert one maintenance record into a single search chunk.

    The chunk includes only the fields that are most useful for semantic search:
    the event time, line, machine, fault label, and technician description.
    """

    timestamp = _get_value(record, "timestamp")
    production_line = _get_value(record, "production_line")
    machine_id = _get_value(record, "machine_id")
    fault_category = _get_value(record, "fault_category")
    description = _get_value(record, "description")

    try:
        # Normalize ISO timestamps when possible so textual chunks stay consistent.
        timestamp = datetime.fromisoformat(timestamp.replace("Z", "+00:00")).isoformat()
    except ValueError:
        pass

    parts = [
        f"timestamp: {timestamp}",
        f"production_line: {production_line}",
        f"machine_id: {machine_id}",
        f"fault_category: {fault_category}",
        f"description: {description}",
    ]
    return "\n".join(parts)