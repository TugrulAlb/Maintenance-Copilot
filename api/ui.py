"""Static demo UI loader for Maintenance Copilot."""

from __future__ import annotations

from pathlib import Path


STATIC_INDEX = Path(__file__).resolve().parent / "static" / "index.html"


def render_index_html() -> str:
    """Render the plain HTML demo UI without frontend build tooling."""

    return STATIC_INDEX.read_text(encoding="utf-8")
