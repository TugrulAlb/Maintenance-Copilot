"""SlowAPI-backed request rate limiting."""

from __future__ import annotations

import os

from fastapi import Request
from slowapi import Limiter


def get_rate_limit_per_minute() -> int:
    value = os.getenv("MAINTENANCE_COPILOT_RATE_LIMIT_PER_MINUTE", "20")
    try:
        return int(value)
    except ValueError:
        return 20


def rate_limit_identity(request: Request) -> str:
    api_key = request.headers.get("X-API-Key")
    if api_key:
        return f"api-key:{api_key}"
    if request.client and request.client.host:
        return f"ip:{request.client.host}"
    return "anonymous"


def ask_rate_limit_rule() -> str:
    limit = get_rate_limit_per_minute()
    if limit <= 0:
        return "1000000/minute"
    return f"{limit}/minute"


limiter = Limiter(key_func=rate_limit_identity)


def reset_rate_limiter() -> None:
    """Clear in-memory SlowAPI counters for tests."""

    limiter.reset()
