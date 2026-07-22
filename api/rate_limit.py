"""Simple in-memory request rate limiting."""

from __future__ import annotations

import os
import time
from collections import defaultdict, deque
from dataclasses import dataclass
from threading import Lock

from fastapi import HTTPException, Request, status


@dataclass
class RateLimitResult:
    allowed: bool
    retry_after_seconds: int


class InMemoryRateLimiter:
    """Fixed-window rate limiter keyed by an identity string."""

    def __init__(self) -> None:
        self._lock = Lock()
        self._events: dict[str, deque[float]] = defaultdict(deque)

    def allow(self, identity: str, limit_per_minute: int) -> RateLimitResult:
        now = time.time()
        window_start = now - 60
        with self._lock:
            events = self._events[identity]
            while events and events[0] < window_start:
                events.popleft()
            if len(events) >= limit_per_minute:
                retry_after = int(max(1, 60 - (now - events[0])))
                return RateLimitResult(False, retry_after)
            events.append(now)
            return RateLimitResult(True, 0)

    def clear(self) -> None:
        with self._lock:
            self._events.clear()


_RATE_LIMITER = InMemoryRateLimiter()


def get_rate_limit_per_minute() -> int:
    value = os.getenv("MAINTENANCE_COPILOT_RATE_LIMIT_PER_MINUTE", "60")
    try:
        return int(value)
    except ValueError:
        return 60


def rate_limit_identity(request: Request) -> str:
    api_key = request.headers.get("X-API-Key")
    if api_key:
        return f"api-key:{api_key}"
    if request.client and request.client.host:
        return f"ip:{request.client.host}"
    return "anonymous"


def enforce_rate_limit(request: Request) -> None:
    limit = get_rate_limit_per_minute()
    if limit <= 0:
        return

    result = _RATE_LIMITER.allow(rate_limit_identity(request), limit)
    if not result.allowed:
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail="Rate limit exceeded",
            headers={"Retry-After": str(result.retry_after_seconds)},
        )


def reset_rate_limiter() -> None:
    _RATE_LIMITER.clear()
