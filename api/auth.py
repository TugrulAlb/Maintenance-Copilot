"""Authentication and authorization helpers for the API."""

from __future__ import annotations

import os
from dataclasses import dataclass

from fastapi import Depends, Header, HTTPException, Request, status


@dataclass(frozen=True)
class AccessContext:
    """Resolved access information for an incoming API request."""

    api_key: str | None
    role: str
    is_authenticated: bool


def _split_env_list(value: str | None) -> set[str]:
    if not value:
        return set()
    return {item.strip() for item in value.split(",") if item.strip()}


def _configured_api_keys() -> set[str]:
    return _split_env_list(os.getenv("MAINTENANCE_COPILOT_API_KEYS"))


def _configured_admin_keys() -> set[str]:
    return _split_env_list(os.getenv("MAINTENANCE_COPILOT_ADMIN_KEYS"))


def _allow_public_demo_mode() -> bool:
    return os.getenv("MAINTENANCE_COPILOT_ALLOW_PUBLIC", "true").strip().lower() in {"1", "true", "yes"}


def get_access_context(
    request: Request,
    x_api_key: str | None = Header(default=None, alias="X-API-Key"),
) -> AccessContext:
    """Authenticate requests via an API key when keys are configured.

    If no keys are configured, the project stays runnable in local demo mode.
    """

    configured_keys = _configured_api_keys()
    admin_keys = _configured_admin_keys()

    if not configured_keys and _allow_public_demo_mode():
        return AccessContext(api_key=x_api_key, role="public", is_authenticated=False)

    if not configured_keys:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="API keys are required")

    if not x_api_key or x_api_key not in configured_keys:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid or missing API key")

    role = "admin" if x_api_key in admin_keys else "user"
    request.state.api_key = x_api_key
    request.state.role = role
    return AccessContext(api_key=x_api_key, role=role, is_authenticated=True)


def require_roles(*allowed_roles: str):
    """Create a FastAPI dependency enforcing access roles."""

    def dependency(context: AccessContext = Depends(get_access_context)) -> AccessContext:
        if context.role not in allowed_roles and context.role != "public":
            raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Insufficient permissions")
        return context

    return dependency
