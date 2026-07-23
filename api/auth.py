"""Authentication and authorization helpers for the API."""

from __future__ import annotations

import os
from dataclasses import dataclass

from collections.abc import Sequence

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


def _configured_key_roles() -> dict[str, str]:
    """Read simple key:role pairs from env and preserve legacy admin-key config."""

    configured_keys = _configured_api_keys()
    admin_keys = _configured_admin_keys()
    roles = {api_key: ("admin" if api_key in admin_keys else "viewer") for api_key in configured_keys}
    raw_mapping = os.getenv("MAINTENANCE_COPILOT_API_KEY_ROLES", "")
    for item in raw_mapping.split(","):
        if ":" not in item:
            continue
        api_key, role = item.split(":", 1)
        api_key = api_key.strip()
        role = role.strip()
        if api_key and role:
            roles[api_key] = role
    return roles


def _allow_public_demo_mode() -> bool:
    return os.getenv("MAINTENANCE_COPILOT_ALLOW_PUBLIC", "true").strip().lower() in {"1", "true", "yes"}


def verify_api_key(
    request: Request,
    x_api_key: str | None = Header(default=None, alias="X-API-Key"),
) -> AccessContext:
    """Authenticate requests via a deliberately simple API-key scheme.

    This is intentionally scoped for a portfolio/internal-tool project. A real
    multi-user consumer product would normally use JWT/OAuth2/OIDC with bearer
    tokens, refresh flows, and enterprise identity integration. API keys are
    still a practical fit for service-to-service calls or small internal demos.
    """

    key_roles = _configured_key_roles()

    if not key_roles and _allow_public_demo_mode():
        return AccessContext(api_key=x_api_key, role="public", is_authenticated=False)

    if not key_roles:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="API keys are required")

    if not x_api_key or x_api_key not in key_roles:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid or missing API key")

    role = key_roles[x_api_key]
    request.state.api_key = x_api_key
    request.state.role = role
    return AccessContext(api_key=x_api_key, role=role, is_authenticated=True)


get_access_context = verify_api_key


def require_role(allowed_roles: Sequence[str], *, allow_public: bool = False):
    """Create a FastAPI dependency enforcing access roles."""

    allowed = set(allowed_roles)

    def dependency(context: AccessContext = Depends(verify_api_key)) -> AccessContext:
        if allow_public and context.role == "public":
            return context
        if context.role not in allowed:
            raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Insufficient permissions")
        return context

    return dependency


def require_roles(*allowed_roles: str):
    """Backward-compatible wrapper for older call sites."""

    return require_role(list(allowed_roles), allow_public=True)
