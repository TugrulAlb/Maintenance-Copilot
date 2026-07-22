"""Shared OpenAI/Azure OpenAI client configuration.

The project supports both classic Azure OpenAI env vars and Azure's
OpenAI-compatible `/openai/v1` endpoint shape. Keeping this logic in one place
prevents classifier, SQL agent, and answer generation from drifting apart.
"""

from __future__ import annotations

import os

from openai import AzureOpenAI, OpenAI


def _env(*names: str) -> str | None:
    for name in names:
        value = os.getenv(name)
        if value:
            return value.strip()
    return None


def build_chat_client() -> AzureOpenAI | OpenAI | None:
    """Build a chat client from configured environment variables."""

    azure_key = _env("AZURE_OPENAI_API_KEY")
    azure_endpoint = _env("AZURE_OPENAI_ENDPOINT", "AZURE_OPENAI_BASE_URL")
    openai_key = _env("OPENAI_API_KEY")
    openai_base_url = _env("OPENAI_BASE_URL")

    if azure_key and azure_endpoint:
        if azure_endpoint.rstrip("/").endswith("/openai/v1"):
            return OpenAI(api_key=azure_key, base_url=azure_endpoint.rstrip("/"))
        return AzureOpenAI(
            api_key=azure_key,
            azure_endpoint=azure_endpoint.rstrip("/"),
            api_version=os.getenv("AZURE_OPENAI_API_VERSION", "2024-06-01"),
        )

    if openai_key:
        kwargs = {"api_key": openai_key}
        if openai_base_url:
            kwargs["base_url"] = openai_base_url.rstrip("/")
        return OpenAI(**kwargs)

    return None


def chat_model(*specific_env_names: str, default: str = "gpt-4o-mini") -> str:
    """Resolve the model/deployment name for a chat completion call."""

    return (
        _env(
            *specific_env_names,
            "AZURE_OPENAI_DEPLOYMENT_NAME",
            "OPENAI_MODEL",
        )
        or default
    )
