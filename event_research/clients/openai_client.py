"""Singleton accessor for the OpenAI SDK client."""

from __future__ import annotations

from openai import OpenAI as _OpenAIClient

from ..config import OPENAI_API_KEY

_client: _OpenAIClient | None = None


def get_openai() -> _OpenAIClient:
    """Return a singleton instance of :class:`openai.OpenAI`."""
    global _client
    if _client is None:
        _client = _OpenAIClient(api_key=OPENAI_API_KEY)
    return _client

__all__ = ["get_openai"] 