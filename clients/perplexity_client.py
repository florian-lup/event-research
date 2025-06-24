"""Shared HTTP session for Perplexity API calls."""

from __future__ import annotations

import requests

_session: requests.Session | None = None


def get_session() -> requests.Session:
    """Return a singleton :class:`requests.Session` configured for Perplexity."""
    global _session
    if _session is None:
        _session = requests.Session()
    return _session

# Backwards-compat alias expected by services.discovery

def get_perplexity_session() -> requests.Session:  # noqa: D401
    """Alias for :func:`get_session` kept for import compatibility."""
    return get_session()

__all__ = ["get_session", "get_perplexity_session"] 