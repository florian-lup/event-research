"""Singleton accessor for the Tavily client."""

from __future__ import annotations

from tavily import TavilyClient

from ..config import TAVILY_API_KEY

_client: TavilyClient | None = None


def get_tavily_client() -> TavilyClient:
    """Return a singleton instance of :class:`tavily.TavilyClient`."""
    global _client
    if _client is None:
        if not TAVILY_API_KEY:
            raise EnvironmentError("TAVILY_API_KEY is not set in environment variables")
        _client = TavilyClient(api_key=TAVILY_API_KEY)
    return _client

__all__ = ["get_tavily_client"] 