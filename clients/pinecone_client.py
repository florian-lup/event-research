"""Singleton accessor for Pinecone and helper for obtaining the Index."""

from __future__ import annotations

from pinecone import Pinecone as _Pinecone, Index

from config import PINECONE_API_KEY, PINECONE_ENVIRONMENT, PINECONE_INDEX_NAME

_pc: _Pinecone | None = None


def get_pinecone() -> _Pinecone:
    """Return a singleton :class:`pinecone.Pinecone` client."""
    global _pc
    if _pc is None:
        _pc = _Pinecone(api_key=PINECONE_API_KEY, environment=PINECONE_ENVIRONMENT)
    return _pc


def get_index() -> Index:  # type: ignore[name-defined]
    """Return the configured Pinecone Index instance."""
    return get_pinecone().Index(PINECONE_INDEX_NAME)

__all__ = ["get_pinecone", "get_index"] 