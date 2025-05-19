"""Singleton accessor for the MongoDB client."""

from __future__ import annotations

from pymongo import MongoClient

from ..config import MONGODB_URI

_client: MongoClient | None = None


def get_mongo_client() -> MongoClient:
    """Return a singleton :class:`pymongo.MongoClient`."""
    global _client
    if _client is None:
        _client = MongoClient(MONGODB_URI)
    return _client

__all__ = ["get_mongo_client"] 