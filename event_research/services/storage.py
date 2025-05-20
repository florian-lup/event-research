"""Persistence layer: Pinecone vector upsert + MongoDB document storage."""

from __future__ import annotations

import logging
from typing import Dict, Any, List, Tuple

from pinecone import Index  # type: ignore

from ..clients.mongodb_client import get_mongo_client
from ..config import (
    DEDUPLICATION_NAMESPACE,
    research_NAMESPACE,
)
from .embeddings import generate_embedding, chunk_text

logger = logging.getLogger(__name__)
_db = get_mongo_client()["events"]["global"]


def upsert_to_pinecone(
    pinecone_index: Index,
    event: Dict[str, Any],
    overview_embedding: List[float],
) -> None:
    """Store overview + chunk vectors for *event* inside Pinecone."""
    logger.info("Upserting event to Pinecone: %s", event["title"])

    event_id = f"event_{hash(event['title'])}"

    # 1) Overview vector (title + summary) goes to deduplication namespace
    overview_metadata = {
        "event_id": event_id,
        "title": event["title"],
        "summary": event["summary"],
    }
    pinecone_index.upsert(
        namespace=DEDUPLICATION_NAMESPACE,
        vectors=[(event_id, overview_embedding, overview_metadata)],
    )

    # 2) Chunked research vectors
    research: str = event.get("research", "")
    if not research:
        logger.warning("No research for event %s – skipping chunk upsert", event["title"])
        return

    vectors: List[Tuple[str, List[float], Dict[str, Any]]] = []
    for idx, chunk in enumerate(chunk_text(research)):
        chunk_id = f"{event_id}_chunk_{idx}"
        chunk_embedding = generate_embedding(chunk)
        chunk_metadata = {
            "event_id": event_id,
            "chunk_index": idx,
            "title": event["title"],
            "sources": event.get("sources", []),
            "text": chunk,
        }
        vectors.append((chunk_id, chunk_embedding, chunk_metadata))

    pinecone_index.upsert(namespace=research_NAMESPACE, vectors=vectors)
    logger.info("Upserted %d research chunks for event %s", len(vectors), event["title"])


def store_to_mongodb(event: Dict[str, Any]) -> None:
    """Insert *event* document into MongoDB."""
    result = _db.insert_one(event)
    logger.info("Stored event to MongoDB with _id=%s", result.inserted_id)

__all__ = ["upsert_to_pinecone", "store_to_mongodb"] 