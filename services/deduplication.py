"""Duplicate detection using Pinecone similarity search."""

from __future__ import annotations

import logging
from typing import Dict, Any, List

from pinecone import Index  # type: ignore

from config import DEDUPLICATION_NAMESPACE

# ---------------------------------------------------------------------------
# Local deduplication thresholds
# ---------------------------------------------------------------------------
SIMILARITY_THRESHOLD: float = 0.8
DEDUPLICATION_TOP_K: int = 5

logger = logging.getLogger(__name__)


def check_duplicates(
    pinecone_index: Index,
    embedding: List[float],
    metadata: Dict[str, Any],
) -> bool:
    """Return ``True`` if an existing vector is semantically similar (≥ threshold)."""
    logger.info("Checking for duplicates for event: %s", metadata.get("title"))

    query_response = pinecone_index.query(
        namespace=DEDUPLICATION_NAMESPACE,
        vector=embedding,
        top_k=DEDUPLICATION_TOP_K,
        include_metadata=False,
    )

    for match in query_response.matches:
        if match.score >= SIMILARITY_THRESHOLD:  # type: ignore[attr-defined]
            logger.info(
                "Found duplicate '%s' – similarity %.2f", metadata.get("title"), match.score
            )
            return True

    logger.info("No duplicates found for: %s", metadata.get("title"))
    return False

__all__ = ["check_duplicates"] 