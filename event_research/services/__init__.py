"""Service layer modules grouping business logic by concern.

This module provides convenience re-exports so that callers can simply do for
example `from event_research.services import search_events` without having to
know which underlying module provides the symbol.
"""

from .discovery import search_events  # noqa: F401
from .research import research_event  # noqa: F401
from .deduplication import check_duplicates  # noqa: F401
from .embeddings import generate_embedding, chunk_text  # noqa: F401
from .storage import upsert_to_pinecone, store_to_mongodb  # noqa: F401

__all__ = [
    "search_events",
    "research_event",
    "check_duplicates",
    "generate_embedding",
    "chunk_text",
    "upsert_to_pinecone",
    "store_to_mongodb",
] 