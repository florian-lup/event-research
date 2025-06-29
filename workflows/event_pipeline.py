"""End-to-end event discovery, enrichment and persistence pipeline."""

from __future__ import annotations

import logging

from logging_config import logging as _  # noqa: F401  # ensure config applied early
from clients import get_pinecone_index
from services import (
    search_events,
    generate_embedding, 
    check_duplicates,
    investigate_event,
    upsert_to_pinecone,
    store_to_mongodb
)

logger = logging.getLogger(__name__)


def run() -> None:
    """Execute the full pipeline once."""
    logger.info("Starting event research workflow")

    # 1. Discover events
    events = search_events()
    total_events_found = len(events)

    # 2. Connect to Pinecone index (must already exist)
    pinecone_index = get_pinecone_index()

    unique_events = []
    duplicate_count = 0
    for event in events:
        combined_text = f"{event['title']} {event['summary']}"
        embedding = generate_embedding(combined_text)

        if not check_duplicates(
            pinecone_index, embedding, {"title": event["title"], "summary": event["summary"]}
        ):
            detailed_event = investigate_event(event)
            unique_events.append((detailed_event, embedding))
        else:
            duplicate_count += 1

    if not unique_events:
        logger.info("All events are duplicates – nothing to process.")
        _log_stats(total_events_found, duplicate_count, 0)
        return

    # 3. Store unique events
    stored_count = 0
    for event, embedding in unique_events:
        upsert_to_pinecone(pinecone_index, event, embedding)
        store_to_mongodb(event)
        stored_count += 1

    _log_stats(total_events_found, duplicate_count, stored_count)


def _log_stats(total_events: int, duplicates: int, stored: int) -> None:
    logger.info("=== Timeline Researcher Statistics ===")
    logger.info("Total events found: %d", total_events)
    logger.info("Duplicate events: %d", duplicates)
    logger.info("Events stored in database: %d", stored)
    logger.info("=====================================")

__all__ = ["run"]
