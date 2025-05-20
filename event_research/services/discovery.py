"""Event discovery via Perplexity API."""

from __future__ import annotations

import logging
from typing import List, Dict, Any

from ..clients.perplexity_client import get_perplexity_session
from ..config import (
    CURRENT_DATE,
    PERPLEXITY_API_KEY,
)
from ..utils.datetime_utils import get_current_timestamp

# ---------------------------------------------------------------------------
# Local Perplexity settings (only used by this service)
# ---------------------------------------------------------------------------
PERPLEXITY_MODEL: str = "sonar-reasoning-pro"
# accepted values: "low", "medium", "high"
PERPLEXITY_CONTEXT_SIZE: str = "high"

from ..utils.text_cleaning import strip_think_blocks, sanitize_llm_text
from ..utils.llm_parsing import extract_structured_json

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Local helpers (thin wrappers around shared utils)
# ---------------------------------------------------------------------------

def _extract_json_from_response(response_text: str) -> Dict[str, Any]:
    """Wrapper invoking the shared LLM-parsing utility."""
    cleaned = strip_think_blocks(response_text)
    return extract_structured_json(cleaned)

def search_events() -> List[Dict[str, Any]]:
    """Query Perplexity for the five most significant global events today."""
    logger.info("Searching for top 5 global events with Perplexity API…")

    headers = {
        "Authorization": f"Bearer {PERPLEXITY_API_KEY}",
        "Content-Type": "application/json",
    }

    data = {
        "model": PERPLEXITY_MODEL,
        "messages": [
            {
                "role": "system",
                "content": (
                    "You are a precise research assistant specialised in real-time global"
                    " news extraction. Strictly follow the user instructions and output"
                    " EXACTLY the JSON that matches the provided schema, no markdown,"
                    " no fences, no commentary, no citations."
                ),
            },
            {
                "role": "user",
                "content": (
                    f"Identify the five most significant global events that occurred on {CURRENT_DATE}.\n"
                    "Requirements:\n"
                    "1) Use reputable international sources published within the last 24 hours.\n"
                    "2) Cover diverse topics and geographic regions (e.g. politics, economy, science, "
                    "technology, environment, health, conflict, culture).\n"
                    "3) Return an array named 'events', where each item contains:\n"
                    "   • title – concise, ≤90 characters, written as a compelling headline.\n"
                    "   • summary – ~500 characters explaining what happened, why it matters, "
                    "and key details.\n"
                ),
            },
        ],
        "search_after_date_filter": CURRENT_DATE,
        "search_before_date_filter": CURRENT_DATE,
        "web_search_options": {"search_context_size": PERPLEXITY_CONTEXT_SIZE},
        "response_format": {
            "type": "json_schema",
            "json_schema": {
                "schema": {
                    "type": "object",
                    "properties": {
                        "events": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "title": {"type": "string"},
                                    "summary": {"type": "string"},
                                },
                                "required": ["title", "summary"],
                            },
                        }
                    },
                    "required": ["events"],
                }
            },
        },
    }

    response = get_perplexity_session().post(
        "https://api.perplexity.ai/chat/completions", headers=headers, json=data
    )

    if response.status_code != 200:
        logger.error(
            "Error from Perplexity API: %s - %s", response.status_code, response.text
        )
        raise RuntimeError(f"Perplexity API error: {response.status_code}")

    response_text: str = response.json()["choices"][0]["message"]["content"]
    logger.debug("Raw Perplexity response: %s", response_text)

    events_data = _extract_json_from_response(response_text)
    events = events_data.get("events", [])

    logger.info("Found %d candidate events", len(events))

    # Build complete event objects with metadata fields
    complete_events: List[Dict[str, Any]] = []
    for event in events:
        cleaned_summary = sanitize_llm_text(event["summary"], title=event["title"], remove_citations=True, remove_markdown=True)

        complete_events.append(
            {
                "date": get_current_timestamp(),
                "title": event["title"],
                "summary": cleaned_summary,
                "research": "",
                "sources": [],
            }
        )

    return complete_events

__all__ = ["search_events"] 