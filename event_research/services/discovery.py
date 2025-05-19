"""Event discovery via Perplexity API."""

from __future__ import annotations

import json
import logging
import re
from typing import List, Dict, Any

from ..clients.perplexity_client import get_perplexity_session
from ..config import (
    CURRENT_DATE,
    PERPLEXITY_API_KEY,
    PERPLEXITY_MODEL,
    PERPLEXITY_CONTEXT_SIZE,
)
from ..utils.text_cleaning import strip_think_blocks

logger = logging.getLogger(__name__)


def _extract_json_from_response(response_text: str) -> Dict[str, Any]:
    """Best-effort extraction of JSON from a Perplexity response string."""
    # Remove hidden "think" blocks first
    response_text = strip_think_blocks(response_text)

    # Try fenced block first
    json_match = re.search(r"```json\s*(.*?)\s*```", response_text, re.DOTALL)

    if not json_match:
        # Fallback to generic braces capture
        json_match = re.search(r"(\{.*\})", response_text, re.DOTALL)
        if not json_match:
            raise ValueError("Could not locate JSON in Perplexity response")

    json_text = json_match.group(1).strip()
    return json.loads(json_text)


def search_events_with_perplexity() -> List[Dict[str, Any]]:
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
                    "   • summary – 400-600 characters explaining what happened, why it matters, "
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
        cleaned_summary = re.sub(r"\[\d+\]", "", event["summary"])  # strip citations

        complete_events.append(
            {
                "date": CURRENT_DATE,
                "title": event["title"],
                "summary": cleaned_summary,
                "report": "",
                "sources": [],
            }
        )

    return complete_events

__all__ = ["search_events_with_perplexity"] 