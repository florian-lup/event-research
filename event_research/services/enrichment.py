"""Event enrichment – detailed report + sources using Tavily and OpenAI."""

from __future__ import annotations

import logging
from typing import Dict, Any, List

from ..clients.openai_client import get_openai
from ..clients.tavily_client import get_tavily_client
from ..config import (
    CURRENT_DATE,
    OPENAI_SEARCH_MODEL,
    OPENAI_ARTICLE_MODEL,
    TAVILY_SEARCH_DEPTH,
    TAVILY_MAX_RESULTS,
    TAVILY_DAYS,
    TAVILY_TIME_RANGE,
)
from ..utils.text_cleaning import sanitize_llm_text

logger = logging.getLogger(__name__)
_openai = get_openai()
_tavily = get_tavily_client()


def _generate_search_query(title: str) -> str:
    """Ask GPT-4o-mini for a concise web-search query."""
    try:
        resp = _openai.chat.completions.create(
            model=OPENAI_SEARCH_MODEL,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are an expert news researcher. Given today's"
                        f" {CURRENT_DATE} event headline, output a concise web search query (≤120 characters)"
                        " that will retrieve high-quality, up-to-date coverage about the event."
                        " Only return the query text—no commentary."
                    ),
                },
                {"role": "user", "content": title},
            ],
            temperature=0.2,
            max_tokens=60,
        )
        return resp.choices[0].message.content.strip().replace("\n", " ")
    except Exception as exc:  # pragma: no cover – network failure
        logger.warning("GPT-4o query generation failed: %s – falling back to title", exc)
        return title


def research_event_details(event: Dict[str, Any]) -> Dict[str, Any]:
    """Populate `event` with a detailed *report* and *sources* list."""
    logger.info("Researching details for event via Tavily: %s", event["title"])

    # 1) Build search query
    search_query = _generate_search_query(event["title"])

    # 2) Tavily search
    tavily_resp = _tavily.search(
        query=search_query,
        topic="news",
        search_depth=TAVILY_SEARCH_DEPTH,
        max_results=TAVILY_MAX_RESULTS,
        days=TAVILY_DAYS,
        time_range=TAVILY_TIME_RANGE,
        include_answer=False,
        include_raw_content=False,
    )
    tavily_results = tavily_resp.get("results", [])

    if not tavily_results:
        logger.warning("Tavily returned no results; skipping detailed enrichment.")
        return event

    # Extract sources and content snippets
    sources: List[str] = []
    content_snippets: List[str] = []
    for res in tavily_results:
        url = res.get("url")
        snippet = res.get("content", "")
        if url:
            sources.append(url)
        if snippet:
            content_snippets.append(snippet)

    # Deduplicate URLs while preserving order
    unique_sources: List[str] = []
    seen = set()
    for url in sources:
        if url not in seen:
            unique_sources.append(url)
            seen.add(url)

    # 3) Generate comprehensive article via GPT-4o
    aggregated_text = "\n\n".join(content_snippets)[:8000]  # truncate defensively
    try:
        article_resp = _openai.chat.completions.create(
            model=OPENAI_ARTICLE_MODEL,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are an award-winning investigative journalist. Write a comprehensive,"
                        " well-structured analysis of the event below, drawing solely from the provided"
                        " source excerpts. Include historical context, current developments, and potential"
                        " future implications."
                    ),
                },
                {
                    "role": "user",
                    "content": f"Event title: {event['title']}\n\nSource excerpts:\n{aggregated_text}",
                },
            ],
            temperature=0.3,
            max_tokens=7000,
        )
        raw_article: str = article_resp.choices[0].message.content
        cleaned_article = sanitize_llm_text(raw_article, remove_citations=True, remove_markdown=False)
    except Exception as exc:  # pragma: no cover – network failure
        logger.error("GPT-4o article generation failed: %s – using concatenated snippets", exc)
        cleaned_article = aggregated_text

    event["report"] = cleaned_article.strip()
    event["sources"] = unique_sources
    logger.info("Enriched event '%s' with %d sources", event["title"], len(unique_sources))
    return event

__all__ = ["research_event_details"] 