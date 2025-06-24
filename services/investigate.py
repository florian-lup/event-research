"""Event enrichment – detailed research + sources using Tavily and OpenAI."""

from __future__ import annotations

import logging
from typing import Dict, Any, List

from clients import get_openai, get_tavily_client
from config import CURRENT_DATE
from utils import sanitize_llm_text

# ---------------------------------------------------------------------------
# Local OpenAI + Tavily settings (specific to this service)
# ---------------------------------------------------------------------------
OPENAI_SEARCH_MODEL: str = "o4-mini"  # short prompt, quick response
OPENAI_ARTICLE_MODEL: str = "gpt-4.1"      # long-form synthesis

TAVILY_SEARCH_DEPTH: str = "advanced"
TAVILY_MAX_RESULTS: int = 15
TAVILY_DAYS: int = 1
TAVILY_TIME_RANGE: str = "day"

logger = logging.getLogger(__name__)
_openai = get_openai()
_tavily = get_tavily_client()


def _generate_search_query(title: str) -> str:
    """Ask GPT-4o-mini for a concise web-search query."""
    try:
        logger.info("Requesting search query from %s for event: %s", OPENAI_SEARCH_MODEL, title)
        resp = _openai.chat.completions.create(
            model=OPENAI_SEARCH_MODEL,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are an expert news researcher. Given today's"
                        f" {CURRENT_DATE} event headline, output a concise web search query (≤250 characters)"
                        " that will retrieve high-quality, up-to-date coverage about the event."
                    ),
                },
                {"role": "user", "content": title},
            ],
        )
        query = resp.choices[0].message.content.strip().replace("\n", " ")
        logger.debug("Raw %s response: %s", OPENAI_SEARCH_MODEL, resp.model_dump_json())
        return query
    except Exception as exc:  # pragma: no cover – network failure
        logger.warning("GPT-4o query generation failed: %s – falling back to title", exc)
        return title


def investigate_event(event: Dict[str, Any]) -> Dict[str, Any]:
    """Populate `event` with a detailed *research* and *sources* list."""
    logger.info("Researching details for event via Tavily: %s", event["title"])

    # 1) Build search query
    search_query = _generate_search_query(event["title"])
    logger.info("Generated search query for Tavily: %s", search_query)

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
    
    logger.info("Tavily returned %d results for query: %s", len(tavily_results), search_query)

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
        logger.info("Requesting detailed article from %s for event: %s", OPENAI_ARTICLE_MODEL, event['title'])
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
        logger.debug("Raw %s response: %s", OPENAI_ARTICLE_MODEL, article_resp.model_dump_json())
        raw_article: str = article_resp.choices[0].message.content
        cleaned_article = sanitize_llm_text(raw_article, remove_citations=True, remove_markdown=False)
    except Exception as exc:  # pragma: no cover – network failure
        logger.error("GPT-4o article generation failed: %s – using concatenated snippets", exc)
        cleaned_article = aggregated_text

    event["research"] = cleaned_article.strip()
    event["sources"] = unique_sources
    logger.info("Enriched event '%s' with %d sources", event["title"], len(unique_sources))
    return event

__all__ = ["investigate_event"] 