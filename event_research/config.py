"""Centralised configuration for event_research.

Environment variables are loaded once and all related constants are
grouped by service for easier maintenance.
"""

from __future__ import annotations

import os
from datetime import datetime
from dotenv import load_dotenv

# ---------------------------------------------------------------------------
# Load environment variables from `.env` (if present)
# ---------------------------------------------------------------------------
load_dotenv()

# ---------------------------------------------------------------------------
# Core credentials (from environment)
# ---------------------------------------------------------------------------
PERPLEXITY_API_KEY: str | None = os.getenv("PERPLEXITY_API_KEY")
OPENAI_API_KEY: str | None = os.getenv("OPENAI_API_KEY")
TAVILY_API_KEY: str | None = os.getenv("TAVILY_API_KEY")
PINECONE_API_KEY: str | None = os.getenv("PINECONE_API_KEY")
PINECONE_ENVIRONMENT: str | None = os.getenv("PINECONE_ENVIRONMENT")
MONGODB_URI: str | None = os.getenv("MONGODB_URI")

# ---------------------------------------------------------------------------
# Perplexity settings
# ---------------------------------------------------------------------------
PERPLEXITY_MODEL: str = "sonar-reasoning"
# accepted values: "low", "medium", "large"
PERPLEXITY_CONTEXT_SIZE: str = "low"

# ---------------------------------------------------------------------------
# OpenAI settings
# ---------------------------------------------------------------------------
OPENAI_SEARCH_MODEL: str = "gpt-4o-mini"  # short prompt, quick response
OPENAI_ARTICLE_MODEL: str = "gpt-4o"      # long-form synthesis

# ---------------------------------------------------------------------------
# Tavily settings
# ---------------------------------------------------------------------------
TAVILY_SEARCH_DEPTH: str = "advanced"
TAVILY_MAX_RESULTS: int = 10
TAVILY_DAYS: int = 1
TAVILY_TIME_RANGE: str = "day"

# ---------------------------------------------------------------------------
# Pinecone settings
# ---------------------------------------------------------------------------
PINECONE_INDEX_NAME: str = "events"
DEDUPLICATION_NAMESPACE: str = "deduplication"
REPORT_NAMESPACE: str = "report"

# ---------------------------------------------------------------------------
# Embedding settings
# ---------------------------------------------------------------------------
EMBEDDING_MODEL: str = "text-embedding-3-large"
EMBEDDING_DIMENSIONS: int = 3072
SIMILARITY_THRESHOLD: float = 0.8

# ---------------------------------------------------------------------------
# Miscellaneous
# ---------------------------------------------------------------------------
CURRENT_DATE: str = datetime.now().strftime("%m/%d/%Y")

# ---------------------------------------------------------------------------
# Re-exported names
# ---------------------------------------------------------------------------
__all__ = [
    # credentials
    "PERPLEXITY_API_KEY",
    "OPENAI_API_KEY",
    "TAVILY_API_KEY",
    "PINECONE_API_KEY",
    "PINECONE_ENVIRONMENT",
    "MONGODB_URI",
    # Perplexity
    "PERPLEXITY_MODEL",
    "PERPLEXITY_CONTEXT_SIZE",
    # OpenAI
    "OPENAI_SEARCH_MODEL",
    "OPENAI_ARTICLE_MODEL",
    # Tavily
    "TAVILY_SEARCH_DEPTH",
    "TAVILY_MAX_RESULTS",
    "TAVILY_DAYS",
    "TAVILY_TIME_RANGE",
    # Pinecone
    "PINECONE_INDEX_NAME",
    "DEDUPLICATION_NAMESPACE",
    "REPORT_NAMESPACE",
    # Embeddings
    "EMBEDDING_MODEL",
    "EMBEDDING_DIMENSIONS",
    "SIMILARITY_THRESHOLD",
    # Misc
    "CURRENT_DATE",
] 