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
# Cross-cutting service settings
# (referenced in more than one component)
# ---------------------------------------------------------------------------
PINECONE_INDEX_NAME: str = "events"
DEDUPLICATION_NAMESPACE: str = "deduplication"
research_NAMESPACE: str = "research"

# ---------------------------------------------------------------------------
# Miscellaneous
# Perplexity Date format for date filtering
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
    # shared
    "PINECONE_INDEX_NAME",
    "DEDUPLICATION_NAMESPACE",
    "research_NAMESPACE",
    # misc
    "CURRENT_DATE",
] 