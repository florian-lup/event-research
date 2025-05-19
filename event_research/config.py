"""Configuration constants and helpers.

All environment variables are loaded once at import time. Any missing
critical variables raise a RuntimeError early so the program fails fast.
"""

from __future__ import annotations

import os
from datetime import datetime
from dotenv import load_dotenv

# ---------------------------------------------------------------------------
# Load .env
# ---------------------------------------------------------------------------
load_dotenv()

# ---------------------------------------------------------------------------
# External service credentials
# ---------------------------------------------------------------------------
PERPLEXITY_API_KEY: str | None = os.getenv("PERPLEXITY_API_KEY")
OPENAI_API_KEY: str | None = os.getenv("OPENAI_API_KEY")
PINECONE_API_KEY: str | None = os.getenv("PINECONE_API_KEY")
PINECONE_ENVIRONMENT: str | None = os.getenv("PINECONE_ENVIRONMENT")
MONGODB_URI: str | None = os.getenv("MONGODB_URI")
TAVILY_API_KEY: str | None = os.getenv("TAVILY_API_KEY")

# ---------------------------------------------------------------------------
# Pinecone configuration
# ---------------------------------------------------------------------------
PINECONE_INDEX_NAME: str = "events"
DEDUPLICATION_NAMESPACE: str = "deduplication"
REPORT_NAMESPACE: str = "report"

# ---------------------------------------------------------------------------
# Embedding model configuration
# ---------------------------------------------------------------------------
EMBEDDING_MODEL: str = "text-embedding-3-large"  # 3072-dim
EMBEDDING_DIMENSIONS: int = 3072
SIMILARITY_THRESHOLD: float = 0.8

# ---------------------------------------------------------------------------
# Misc.
# ---------------------------------------------------------------------------
CURRENT_DATE: str = datetime.now().strftime("%m/%d/%Y")

__all__ = [
    "PERPLEXITY_API_KEY",
    "OPENAI_API_KEY",
    "PINECONE_API_KEY",
    "PINECONE_ENVIRONMENT",
    "MONGODB_URI",
    "TAVILY_API_KEY",
    "PINECONE_INDEX_NAME",
    "DEDUPLICATION_NAMESPACE",
    "REPORT_NAMESPACE",
    "EMBEDDING_MODEL",
    "EMBEDDING_DIMENSIONS",
    "SIMILARITY_THRESHOLD",
    "CURRENT_DATE",
] 