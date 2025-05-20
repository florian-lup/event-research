"""Embedding utilities using the OpenAI API."""

from __future__ import annotations

import logging
from typing import Iterable, List

from ..clients.openai_client import get_openai

# ---------------------------------------------------------------------------
# Local embedding settings
# ---------------------------------------------------------------------------
EMBEDDING_MODEL: str = "text-embedding-3-large"
EMBEDDING_DIMENSIONS: int = 3072

logger = logging.getLogger(__name__)
_openai = get_openai()


def generate_embedding(text: str) -> List[float]:
    """Generate a vector embedding for *text* using the configured model."""
    logger.info("Generating embedding for text (first 50 chars): %s…", text[:50])
    response = _openai.embeddings.create(
        model=EMBEDDING_MODEL,
        input=text,
        dimensions=EMBEDDING_DIMENSIONS,
    )
    embedding = response.data[0].embedding
    logger.debug("Generated embedding of length %d", len(embedding))
    return embedding


def chunk_text(text: str, max_tokens: int = 300) -> Iterable[str]:
    """Split *text* into ~token-sized chunks using simple word slicing."""
    if not text:
        return []
    words = text.split()
    for i in range(0, len(words), max_tokens):
        yield " ".join(words[i : i + max_tokens])

__all__ = ["generate_embedding", "chunk_text"] 