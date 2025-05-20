"""Shared helper utilities used across services."""

from __future__ import annotations

import re
from typing import Final, Any

# ---------------------------------------------------------------------------
# Public helpers
# ---------------------------------------------------------------------------

def strip_think_blocks(text: str) -> str:
    """Extract content after the closing </think> tag from an LLM response.

    Handles missing tags and safely removes JSON code fences if present.
    """
    if not text:
        return text.strip()

    marker: Final[str] = "</think>"
    idx: int = text.rfind(marker)

    # Fallback to full text if marker is missing
    after: str = text if idx == -1 else text[idx + len(marker) :]

    cleaned: str = after.strip()

    # Remove JSON code fences if present
    if cleaned.startswith("```json"):
        cleaned = cleaned[len("```json") :].strip()
    if cleaned.startswith("```"):
        cleaned = cleaned[3:].strip()
    if cleaned.endswith("```"):
        cleaned = cleaned[:-3].strip()

    return cleaned

def sanitize_llm_text(
    text: str,
    *,
    remove_citations: bool = True,
    remove_markdown: bool = False,
    **_: Any,
) -> str:
    """Standardise LLM output for downstream parsing.

    Parameters
    ----------
    text : str
        Raw LLM response.
    remove_citations : bool, default True
        Remove numeric (``[1]``) and textual (``[Reuters]``) citations.
    remove_markdown : bool, default False
        Strip headings, lists, emphasis, and horizontal rules.
    """
    cleaned: str = strip_think_blocks(text)

    if remove_citations:
        # Remove numeric citations ([1], [12])
        cleaned = re.sub(r"\[\d+\]", "", cleaned)
        # Remove textual citations ([Reuters], [NYT])
        cleaned = re.sub(r"\[[A-Za-z][^\]]+\]", "", cleaned)

    if remove_markdown:
        # Strip headings (# Heading, ## Heading)
        cleaned = re.sub(r"^#{1,6}\s*", "", cleaned, flags=re.MULTILINE)
        # Strip unordered list bullets (-, *, +)
        cleaned = re.sub(r"^\s*[-*+]\s+", "", cleaned, flags=re.MULTILINE)
        # Strip numbered list markers (1. 2. etc.)
        cleaned = re.sub(r"^\s*\d+\.\s+", "", cleaned, flags=re.MULTILINE)
        # Remove horizontal rules (--- or *** lines)
        cleaned = re.sub(r"^(?:-{3,}|\*{3,})$", "", cleaned, flags=re.MULTILINE)
        # Remove emphasis (**bold**, *italic*, __bold__, _italic_)
        cleaned = re.sub(r"(\*\*|__|\*|_)", "", cleaned)

    return cleaned.strip()

__all__ = ["strip_think_blocks", "sanitize_llm_text"] 