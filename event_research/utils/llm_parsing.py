"""Utilities for parsing structured outputs returned by LLM calls.

Currently only contains helpers for extracting the JSON payload that
follows a Perplexity `<think>` block.  The implementation is shared
between different services so we keep it in one place.
"""

from __future__ import annotations

import json
import re
from typing import Any, Dict

from .text_cleaning import strip_think_blocks

__all__ = ["extract_structured_json"]


def extract_structured_json(response_text: str) -> Dict[str, Any]:
    """Robustly extract JSON from an LLM response.

    Parameters
    ----------
    response_text
        The raw message content returned by the Perplexity API.

    Returns
    -------
    dict[str, Any]
        The parsed JSON object.  When the top-level parsed value is a list
        (i.e. the LLM directly returned the value of an *array* field), it
        is wrapped into ``{"events": <list>}`` so that downstream code can
        always rely on accessing the ``"events"`` key.

    Raises
    ------
    ValueError
        If no valid JSON snippet can be located in *response_text*.
    """

    cleaned: str = strip_think_blocks(response_text).strip()

    # 1. Try to parse the whole string first (fast path)
    try:
        parsed = json.loads(cleaned)
        if isinstance(parsed, list):
            return {"events": parsed}
        return parsed  # type: ignore[return-value]
    except json.JSONDecodeError:
        pass

    # 2. Search for fenced JSON block, with or without explicit `json` label
    fenced = re.search(
        r"```(?:json)?\s*([\[{].*?[\]}])\s*```",
        cleaned,
        flags=re.DOTALL | re.IGNORECASE,
    )
    if fenced:
        snippet = fenced.group(1).strip()
        try:
            parsed = json.loads(snippet)
            if isinstance(parsed, list):
                return {"events": parsed}
            return parsed  # type: ignore[return-value]
        except json.JSONDecodeError:
            cleaned = snippet  # Narrow search space.

    # 3. Progressive truncation from first {{ or [
    first_curly = cleaned.find("{")
    first_bracket = cleaned.find("[")
    starts = [i for i in (first_curly, first_bracket) if i != -1]
    if not starts:
        raise ValueError("Could not locate JSON in Perplexity response")

    candidate = cleaned[min(starts):]

    for end in range(len(candidate), 0, -1):
        snippet = candidate[:end].strip()
        try:
            parsed = json.loads(snippet)
            if isinstance(parsed, list):
                return {"events": parsed}
            return parsed  # type: ignore[return-value]
        except json.JSONDecodeError:
            continue

    raise ValueError("Could not locate JSON in Perplexity response") 