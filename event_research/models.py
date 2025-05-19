"""Domain models used across the project."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import List

# Type alias for 3072-dimensional embedding vector
Embedding = List[float]


@dataclass(slots=True)
class Event:
    """A significant global event and its associated metadata."""

    date: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    title: str = ""
    summary: str = ""
    report: str = ""
    sources: List[str] = field(default_factory=list)

    def overview_text(self) -> str:
        """Return the concatenation of title + summary used for embeddings."""
        return f"{self.title} {self.summary}"

__all__ = ["Event", "Embedding"] 