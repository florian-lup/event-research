"""Definition of the `Event` dataclass used throughout the project."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List

from ..config import CURRENT_DATE

# Type alias for embedding vectors
Embedding = List[float]


@dataclass(slots=True)
class Event:
    """A significant global event and its associated metadata."""

    date: str = field(default_factory=lambda: CURRENT_DATE)
    title: str = ""
    summary: str = ""
    report: str = ""
    sources: List[str] = field(default_factory=list)

    def overview_text(self) -> str:
        """Return the concatenation of title and summary for embedding."""
        return f"{self.title} {self.summary}"

__all__ = ["Event", "Embedding"]
