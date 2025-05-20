"""Utility functions for working with dates and times."""

from datetime import datetime, timezone

__all__ = [
    "get_current_timestamp",
]

def get_current_timestamp() -> datetime:
    """Return the current UTC datetime with micro-second precision.

    This object can be stored directly in MongoDB where it will be written as
    a BSON Date. When you need a human-readable version, pass the datetime
    to ``format_timestamp``.
    """
    return datetime.now(tz=timezone.utc)