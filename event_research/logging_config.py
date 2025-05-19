"""Centralised logging configuration.

Importing this module sets the default logging format/level. Other modules
should simply import `logging` and call `logging.getLogger(__name__)`.
"""

import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)

__all__ = ["logging"] 