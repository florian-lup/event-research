"""Top-level package for the event-research project.

This package simply exposes the public run() helper so callers can do
`python -m event_research` or `from event_research import pipeline; pipeline.run()`.
"""

from importlib import metadata as _metadata

try:
    __version__: str = _metadata.version(__name__)
except _metadata.PackageNotFoundError:  # pragma: no cover – running from source
    __version__ = "0.0.0"

from .pipeline import run  # convenience re-export

__all__ = ["run", "__version__"] 