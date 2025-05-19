"""CLI entry-point for `python -m event_research`."""

from .workflows.event_pipeline import run

if __name__ == "__main__":
    run() 