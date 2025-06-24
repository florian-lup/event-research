"""Entry point for deployment platforms that expect main.py"""
from workflows.event_pipeline import run

if __name__ == "__main__":
    run() 