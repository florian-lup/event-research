# Event Research Pipeline

A comprehensive pipeline for discovering, investigating, deduplicating, and storing significant events. This project uses AI-powered research tools and vector databases to maintain a curated collection of events with their details and contexts.

## Features

- **Discovery**: Automatically searches for significant events from various sources
- **Deduplication**: Uses vector similarity to prevent duplicate events from being processed
- **Investigation**: Enriches event data with additional context and details
- **Storage**: Stores processed events in both MongoDB Atlas and Pinecone database

## Installation

1. Clone the repository
```bash
git clone https://github.com/florian-lup/event-research.git
cd event-research
```

2. Create and activate a virtual environment
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies
```bash
pip install -r requirements.txt
```

4. Create a `.env` file in the project root directory with your API keys:
```
PERPLEXITY_API_KEY=your_perplexity_key
OPENAI_API_KEY=your_openai_key
TAVILY_API_KEY=your_tavily_key
PINECONE_API_KEY=your_pinecone_key
PINECONE_ENVIRONMENT=your_pinecone_environment
MONGODB_URI=your_mongodb_connection_string
```

## Usage

Run the entire pipeline:
```bash
python -m event_research
```

Or import and use in your own Python code:
```python
from event_research import run
run()
```

## Project Structure

- `event_research/`: Main package
  - `clients/`: API clients for various services
  - `models/`: Data models
  - `services/`: Core services for different parts of the pipeline
  - `utils/`: Utility functions and helpers
  - `workflows/`: End-to-end workflows
- `main.py`: Entry point for deployment platforms
- `tests/` – unit tests for functions and workflows
- `requirements.txt` – pinned dependencies
- `.env.example` – environment variables



## Dependencies

- Perplexity (for event search and discovery)
- OpenAI (for embeddings and research)
- Pinecone (for vector storage and similarity search)
- MongoDB (for persistent document storage)
- Tavily (for web search capabilities)
- Additional utilities (see `requirements.txt`)