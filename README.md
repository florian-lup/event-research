# Timeline Researcher

A tool that automatically researches current global events, deduplicates them, and stores them in both Pinecone (for vector search) and MongoDB (for persistent storage).

## Features

- Searches for top 5 global events of the day using Perplexity API
- Generates embeddings for each event using OpenAI
- Deduplicates events using Pinecone similarity search
- Researches additional details for each unique event with gpt and tavily
- Stores events in Pinecone for vector search and MongoDB for persistent storage

## Requirements

- Python 3.8+
- Perplexity API key
- OpenAI API key
- Pinecone API key
- MongoDB connection string

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/timeline-researcher.git
cd timeline-researcher
```

2. Install the dependencies:
```bash
pip install -r requirements.txt
```

3. Create a `.env` file with your API keys:
```
PERPLEXITY_API_KEY=your_perplexity_api_key
OPENAI_API_KEY=your_openai_api_key
PINECONE_API_KEY=your_pinecone_api_key
PINECONE_ENVIRONMENT=your_pinecone_environment
MONGODB_URI=your_mongodb_uri
```

## Usage

Run the pipeline via the package's entry-point:

```bash
python -m event_research
```

The command will:
1. Discover top global events using the Perplexity API.
2. Embed each headline+summary and deduplicate them with Pinecone.
3. Enrich unique events with detailed analysis and sources.
4. Persist vectors to Pinecone and full documents to MongoDB.

## Project Structure (simplified)

- `event_research/`
  - `__init__.py` – exposes `run()` helper.
  - `__main__.py` – enables `python -m event_research`.
  - `pipeline.py` – <80-line orchestration of the workflow.
  - `config.py`, `logging_config.py` – configuration & logging.
  - `clients/` – singleton wrappers for OpenAI, Pinecone, MongoDB, Tavily, Perplexity.
  - `services/` – discovery, embeddings, deduplication, enrichment, storage.
  - `models.py` – dataclass `Event`.
  - `utils/` – shared helpers (e.g. text cleaning).
- `tests/` – unit tests for utilities and models.
- `requirements.txt` – pinned dependencies.
- `.env.example` – template for environment variables (real `.env` ignored).

## Notes

- The script uses current date filters to ensure recent events
- Duplicate detection uses a similarity threshold of 0.8
- All steps are logged for monitoring and debugging 