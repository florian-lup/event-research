# Event Research Pipeline

A comprehensive pipeline for discovering, investigating, deduplicating, and storing significant events. This project uses AI-powered research tools and vector databases to build and maintain a curated collection of global events with rich context and details.

## Features

- **Discovery**: Leverages the Tavily and Perplexity APIs to automatically search for and identify potentially significant events from a wide range of online sources.
- **Investigation**: Utilizes OpenAI's powerful language models to enrich the discovered events. It generates concise summaries, identifies key people, organizations, and locations involved, and extracts precise dates and times.
- **Deduplication**: Generates vector embeddings for each event's description using OpenAI's models. It then uses Pinecone's vector database to perform similarity searches, effectively identifying and filtering out duplicate events.
- **Storage**: Persists the processed events in a dual-database system. Structured event data (summaries, dates, entities) is stored in MongoDB Atlas for robust querying and retrieval, while the corresponding vector embeddings are stored in Pinecone for efficient real-time similarity searches.

## Technologies Used

- **AI & LLM Services**:
  - OpenAI: For data enrichment, summarization, and generating vector embeddings.
  - Perplexity AI: For AI-powered search and discovery.
  - Tavily: For AI-powered research and discovery.
- **Databases**:
  - MongoDB Atlas: For primary storage of structured event data.
  - Pinecone: For storing vector embeddings and performing similarity searches.
- **Core Python Libraries**:
  - `requests` & `httpx`: For making HTTP requests to external APIs.
  - `pymongo`: The official Python driver for MongoDB.
  - `pinecone-client`: The official client for interacting with Pinecone.
  - `openai`: The official Python library for the OpenAI API.
  - `python-dotenv`: For managing environment variables.
  - `tavily-python`: The official client for the Tavily API.

## Workflow

The entire process is orchestrated by the `event_pipeline.py` workflow, which seamlessly connects the different services:

1.  **Discover**: New events are found.
2.  **Investigate**: Events are enriched with details.
3.  **Deduplicate**: Redundant events are filtered out.
4.  **Store**: Clean, unique events are saved to the databases.

## Getting Started

1.  **Clone the repository:**
    ```bash
    git clone <repository-url>
    cd timeline-journalist
    ```
2.  **Set up a virtual environment:**
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
    ```
3.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```
4.  **Configure environment variables:**
    Create a `.env` file in the root directory and add the necessary API keys and connection strings for the services used (OpenAI, MongoDB, Pinecone, etc.).

5.  **Run the pipeline:**
    ```bash
    python main.py
    ```
