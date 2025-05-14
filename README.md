# Timeline Researcher

A tool that automatically researches current global events, deduplicates them, and stores them in both Pinecone (for vector search) and MongoDB (for persistent storage).

## Features

- Searches for top 5 global events of the day using Perplexity API
- Generates embeddings for each event using OpenAI
- Deduplicates events using Pinecone similarity search
- Researches additional details for each unique event
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

Run the script to start the event research workflow:

```bash
python timeline_researcher.py
```

The script will:
1. Search for top global events using Perplexity API
2. Generate embeddings and check for duplicates in Pinecone
3. Research additional details for unique events
4. Store events in both Pinecone and MongoDB

## Project Structure

- `timeline_researcher.py`: Main script that implements the workflow
- `requirements.txt`: Python dependencies
- `.env`: Configuration file for API keys (not included in repository)

## Notes

- The script uses current date filters to ensure recent events
- Duplicate detection uses a similarity threshold of 0.8
- All steps are logged for monitoring and debugging 