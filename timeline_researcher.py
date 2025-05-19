#!/usr/bin/env python3
import os
import json
import logging
import datetime
import re
from datetime import datetime
from dotenv import load_dotenv
import requests
from pinecone import Pinecone
from pymongo import MongoClient
from tavily import TavilyClient

# Configure logging with StreamHandler
logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler()
    ]
)

# Load environment variables from .env file
load_dotenv()

# API credentials from environment variables
PERPLEXITY_API_KEY = os.getenv('PERPLEXITY_API_KEY')
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
PINECONE_API_KEY = os.getenv('PINECONE_API_KEY')
PINECONE_ENVIRONMENT = os.getenv('PINECONE_ENVIRONMENT')
MONGODB_URI = os.getenv('MONGODB_URI')
TAVILY_API_KEY = os.getenv('TAVILY_API_KEY')

# Initialize service clients
logger.info("Initializing API clients...")
pc = Pinecone(api_key=PINECONE_API_KEY)
logger.info("Initialized Pinecone client")
mongo_client = MongoClient(MONGODB_URI)
logger.info("Initialized MongoDB client")

# Initialize OpenAI client (singleton pattern)
from openai import OpenAI as _OpenAIClient
openai_client = _OpenAIClient(api_key=OPENAI_API_KEY)
logger.info("Initialized OpenAI client")

# Initialize Tavily client (singleton)
tavily_client = TavilyClient(api_key=TAVILY_API_KEY)
logger.info("Initialized Tavily client")

# Shared HTTP session for Perplexity API requests
perplexity_session = requests.Session()

# Lazy-initialized Pinecone index
pinecone_index = None  # type: ignore

# Configuration constants
PINECONE_INDEX_NAME = 'events'
EMBEDDING_MODEL = 'text-embedding-3-large'  # 3072-dim embeddings for improved semantic quality
EMBEDDING_DIMENSIONS = 3072
SIMILARITY_THRESHOLD = 0.8
CURRENT_DATE = datetime.now().strftime("%m/%d/%Y")

# Pinecone namespace organization
DEDUPLICATION_NAMESPACE = 'deduplication'  # One vector per event (title + summary)
REPORT_NAMESPACE = 'report'    # Many vectors per event (chunked report)

def _strip_think_blocks(text: str) -> str:
    """Extract content after the closing </think> tag from LLM response.
    
    Handles missing tags and removes JSON code fences if present.
    """
    if not text:
        return text.strip()

    marker = "</think>"
    idx = text.rfind(marker)

    # Fallback to full text if marker is missing
    after = text if idx == -1 else text[idx + len(marker):]

    cleaned = after.strip()

    # Remove JSON code fences if present
    if cleaned.startswith("```json"):
        cleaned = cleaned[len("```json"):].strip()
    if cleaned.startswith("```"):
        cleaned = cleaned[3:].strip()
    if cleaned.endswith("```"):
        cleaned = cleaned[:-3].strip()

    return cleaned

def _sanitize_llm_text(text: str, *, remove_citations: bool = True, remove_markdown: bool = False) -> str:
    """Standardize LLM output for downstream parsing.

    Parameters
    ----------
    text : str
        Raw LLM response
    remove_citations : bool, default True
        Remove numeric ([1]) and textual ([Reuters]) citations
    remove_markdown : bool, default False
        Strip headings, lists, emphasis, and horizontal rules

    Returns
    -------
    str
        Cleaned text for parsing or storage
    """
    cleaned = _strip_think_blocks(text)

    if remove_citations:
        # Remove numeric citations ([1], [12])
        cleaned = re.sub(r"\[\d+\]", "", cleaned)
        # Remove textual citations ([Reuters], [NYT])
        cleaned = re.sub(r"\[[A-Za-z][^\]]+\]", "", cleaned)

    if remove_markdown:
        # Strip headings (# Heading, ## Heading)
        cleaned = re.sub(r"^#{1,6}\s*", "", cleaned, flags=re.MULTILINE)
        # Strip unordered list bullets (-, *, +)
        cleaned = re.sub(r"^\s*[-*+]\s+", "", cleaned, flags=re.MULTILINE)
        # Strip numbered list markers (1. 2. etc.)
        cleaned = re.sub(r"^\s*\d+\.\s+", "", cleaned, flags=re.MULTILINE)
        # Remove horizontal rules (--- or *** lines)
        cleaned = re.sub(r"^(?:-{3,}|\*{3,})$", "", cleaned, flags=re.MULTILINE)
        # Remove emphasis (**bold**, *italic*, __bold__, _italic_)
        cleaned = re.sub(r"(\*\*|__|\*|_)", "", cleaned)

    return cleaned.strip()

def search_events_with_perplexity():
    """Query Perplexity API for current significant global events.
    
    Returns a list of event objects containing title and summary fields.
    Each event is enriched with placeholders for content and sources.
    """
    logger.info("Searching for top 5 global events with Perplexity API...")
    
    headers = {
        "Authorization": f"Bearer {PERPLEXITY_API_KEY}",
        "Content-Type": "application/json"
    }
    
    data = {
        "model": "sonar-reasoning",
        "messages": [
            {
                "role": "system",
                "content": (
                    "You are a precise research assistant specialised in real-time global news extraction. "
                    "Strictly follow the user instructions and output EXACTLY the JSON that matches the provided schema, no markdown, no fences, no commentary, no citations."
                )
            },
            {
                "role": "user",
                "content": (
                    f"Identify the five most significant global events that occurred on {CURRENT_DATE}.\n"
                    "Requirements:\n"
                    "1) Use reputable international sources published within the last 24 hours.\n"
                    "2) Cover diverse topics and geographic regions (e.g. politics, economy, science, technology, environment, health, conflict, culture).\n"
                    "3) Return an array named 'events', where each item contains: \n"
                    "   • title – concise, ≤90 characters, written as a compelling headline.\n"
                    "   • summary – 400-600 characters explaining what happened, why it matters, and key details.\n"
                )
            }
        ],
        "search_after_date_filter": CURRENT_DATE,
        "search_before_date_filter": CURRENT_DATE,
        "web_search_options": {"search_context_size": "medium"},
        "response_format": {
            "type": "json_schema",
            "json_schema": {
                "schema": {
                    "type": "object",
                    "properties": {
                        "events": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "title": {"type": "string"},
                                    "summary": {"type": "string"}
                                },
                                "required": ["title", "summary"]
                            }
                        }
                    },
                    "required": ["events"]
                }
            }
        }
    }
    
    response = perplexity_session.post(
        "https://api.perplexity.ai/chat/completions",
        headers=headers,
        json=data
    )
    
    if response.status_code != 200:
        logger.error(f"Error from Perplexity API: {response.status_code} - {response.text}")
        raise Exception(f"Perplexity API error: {response.status_code}")
    
    # Extract JSON from response
    response_text = response.json()['choices'][0]['message']['content']
    logger.info(f"Response from Perplexity: {response_text[:100]}...")
    
    # Clean response and extract JSON
    response_text = _strip_think_blocks(response_text)
    
    # Extract JSON from possible formats (with or without code fences)
    json_match = re.search(r'```json\s*(.*?)\s*```', response_text, re.DOTALL)
    
    if not json_match:
        # Fallback to generic JSON pattern
        json_match = re.search(r'(\{.*\})', response_text, re.DOTALL)
        if not json_match:
            logger.error(f"Could not parse JSON from Perplexity response: {response_text}")
            raise Exception("Failed to parse JSON from Perplexity response")
    
    json_text = json_match.group(1).strip()
    try:
        events_data = json.loads(json_text)
        events = events_data.get('events', [])
    except json.JSONDecodeError as e:
        logger.error(f"JSON parsing error: {e}. Text: {json_text}")
        raise Exception(f"Failed to parse JSON returned by Perplexity: {e}")
    
    # Build complete event objects with metadata fields
    complete_events = []
    for event in events:
        # Remove citation markers from summary
        cleaned_summary = re.sub(r'\[\d+\]', '', event['summary'])
        
        complete_events.append({
            'date': datetime.now().isoformat(),
            'title': event['title'],
            'summary': cleaned_summary,
            'report': '',
            'sources': []
        })
    
    logger.info(f"Found {len(complete_events)} events with Perplexity API")
    return complete_events

def generate_embedding(text):
    """Generate vector embedding using OpenAI API.
    
    Converts text to EMBEDDING_DIMENSIONS-dimensional vector using EMBEDDING_MODEL.
    """
    logger.info(f"Generating embedding for text: {text[:50]}...")

    try:
        response = openai_client.embeddings.create(
            model=EMBEDDING_MODEL,
            input=text,
            dimensions=EMBEDDING_DIMENSIONS,
        )
        embedding = response.data[0].embedding
        logger.info(f"Generated embedding successfully (dim={len(embedding)})")
        return embedding
    except Exception as e:
        logger.error(f"Error generating embedding: {e}")
        raise

def check_duplicate_in_pinecone(pinecone_index, embedding, metadata):
    """Detect semantic duplicates using vector similarity.
    
    Returns True if any existing vector exceeds SIMILARITY_THRESHOLD.
    """
    logger.info(f"Checking for duplicates for: {metadata['title']}")
    
    query_response = pinecone_index.query(
        namespace=DEDUPLICATION_NAMESPACE,
        vector=embedding,
        top_k=5,
        include_metadata=True
    )
    
    # Check similarity scores against threshold
    for match in query_response.matches:
        if match.score >= SIMILARITY_THRESHOLD:
            logger.info(f"Found duplicate: {metadata['title']} - Similarity: {match.score}")
            return True
    
    logger.info(f"No duplicates found for: {metadata['title']}")
    return False

def research_event_details(event):
    """Enrich event with detailed content and sources.
    
    1. Generates optimized search query via GPT-4o-mini
    2. Retrieves relevant news via Tavily
    3. Creates comprehensive analysis via GPT-4o
    4. Adds content and source URLs to event object
    """
    logger.info(f"Researching details for event via Tavily: {event['title']}")

    if not TAVILY_API_KEY:
        raise EnvironmentError("TAVILY_API_KEY is not set in environment variables")

    # 1) Generate optimal search query
    try:
        query_resp = openai_client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are an expert news researcher. "
                        f"Given today's {CURRENT_DATE} event headline, output a concise web search query (≤120 characters) "
                        "that will retrieve high-quality, up-to-date coverage about the event. "
                        "Only return the query text—no commentary."
                    ),
                },
                {"role": "user", "content": event["title"]},
            ],
            temperature=0.2,
            max_tokens=60,
        )

        search_query = query_resp.choices[0].message.content.strip().replace("\n", " ")
    except Exception as e:
        # Fallback to title if LLM fails
        logger.warning(f"GPT-4o query generation failed: {e}. Falling back to title.")
        search_query = event["title"]

    logger.info(f"Tavily search query: {search_query}")

    # 2) Retrieve news coverage via Tavily
    try:
        tavily_response = tavily_client.search(
            query=search_query,
            topic="news",
            search_depth="advanced",
            max_results=10,
            days=1,
            time_range="day",
            include_answer=False,
            include_raw_content=False,
        )
        tavily_results = tavily_response.get("results", [])
    except Exception as e:
        logger.error(f"Tavily SDK search failed: {e}")
        raise

    if not tavily_results:
        logger.warning("Tavily returned no results; skipping detailed enrichment.")
        return event

    # Extract sources and content snippets
    sources: list[str] = []
    content_snippets: list[str] = []

    for res in tavily_results:
        url = res.get("url")
        snippet = res.get("content", "")

        if url:
            sources.append(url)
        if snippet:
            content_snippets.append(snippet)

    # Deduplicate URLs while preserving order
    seen_urls = set()
    unique_sources = []
    for url in sources:
        if url not in seen_urls:
            unique_sources.append(url)
            seen_urls.add(url)

    # 3) Generate comprehensive article with GPT-4o
    aggregated_text = "\n\n".join(content_snippets)[:8000]  # truncate defensively

    try:
        article_resp = openai_client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are an award-winning investigative journalist. "
                        "Write a comprehensive, well-structured analysis of the event below, "
                        "drawing solely from the provided source excerpts. "
                        "Include historical context, current developments, and potential future implications."
                    ),
                },
                {
                    "role": "user",
                    "content": (
                        f"Event title: {event['title']}\n\n" +
                        "Source excerpts:\n" + aggregated_text
                    ),
                },
            ],
            temperature=0.3,
            max_tokens=7000,
        )

        raw_article = article_resp.choices[0].message.content
        cleaned_article = _sanitize_llm_text(raw_article, remove_citations=True, remove_markdown=False)
    except Exception as e:
        logger.error(f"GPT-4o article generation failed: {e}. Using concatenated snippets.")
        cleaned_article = aggregated_text

    # 4) Update event with report and sources
    event["report"] = cleaned_article.strip()
    event["sources"] = unique_sources

    logger.info(
        f"Successfully enriched event '{event['title']}' with {len(unique_sources)} sources and detailed analysis."
    )

    return event

def _chunk_text(text: str, max_tokens: int = 300):
    """Split report text into roughly token-sized chunks for vectorization.
    
    Uses simple word-based splitting as token approximation.
    """
    if not text:
        return []
    words = text.split()
    for i in range(0, len(words), max_tokens):
        yield ' '.join(words[i:i + max_tokens])

def upsert_to_pinecone(pinecone_index, event, overview_embedding):
    """Store event vectors in Pinecone index.
    
    1. Stores overview vector (title+summary) in DEDUPLICATION_NAMESPACE
    2. Chunks and stores report vectors in REPORT_NAMESPACE
    """
    logger.info(f"Upserting event (overview + chunks) to Pinecone: {event['title']}")

    # Generate stable ID from title hash
    event_id = f"event_{hash(event['title'])}"

    # 1) Store overview vector
    overview_metadata = {
        'event_id': event_id,
        'title': event['title'],
        'summary': event['summary'],
    }

    pinecone_index.upsert(
        namespace=DEDUPLICATION_NAMESPACE,
        vectors=[
            (event_id, overview_embedding, overview_metadata)
        ]
    )

    # 2) Store report chunk vectors
    report = event.get('report', '')
    if not report:
        logger.warning(f"No report found for event {event['title']} – skipping chunk upsert")
        return

    chunks = list(_chunk_text(report))
    vectors_to_upsert = []
    for idx, chunk in enumerate(chunks):
        chunk_embedding = generate_embedding(chunk)
        chunk_id = f"{event_id}_chunk_{idx}"
        chunk_metadata = {
            'event_id': event_id,
            'chunk_index': idx,
            'title': event['title'],
            'sources': event.get('sources', []),
            'text': chunk  # Store chunk text for retrieval
        }
        vectors_to_upsert.append((chunk_id, chunk_embedding, chunk_metadata))

    pinecone_index.upsert(
        namespace=REPORT_NAMESPACE,
        vectors=vectors_to_upsert
    )

    logger.info(
        f"Successfully upserted {len(vectors_to_upsert)} report chunks and overview vector for event: {event['title']}"
    )

def store_to_mongodb(event):
    """Save complete event document to MongoDB collection."""
    logger.info(f"Storing event to MongoDB: {event['title']}")
    
    db = mongo_client['events']
    collection = db['global']
    
    # Insert event document
    result = collection.insert_one(event)
    
    logger.info(f"Successfully stored event to MongoDB with ID: {result.inserted_id}")

def main():
    """Execute end-to-end event discovery and storage pipeline.
    
    1. Discover events via Perplexity API
    2. Generate embeddings for deduplication
    3. Filter out duplicates using vector similarity
    4. Enrich unique events with detailed content
    5. Store in vector and document databases
    """
    logger.info("Starting timeline researcher workflow")
    
    try:
        # 1. Discover recent significant events
        events = search_events_with_perplexity()
        total_events_found = len(events)
        
        # Connect to Pinecone index (must exist in Pinecone dashboard)
        pinecone_index = pc.Index(PINECONE_INDEX_NAME)
        
        # Process events for storage
        unique_events = []
        duplicate_count = 0
        for event in events:
            # 2. Generate vector representation
            combined_text = f"{event['title']} {event['summary']}"
            embedding = generate_embedding(combined_text)
            
            # 3. Detect semantic duplicates
            is_duplicate = check_duplicate_in_pinecone(pinecone_index, embedding, {'title': event['title'], 'summary': event['summary']})
            
            if not is_duplicate:
                # 4. Enrich with detailed analysis
                detailed_event = research_event_details(event)
                unique_events.append((detailed_event, embedding))
            else:
                duplicate_count += 1
            
            # Early exit if all events are duplicates
            if not unique_events:
                logger.info("All events are duplicates, stopping workflow")
                # Log final stats
                logger.info(f"=== Timeline Researcher Statistics ===")
                logger.info(f"Total events found: {total_events_found}")
                logger.info(f"Duplicate events: {duplicate_count}")
                logger.info(f"Unique events processed: {len(unique_events)}")
                logger.info(f"===================================")
                return
        
        # 5. Store unique events in databases
        stored_count = 0
        for event, embedding in unique_events:
            # Store vector representations
            upsert_to_pinecone(pinecone_index, event, embedding)
            
            # Store complete document
            store_to_mongodb(event)
            stored_count += 1
        
        # Log execution summary
        logger.info(f"=== Timeline Researcher Statistics ===")
        logger.info(f"Total events found: {total_events_found}")
        logger.info(f"Duplicate events: {duplicate_count}")
        logger.info(f"Unique events processed: {len(unique_events)}")
        logger.info(f"Events stored in database: {stored_count}")
        logger.info(f"===================================")
    
    except Exception as e:
        logger.error(f"Error in timeline researcher workflow: {str(e)}")
        raise

if __name__ == "__main__":
    main() 