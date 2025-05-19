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

# Set up logger
logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler()
    ]
)

# Load environment variables
load_dotenv()

# API configuration
PERPLEXITY_API_KEY = os.getenv('PERPLEXITY_API_KEY')
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
PINECONE_API_KEY = os.getenv('PINECONE_API_KEY')
PINECONE_ENVIRONMENT = os.getenv('PINECONE_ENVIRONMENT')
MONGODB_URI = os.getenv('MONGODB_URI')

# Initialize API clients
logger.info("Initializing API clients...")
pc = Pinecone(api_key=PINECONE_API_KEY)
logger.info("Initialized Pinecone client")
mongo_client = MongoClient(MONGODB_URI)
logger.info("Initialized MongoDB client")

# Initialize OpenAI client (singleton pattern)
from openai import OpenAI as _OpenAIClient
openai_client = _OpenAIClient(api_key=OPENAI_API_KEY)
logger.info("Initialized OpenAI client")

# Shared HTTP session for Perplexity to optimize connection reuse
perplexity_session = requests.Session()

# Shared Pinecone index (initialized lazily)
pinecone_index = None  # type: ignore

# Constants
PINECONE_INDEX_NAME = 'events'
# Embedding configuration (using a larger model for improved semantic quality)
EMBEDDING_MODEL = 'text-embedding-3-large'
EMBEDDING_DIMENSIONS = 3072
SIMILARITY_THRESHOLD = 0.8
CURRENT_DATE = datetime.now().strftime("%m/%d/%Y")

# Pinecone namespace conventions
OVERVIEW_NAMESPACE = 'overview'  # One vector per event (title + summary)
CONTENT_NAMESPACE = 'content'    # Many vectors per event (chunked content)

def _strip_think_blocks(text: str) -> str:
    """Return the part of the reply that comes *after* the closing </think> tag."""

    if not text:
        return text.strip()

    marker = "</think>"
    idx = text.rfind(marker)

    # If the marker is missing, fall back to full text (should be rare).
    after = text if idx == -1 else text[idx + len(marker):]

    cleaned = after.strip()

    # Remove ```json or generic ``` fences if present.
    if cleaned.startswith("```json"):
        cleaned = cleaned[len("```json"):].strip()
    if cleaned.startswith("```"):
        cleaned = cleaned[3:].strip()
    if cleaned.endswith("```"):
        cleaned = cleaned[:-3].strip()

    return cleaned

def _sanitize_llm_text(text: str, *, remove_citations: bool = True, remove_markdown: bool = False) -> str:
    """Clean raw LLM output so downstream parsing is robust.

    Steps performed:
    1. Strip chain-of-thought blocks and surrounding ``` fences via ``_strip_think_blocks``.
    2. Optionally remove inline citation markers:
       • Numeric references like ``[1]`` or ``[23]``.
       • Source tags like ``[Reuters]`` or ``[BBC]``.
    3. Optionally remove basic markdown syntax (headings, lists, emphasis, horizontal rules)

    Parameters
    ----------
    text : str
        Raw text from the LLM.
    remove_citations : bool, default True
        If ``True``, both numeric and textual inline citations are removed.
    remove_markdown : bool, default False
        If ``True``, basic markdown syntax (headings, lists, emphasis, horizontal rules) is stripped.

    Returns
    -------
    str
        Sanitised text ready for JSON parsing or storage.
    """

    cleaned = _strip_think_blocks(text)

    if remove_citations:
        # Remove numeric citations like [1], [12]
        cleaned = re.sub(r"\[\d+\]", "", cleaned)
        # Remove textual citations like [Reuters], [NYT]
        cleaned = re.sub(r"\[[A-Za-z][^\]]+\]", "", cleaned)

    if remove_markdown:
        # Strip heading markers (e.g., `# Heading`, `## Heading`)
        cleaned = re.sub(r"^#{1,6}\s*", "", cleaned, flags=re.MULTILINE)
        # Strip unordered list bullets (-, *, +)
        cleaned = re.sub(r"^\s*[-*+]\s+", "", cleaned, flags=re.MULTILINE)
        # Strip numbered list markers (1. 2. etc.)
        cleaned = re.sub(r"^\s*\d+\.\s+", "", cleaned, flags=re.MULTILINE)
        # Remove horizontal rules (--- or *** lines)
        cleaned = re.sub(r"^(?:-{3,}|\*{3,})$", "", cleaned, flags=re.MULTILINE)
        # Remove emphasis markers (**bold**, *italic*, __bold__, _italic_)
        cleaned = re.sub(r"(\*\*|__|\*|_)", "", cleaned)

    return cleaned.strip()

def search_events_with_perplexity():
    """Query Perplexity API to identify significant global events from today"""
    logger.info("Searching for top 5 global events with Perplexity API...")
    
    headers = {
        "Authorization": f"Bearer {PERPLEXITY_API_KEY}",
        "Content-Type": "application/json"
    }
    
    data = {
        "model": "sonar-reasoning-pro",
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
    
    # Extract JSON from the response
    response_text = response.json()['choices'][0]['message']['content']
    logger.info(f"Response from Perplexity: {response_text[:100]}...")  # Log beginning of response
    
    # Strip the <think> reasoning section and any ```json fences but leave JSON content untouched
    response_text = _strip_think_blocks(response_text)
    
    # Extract JSON from various possible formats
    json_match = re.search(r'```json\s*(.*?)\s*```', response_text, re.DOTALL)
    
    if not json_match:
        # Fallback to find JSON without markdown code blocks
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
    
    # Construct complete event objects with placeholder fields
    complete_events = []
    for event in events:
        # Clean citation markers from summary
        cleaned_summary = re.sub(r'\[\d+\]', '', event['summary'])
        
        complete_events.append({
            'date': datetime.now().isoformat(),
            'title': event['title'],
            'summary': cleaned_summary,
            'content': '',
            'sources': []
        })
    
    logger.info(f"Found {len(complete_events)} events with Perplexity API")
    return complete_events

def generate_embedding(text):
    """Convert text to vector representation using OpenAI embeddings API"""
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
    """Detect semantic duplicates by checking vector similarity against threshold"""
    logger.info(f"Checking for duplicates for: {metadata['title']}")
    
    query_response = pinecone_index.query(
        namespace=OVERVIEW_NAMESPACE,
        vector=embedding,
        top_k=5,
        include_metadata=True
    )
    
    # Compare similarity scores against threshold
    for match in query_response.matches:
        if match.score >= SIMILARITY_THRESHOLD:
            logger.info(f"Found duplicate: {metadata['title']} - Similarity: {match.score}")
            return True
    
    logger.info(f"No duplicates found for: {metadata['title']}")
    return False

def research_event_details(event):
    """Enrich event with comprehensive analysis and source citations"""
    logger.info(f"Researching details for event: {event['title']}")
    
    headers = {
        "Authorization": f"Bearer {PERPLEXITY_API_KEY}",
        "Content-Type": "application/json"
    }
    
    data = {
        "model": "sonar-deep-research",
        "messages": [
            {
                "role": "system",
                "content": (
                    "You are an expert investigative journalist who produces comprehensive, in-depth analyses of current events. "
                    "Your analyses are thorough, well-researched, and include historical context, current developments, and future implications. "
                )
            },
            {
                "role": "user",
                "content": (
                    f"Provide a comprehensive, in-depth analysis of this significant global event: '{event['title']}'."
                )
            }
        ],
        "web_search_options": {"search_context_size": "high"}
    }
    
    response = perplexity_session.post(
        "https://api.perplexity.ai/chat/completions",
        headers=headers,
        json=data
    )
    
    if response.status_code != 200:
        logger.error(f"Error from Perplexity API: {response.status_code} - {response.text}")
        raise Exception(f"Perplexity API error: {response.status_code}")
    
    # Parse the full JSON response
    response_json = response.json()
    
    # Extract content from the response
    response_text = response_json['choices'][0]['message']['content']
    logger.info(f"Response from Perplexity (event research): {response_text[:100]}...")
    
    # Sanitize content (strip chain-of-thought, citations, fences)
    content = _sanitize_llm_text(response_text, remove_citations=True, remove_markdown=False)
    
    # Extract citation metadata
    sources = response_json.get('citations', [])
    logger.info(f"Found {len(sources)} citations from Perplexity API")
    
    # Update event with detailed information
    event['content'] = content
    event['sources'] = sources
    
    logger.info(f"Successfully researched details for event: {event['title']}")
    return event

def _chunk_text(text: str, max_tokens: int = 300):
    """Simple word-based chunker (≈ tokens) to split long content into chunks."""
    if not text:
        return []
    words = text.split()
    for i in range(0, len(words), max_tokens):
        yield ' '.join(words[i:i + max_tokens])

def upsert_to_pinecone(pinecone_index, event, overview_embedding):
    """Upsert overview vector (dedup) and chunked content vectors into their respective namespaces."""
    logger.info(f"Upserting event (overview + chunks) to Pinecone: {event['title']}")

    # Deterministic ID derived from title hash (stable across runs)
    event_id = f"event_{hash(event['title'])}"

    # 1) Upsert overview vector (title + summary) ---------------------------------
    overview_metadata = {
        'event_id': event_id,
        'title': event['title'],
        'summary': event['summary'],
    }

    pinecone_index.upsert(
        namespace=OVERVIEW_NAMESPACE,
        vectors=[
            (event_id, overview_embedding, overview_metadata)
        ]
    )

    # 2) Upsert chunked content vectors ------------------------------------------
    content = event.get('content', '')
    if not content:
        logger.warning(f"No content found for event {event['title']} – skipping chunk upsert")
        return

    chunks = list(_chunk_text(content))
    vectors_to_upsert = []
    for idx, chunk in enumerate(chunks):
        chunk_embedding = generate_embedding(chunk)
        chunk_id = f"{event_id}_chunk_{idx}"
        chunk_metadata = {
            'event_id': event_id,
            'chunk_index': idx,
            'title': event['title'],
            'sources': event.get('sources', []),
            'text': chunk  # Store the actual chunk text for downstream retrieval
        }
        vectors_to_upsert.append((chunk_id, chunk_embedding, chunk_metadata))

    pinecone_index.upsert(
        namespace=CONTENT_NAMESPACE,
        vectors=vectors_to_upsert
    )

    logger.info(
        f"Successfully upserted {len(vectors_to_upsert)} content chunks and overview vector for event: {event['title']}"
    )

def store_to_mongodb(event):
    """Persist full event details to MongoDB document database"""
    logger.info(f"Storing event to MongoDB: {event['title']}")
    
    db = mongo_client['events']
    collection = db['global']
    
    # Insert complete event document
    result = collection.insert_one(event)
    
    logger.info(f"Successfully stored event to MongoDB with ID: {result.inserted_id}")

def main():
    """Orchestrate end-to-end event discovery, deduplication, and storage workflow"""
    logger.info("Starting timeline researcher workflow")
    
    try:
        # Step 1: Search for recent significant events
        events = search_events_with_perplexity()
        total_events_found = len(events)
        
        # Open an existing Pinecone index (must be pre-created via dashboard)
        pinecone_index = pc.Index(PINECONE_INDEX_NAME)
        
        # Process each event for storage
        unique_events = []
        duplicate_count = 0
        for event in events:
            # Step 2: Generate vector representation
            combined_text = f"{event['title']} {event['summary']}"
            embedding = generate_embedding(combined_text)
            
            # Step 3: Detect semantic duplicates
            is_duplicate = check_duplicate_in_pinecone(pinecone_index, embedding, {'title': event['title'], 'summary': event['summary']})
            
            if not is_duplicate:
                # Step 4: Enrich with detailed analysis
                detailed_event = research_event_details(event)
                unique_events.append((detailed_event, embedding))
            else:
                duplicate_count += 1
            
            # Early exit if no unique events found
            if not unique_events:
                logger.info("All events are duplicates, stopping workflow")
                # Log statistics before exiting
                logger.info(f"=== Timeline Researcher Statistics ===")
                logger.info(f"Total events found: {total_events_found}")
                logger.info(f"Duplicate events: {duplicate_count}")
                logger.info(f"Unique events processed: {len(unique_events)}")
                logger.info(f"===================================")
                return
        
        # Steps 5 & 6: Persist unique events to databases
        stored_count = 0
        for event, embedding in unique_events:
            # Store vector representation
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