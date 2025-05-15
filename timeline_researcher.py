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
from openai import OpenAI

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
EMBEDDING_MODEL = 'text-embedding-3-small'
EMBEDDING_DIMENSIONS = 1536
SIMILARITY_THRESHOLD = 0.8
CURRENT_DATE = datetime.now().strftime("%m/%d/%Y")

def create_pinecone_index_if_not_exists():
    """Create or retrieve the Pinecone vector index using singleton pattern"""
    global pinecone_index
    if pinecone_index is not None:
        return pinecone_index

    logger.info("Checking if Pinecone index exists…")
    indexes = pc.list_indexes()
    index_names = [index.name for index in indexes]

    if PINECONE_INDEX_NAME not in index_names:
        logger.info(f"Creating Pinecone index: {PINECONE_INDEX_NAME}")
        pc.create_index(
            name=PINECONE_INDEX_NAME,
            dimension=EMBEDDING_DIMENSIONS,
            metric='cosine',
        )
        logger.info("Pinecone index created")
    else:
        logger.info("Pinecone index already exists")

    pinecone_index = pc.Index(PINECONE_INDEX_NAME)
    return pinecone_index

def search_events_with_perplexity():
    """Query Perplexity API to identify significant global events from today"""
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
                    "Respond ONLY in English and strictly follow the user instructions. Output EXACTLY the JSON that matches the provided schema – no markdown fences, no commentary."
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
                    "4) Use neutral, factual language." 
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
    
    # Handle <think> tag sections if present
    think_match = re.search(r'<think>(.*?)</think>(.*)', response_text, re.DOTALL)
    if think_match:
        # Extract the content after </think>
        response_text = think_match.group(2).strip()
    
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
        raise Exception(f"Failed to parse JSON: {e}")
    
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
        "model": "sonar-reasoning-pro",
        "messages": [
            {
                "role": "system",
                "content": (
                    "You are an expert investigative journalist who produces comprehensive, in-depth analyses of current events. "
                    "Your analyses are thorough, well-researched, and include historical context, current developments, and future implications. "
                    "Respond ONLY in English and deliver factual content with professional tone. Structure your response using clear GitHub-flavoured Markdown. Formatting rules:\n"
                    "• Use headings with ##, ###, etc.\n"
                    "• Insert ONE blank line between paragraphs.\n"
                    "• Insert ONE blank line *before* starting any list (-, *, +, 1. …).\n"
                    "• Avoid trailing spaces at the ends of lines.\n"
                    "• Use **bold**, *italic* when needed.\n"
                )
            },
            {
                "role": "user",
                "content": (
                    f"Provide a comprehensive, in-depth analysis of this significant global event: '{event['title']}'.\n"
                    "Requirements:\n"
                    "1) Deliver a thorough analysis including historical context, key players, current developments, global implications, and potential future outcomes.\n"
                    "2) Include relevant statistics, expert opinions, and critical perspectives where available.\n"
                    "3) Write in a formal, analytical style appropriate for a serious news publication.\n"
                    "4) Use only verifiable facts from reputable sources."
                )
            }
        ],
        "search_recency_filter": "day",
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
    
    # Remove <think> block if present
    think_match = re.search(r'<think>(.*?)</think>(.*)', response_text, re.DOTALL | re.IGNORECASE)
    if think_match:
        response_text = think_match.group(2).strip()

    # Remove citation markers like [1], [2] while preserving markdown formatting
    cleaned_content = re.sub(r'\[\d+\]', '', response_text).strip()
    content = cleaned_content
    
    # Extract citation metadata
    sources = response_json.get('citations', [])
    logger.info(f"Found {len(sources)} citations from Perplexity API")
    
    # Update event with detailed information
    event['content'] = content
    event['sources'] = sources
    
    logger.info(f"Successfully researched details for event: {event['title']}")
    return event

def upsert_to_pinecone(pinecone_index, event, embedding):
    """Store event vector in Pinecone for semantic search and deduplication"""
    logger.info(f"Upserting event to Pinecone: {event['title']}")
    
    # Create deterministic ID from title hash
    event_id = f"event_{hash(event['title'])}"
    
    # Store essential metadata for search results
    metadata = {
        'title': event['title'],
        'summary': event['summary']
    }
    
    # Store vector and metadata
    pinecone_index.upsert(
        vectors=[
            (event_id, embedding, metadata)
        ]
    )
    
    logger.info(f"Successfully upserted event to Pinecone: {event['title']}")

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
        
        # Initialize vector database
        pinecone_index = create_pinecone_index_if_not_exists()
        
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