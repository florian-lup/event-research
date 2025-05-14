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

# Load environment variables
load_dotenv()

# API Keys and Configuration
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

# Initialize OpenAI client once (best practice)
from openai import OpenAI as _OpenAIClient
openai_client = _OpenAIClient(api_key=OPENAI_API_KEY)
logger.info("Initialized OpenAI client")

# Shared HTTP session for Perplexity – keeps TCP connection alive
perplexity_session = requests.Session()

# Shared Pinecone index (initialised lazily)
pinecone_index = None  # type: ignore

# Constants
PINECONE_INDEX_NAME = 'events'
EMBEDDING_MODEL = 'text-embedding-3-small'
EMBEDDING_DIMENSIONS = 1536
SIMILARITY_THRESHOLD = 0.8
CURRENT_DATE = datetime.now().strftime("%m/%d/%Y")

def create_pinecone_index_if_not_exists():
    """Return a shared Pinecone index, creating it on first call."""
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
    """Search for top 5 global events using Perplexity API"""
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
                "content": "You are a helpful assistant."
            },
            {
                "role": "user",
                "content": "Research and identify top 5 of the most significant global events"
            }
        ],
        "temperature": 0.5,
        "search_after_date_filter": CURRENT_DATE,
        "search_before_date_filter": CURRENT_DATE,
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
    
    # Extract JSON from the response (after the <think> section)
    response_text = response.json()['choices'][0]['message']['content']
    logger.info(f"Response from Perplexity: {response_text[:100]}...")  # Log beginning of response
    
    # Check for <think> tag and extract content after it
    think_match = re.search(r'<think>(.*?)</think>(.*)', response_text, re.DOTALL)
    if think_match:
        # Extract the content after </think>
        response_text = think_match.group(2).strip()
    
    # Try to find JSON in the response
    json_match = re.search(r'```json\s*(.*?)\s*```', response_text, re.DOTALL)
    
    if not json_match:
        # Try to find JSON without the markdown code blocks
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
    
    # Create complete event objects
    complete_events = []
    for event in events:
        complete_events.append({
            'date': datetime.now().isoformat(),
            'title': event['title'],
            'summary': event['summary'],
            'content': '',
            'sources': []
        })
    
    logger.info(f"Found {len(complete_events)} events with Perplexity API")
    return complete_events

def generate_embedding(text):
    """Generate embedding vector for the given text using the shared OpenAI client"""
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
    """Check if event already exists in Pinecone"""
    logger.info(f"Checking for duplicates for: {metadata['title']}")
    
    query_response = pinecone_index.query(
        vector=embedding,
        top_k=5,
        include_metadata=True
    )
    
    # Check if any result has similarity above threshold
    for match in query_response.matches:
        if match.score >= SIMILARITY_THRESHOLD:
            logger.info(f"Found duplicate: {metadata['title']} - Similarity: {match.score}")
            return True
    
    logger.info(f"No duplicates found for: {metadata['title']}")
    return False

def research_event_details(event):
    """Research additional details for an event using Perplexity API"""
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
                "content": "You are a helpful assistant."
            },
            {
                "role": "user",
                "content": f"Research this event '{event['title']}' and create an analysis"
            }
        ],
        "temperature": 0.5,
        "search_recency_filter": "day"
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
    
    # Extract content from the response (after the <think> section)
    response_text = response_json['choices'][0]['message']['content']
    logger.info(f"Response from Perplexity (event research): {response_text[:100]}...")  # Log beginning of response
    
    # Remove <think> block if present
    think_match = re.search(r'<think>(.*?)</think>(.*)', response_text, re.DOTALL | re.IGNORECASE)
    if think_match:
        response_text = think_match.group(2).strip()

    # Extract content (whole text without any filtering)
    content = response_text.strip()
    
    # Extract sources from the citations field in the API response
    sources = response_json.get('citations', [])
    logger.info(f"Found {len(sources)} citations from Perplexity API")
    
    # Update event
    event['content'] = content
    event['sources'] = sources
    
    logger.info(f"Successfully researched details for event: {event['title']}")
    return event

def upsert_to_pinecone(pinecone_index, event, embedding):
    """Upsert event to Pinecone"""
    logger.info(f"Upserting event to Pinecone: {event['title']}")
    
    # Create a unique ID for the event
    event_id = f"event_{hash(event['title'])}"
    
    # Prepare metadata
    metadata = {
        'title': event['title'],
        'summary': event['summary']
    }
    
    # Upsert to Pinecone
    pinecone_index.upsert(
        vectors=[
            (event_id, embedding, metadata)
        ]
    )
    
    logger.info(f"Successfully upserted event to Pinecone: {event['title']}")

def store_to_mongodb(event):
    """Store event to MongoDB"""
    logger.info(f"Storing event to MongoDB: {event['title']}")
    
    db = mongo_client['events']
    collection = db['global']
    
    # Insert event
    result = collection.insert_one(event)
    
    logger.info(f"Successfully stored event to MongoDB with ID: {result.inserted_id}")

def main():
    """Main workflow function"""
    logger.info("Starting timeline researcher workflow")
    
    try:
        # Step 1: Search events with Perplexity API
        events = search_events_with_perplexity()
        total_events_found = len(events)
        
        # Create/Get Pinecone index
        pinecone_index = create_pinecone_index_if_not_exists()
        
        # Process each event
        unique_events = []
        duplicate_count = 0
        for event in events:
            # Step 2: Generate embedding
            combined_text = f"{event['title']} {event['summary']}"
            embedding = generate_embedding(combined_text)
            
            # Step 3: Check for duplicates
            is_duplicate = check_duplicate_in_pinecone(pinecone_index, embedding, {'title': event['title'], 'summary': event['summary']})
            
            if not is_duplicate:
                # Step 4: Research event details
                detailed_event = research_event_details(event)
                unique_events.append((detailed_event, embedding))
            else:
                duplicate_count += 1
            
            # If all events are duplicates, stop
            if not unique_events:
                logger.info("All events are duplicates, stopping workflow")
                # Log statistics before exiting
                logger.info(f"=== Timeline Researcher Statistics ===")
                logger.info(f"Total events found: {total_events_found}")
                logger.info(f"Duplicate events: {duplicate_count}")
                logger.info(f"Unique events processed: {len(unique_events)}")
                logger.info(f"===================================")
                return
        
        # Steps 5 & 6: Store unique events
        stored_count = 0
        for event, embedding in unique_events:
            # Upsert to Pinecone
            upsert_to_pinecone(pinecone_index, event, embedding)
            
            # Store to MongoDB
            store_to_mongodb(event)
            stored_count += 1
        
        # Log final statistics
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