#!/usr/bin/env python3
import logging
from timeline_researcher import (
    search_events_with_perplexity,
    generate_embedding,
    check_duplicate_in_pinecone,
    research_event_details,
    upsert_to_pinecone,
    store_to_mongodb,
    create_pinecone_index_if_not_exists
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def test_search_events():
    """Test event searching via Perplexity API"""
    logger.info("Testing search_events_with_perplexity function...")
    try:
        events = search_events_with_perplexity()
        logger.info(f"SUCCESS: Found {len(events)} events")
        for i, event in enumerate(events):
            logger.info(f"Event {i+1}: {event['title']}")
        return events
    except Exception as e:
        logger.error(f"FAILED: Error searching events: {str(e)}")
        return None

def test_embedding_generation(events):
    """Test text-to-vector embedding conversion"""
    logger.info("Testing generate_embedding function...")
    if not events:
        logger.warning("No events to generate embeddings for, skipping test")
        return None
    
    try:
        event = events[0]
        combined_text = f"{event['title']} {event['summary']}"
        embedding = generate_embedding(combined_text)
        logger.info(f"SUCCESS: Generated embedding with {len(embedding)} dimensions")
        return embedding
    except Exception as e:
        logger.error(f"FAILED: Error generating embedding: {str(e)}")
        return None

def test_pinecone_index():
    """Test Pinecone index creation or retrieval"""
    logger.info("Testing create_pinecone_index_if_not_exists function...")
    try:
        index = create_pinecone_index_if_not_exists()
        logger.info(f"SUCCESS: Got Pinecone index")
        return index
    except Exception as e:
        logger.error(f"FAILED: Error creating/getting Pinecone index: {str(e)}")
        return None

def test_duplicate_check(index, embedding, event):
    """Test similarity-based duplicate detection"""
    logger.info("Testing check_duplicate_in_pinecone function...")
    if not index or embedding is None or not event:
        logger.warning("Missing required inputs, skipping duplicate check test")
        return None
    
    try:
        metadata = {'title': event['title'], 'summary': event['summary']}
        is_duplicate = check_duplicate_in_pinecone(index, embedding, metadata)
        logger.info(f"SUCCESS: Duplicate check result: {is_duplicate}")
        return is_duplicate
    except Exception as e:
        logger.error(f"FAILED: Error checking for duplicates: {str(e)}")
        return None

def test_event_research(event):
    """Test detailed event information retrieval"""
    logger.info("Testing research_event_details function...")
    if not event:
        logger.warning("No event to research, skipping test")
        return None
    
    try:
        detailed_event = research_event_details(event)
        logger.info(f"SUCCESS: Researched event details")
        logger.info(f"Content length: {len(detailed_event['content'])}")
        logger.info(f"Sources count: {len(detailed_event['sources'])}")
        return detailed_event
    except Exception as e:
        logger.error(f"FAILED: Error researching event details: {str(e)}")
        return None

def test_pinecone_upsert(index, event, embedding):
    """Test vector database storage"""
    logger.info("Testing upsert_to_pinecone function...")
    if not index or not event or embedding is None:
        logger.warning("Missing required inputs, skipping Pinecone upsert test")
        return False
    
    try:
        upsert_to_pinecone(index, event, embedding)
        logger.info(f"SUCCESS: Upserted event to Pinecone")
        return True
    except Exception as e:
        logger.error(f"FAILED: Error upserting to Pinecone: {str(e)}")
        return False

def test_mongodb_store(event):
    """Test document database storage"""
    logger.info("Testing store_to_mongodb function...")
    if not event:
        logger.warning("No event to store, skipping test")
        return False
    
    try:
        store_to_mongodb(event)
        logger.info(f"SUCCESS: Stored event to MongoDB")
        return True
    except Exception as e:
        logger.error(f"FAILED: Error storing to MongoDB: {str(e)}")
        return False

def main():
    """Execute full workflow test sequence"""
    logger.info("Starting timeline researcher workflow tests")
    
    # Test search
    events = test_search_events()
    if not events:
        logger.error("Cannot continue tests without events")
        return
    
    # Test embedding generation
    embedding = test_embedding_generation(events)
    if embedding is None:
        logger.error("Cannot continue tests without embeddings")
        return
    
    # Test Pinecone index
    index = test_pinecone_index()
    if not index:
        logger.error("Cannot continue tests without Pinecone index")
        return
    
    # Test duplicate check
    is_duplicate = test_duplicate_check(index, embedding, events[0])
    
    # Test event research
    detailed_event = test_event_research(events[0])
    if not detailed_event:
        logger.error("Cannot continue tests without detailed event")
        return
    
    # Test Pinecone upsert
    upsert_success = test_pinecone_upsert(index, detailed_event, embedding)
    
    # Test MongoDB store
    store_success = test_mongodb_store(detailed_event)
    
    # Print test summary
    logger.info("=== TEST SUMMARY ===")
    logger.info(f"Search Events: {'SUCCESS' if events else 'FAILED'}")
    logger.info(f"Embedding Generation: {'SUCCESS' if embedding is not None else 'FAILED'}")
    logger.info(f"Pinecone Index: {'SUCCESS' if index else 'FAILED'}")
    logger.info(f"Duplicate Check: {'SUCCESS' if is_duplicate is not None else 'FAILED'}")
    logger.info(f"Event Research: {'SUCCESS' if detailed_event else 'FAILED'}")
    logger.info(f"Pinecone Upsert: {'SUCCESS' if upsert_success else 'FAILED'}")
    logger.info(f"MongoDB Store: {'SUCCESS' if store_success else 'FAILED'}")

if __name__ == "__main__":
    main() 