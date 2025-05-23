import unittest
from unittest.mock import patch, MagicMock
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from event_research.workflows.event_pipeline import run


class TestEventPipeline(unittest.TestCase):

    def setUp(self):
        # Test event data
        self.test_events = [
            {
                "date": "2023-07-01",
                "title": "Test Event 1",
                "summary": "This is the first test event summary.",
                "research": "",
                "sources": []
            },
            {
                "date": "2023-07-01",
                "title": "Test Event 2",
                "summary": "This is the second test event summary.",
                "research": "",
                "sources": []
            }
        ]
        
        # Mock embedding
        self.test_embedding = [0.1] * 3072
        
        # Mock results from research
        self.researched_event = {
            "date": "2023-07-01",
            "title": "Test Event 1",
            "summary": "This is the first test event summary.",
            "research": "This is a detailed research about the event.",
            "sources": ["https://example.com/1", "https://example.com/2"]
        }

    @patch('event_research.workflows.event_pipeline.get_pinecone_index')
    @patch('event_research.workflows.event_pipeline.search_events')
    @patch('event_research.workflows.event_pipeline.generate_embedding')
    @patch('event_research.workflows.event_pipeline.check_duplicates')
    @patch('event_research.workflows.event_pipeline.investigate_event')
    @patch('event_research.workflows.event_pipeline.upsert_to_pinecone')
    @patch('event_research.workflows.event_pipeline.store_to_mongodb')
    def test_run_with_unique_events(
        self, 
        mock_store_mongodb, 
        mock_upsert_pinecone, 
        mock_investigate_event, 
        mock_check_duplicates, 
        mock_generate_embedding, 
        mock_search_events, 
        mock_get_pinecone_index
    ):
        # Setup mocks
        mock_search_events.return_value = self.test_events
        mock_generate_embedding.return_value = self.test_embedding
        mock_check_duplicates.return_value = False  # No duplicates
        mock_investigate_event.return_value = self.researched_event
        mock_pinecone_index = MagicMock()
        mock_get_pinecone_index.return_value = mock_pinecone_index
        
        # Call the pipeline
        run()
        
        # Assertions
        mock_search_events.assert_called_once()
        self.assertEqual(mock_generate_embedding.call_count, 2)  # Called for each event
        self.assertEqual(mock_check_duplicates.call_count, 2)  # Called for each event
        self.assertEqual(mock_investigate_event.call_count, 2)  # Called for each unique event
        self.assertEqual(mock_upsert_pinecone.call_count, 2)  # Called for each processed event
        self.assertEqual(mock_store_mongodb.call_count, 2)  # Called for each processed event

    @patch('event_research.workflows.event_pipeline.get_pinecone_index')
    @patch('event_research.workflows.event_pipeline.search_events')
    @patch('event_research.workflows.event_pipeline.generate_embedding')
    @patch('event_research.workflows.event_pipeline.check_duplicates')
    @patch('event_research.workflows.event_pipeline.investigate_event')
    @patch('event_research.workflows.event_pipeline.upsert_to_pinecone')
    @patch('event_research.workflows.event_pipeline.store_to_mongodb')
    def test_run_with_all_duplicates(
        self, 
        mock_store_mongodb, 
        mock_upsert_pinecone, 
        mock_investigate_event, 
        mock_check_duplicates, 
        mock_generate_embedding, 
        mock_search_events, 
        mock_get_pinecone_index
    ):
        # Setup mocks
        mock_search_events.return_value = self.test_events
        mock_generate_embedding.return_value = self.test_embedding
        mock_check_duplicates.return_value = True  # All are duplicates
        mock_pinecone_index = MagicMock()
        mock_get_pinecone_index.return_value = mock_pinecone_index
        
        # Call the pipeline
        run()
        
        # Assertions
        mock_search_events.assert_called_once()
        self.assertEqual(mock_generate_embedding.call_count, 2)  # Called for each event
        self.assertEqual(mock_check_duplicates.call_count, 2)  # Called for each event
        mock_investigate_event.assert_not_called()  # Not called for duplicates
        mock_upsert_pinecone.assert_not_called()  # Not called when all are duplicates
        mock_store_mongodb.assert_not_called()  # Not called when all are duplicates

    @patch('event_research.workflows.event_pipeline.get_pinecone_index')
    @patch('event_research.workflows.event_pipeline.search_events')
    @patch('event_research.workflows.event_pipeline.generate_embedding')
    @patch('event_research.workflows.event_pipeline.check_duplicates')
    @patch('event_research.workflows.event_pipeline.investigate_event')
    @patch('event_research.workflows.event_pipeline.upsert_to_pinecone')
    @patch('event_research.workflows.event_pipeline.store_to_mongodb')
    def test_run_with_mixed_results(
        self, 
        mock_store_mongodb, 
        mock_upsert_pinecone, 
        mock_investigate_event, 
        mock_check_duplicates, 
        mock_generate_embedding, 
        mock_search_events, 
        mock_get_pinecone_index
    ):
        # Setup mocks
        mock_search_events.return_value = self.test_events
        mock_generate_embedding.return_value = self.test_embedding
        # First event is unique, second is duplicate
        mock_check_duplicates.side_effect = [False, True]
        mock_investigate_event.return_value = self.researched_event
        mock_pinecone_index = MagicMock()
        mock_get_pinecone_index.return_value = mock_pinecone_index
        
        # Call the pipeline
        run()
        
        # Assertions
        mock_search_events.assert_called_once()
        self.assertEqual(mock_generate_embedding.call_count, 2)  # Called for each event
        self.assertEqual(mock_check_duplicates.call_count, 2)  # Called for each event
        mock_investigate_event.assert_called_once()  # Called only for the unique event
        mock_upsert_pinecone.assert_called_once()  # Called only for the unique event
        mock_store_mongodb.assert_called_once()  # Called only for the unique event

    @patch('event_research.workflows.event_pipeline.get_pinecone_index')
    @patch('event_research.workflows.event_pipeline.search_events')
    def test_run_with_no_events(self, mock_search_events, mock_get_pinecone_index):
        # Setup mocks
        mock_search_events.return_value = []  # No events found
        mock_pinecone_index = MagicMock()
        mock_get_pinecone_index.return_value = mock_pinecone_index
        
        # Call the pipeline
        run()
        
        # Assertions
        mock_search_events.assert_called_once()
        mock_get_pinecone_index.assert_called_once()
        # No further processing should occur


if __name__ == '__main__':
    unittest.main() 