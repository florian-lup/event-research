import unittest
from unittest.mock import patch, MagicMock, call
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from services import upsert_to_pinecone, store_to_mongodb


class TestStorage(unittest.TestCase):

    def setUp(self):
        # Setup test data
        self.test_event = {
            "title": "Test Event",
            "summary": "This is a test event summary.",
            "research": "This is a detailed research about the test event. It contains multiple sentences to test chunking.",
            "sources": ["https://example.com/1", "https://example.com/2"]
        }
        self.test_embedding = [0.1] * 3072
        self.mock_index = MagicMock()

    @patch('event_research.services.storage.generate_embedding')
    @patch('event_research.services.storage.chunk_text')
    def test_upsert_to_pinecone_with_research(self, mock_chunk_text, mock_generate_embedding):
        # Setup mocks
        mock_chunk_text.return_value = ["Chunk 1", "Chunk 2"]
        mock_generate_embedding.return_value = self.test_embedding
        
        # Call the function
        upsert_to_pinecone(self.mock_index, self.test_event, self.test_embedding)
        
        # Assertions - verify the index was called twice (once for overview, once for chunks)
        self.assertEqual(self.mock_index.upsert.call_count, 2)
        
        # Check the first call for overview vector
        first_call_namespace = self.mock_index.upsert.call_args_list[0][1]['namespace']
        first_call_vectors = self.mock_index.upsert.call_args_list[0][1]['vectors']
        self.assertTrue(any("event_id" in vector[2] for vector in first_call_vectors))
        self.assertTrue(any("title" in vector[2] for vector in first_call_vectors))
        self.assertTrue(any("summary" in vector[2] for vector in first_call_vectors))
        
        # Check the second call for chunk vectors
        second_call_namespace = self.mock_index.upsert.call_args_list[1][1]['namespace']
        second_call_vectors = self.mock_index.upsert.call_args_list[1][1]['vectors']
        self.assertEqual(len(second_call_vectors), 2)  # Two chunks
        self.assertTrue(any("chunk_index" in vector[2] for vector in second_call_vectors))
        self.assertTrue(any("sources" in vector[2] for vector in second_call_vectors))
        
        # Verify chunk_text was called with the research
        mock_chunk_text.assert_called_once_with(self.test_event["research"])
        
        # Verify generate_embedding was called for each chunk
        self.assertEqual(mock_generate_embedding.call_count, 2)

    @patch('event_research.services.storage.chunk_text')
    def test_upsert_to_pinecone_no_research(self, mock_chunk_text):
        # Setup event with no research
        event_no_research = {
            "title": "Test Event",
            "summary": "This is a test event summary.",
            "research": "",
            "sources": []
        }
        
        # Call the function
        upsert_to_pinecone(self.mock_index, event_no_research, self.test_embedding)
        
        # Assertions - verify the index was called only once (for overview)
        self.mock_index.upsert.assert_called_once()
        
        # Verify chunk_text was not called
        mock_chunk_text.assert_not_called()

    @patch('event_research.services.storage._db')
    def test_store_to_mongodb(self, mock_db):
        # Setup mock
        mock_insert_result = MagicMock()
        mock_insert_result.inserted_id = "test_id_123"
        mock_db.insert_one.return_value = mock_insert_result
        
        # Call the function
        store_to_mongodb(self.test_event)
        
        # Assertions
        mock_db.insert_one.assert_called_once_with(self.test_event)


if __name__ == '__main__':
    unittest.main() 