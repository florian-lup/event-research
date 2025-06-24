import unittest
from unittest.mock import patch, MagicMock
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from services import generate_embedding, chunk_text


class TestEmbeddings(unittest.TestCase):

    @patch('event_research.services.embeddings._openai')
    def test_generate_embedding(self, mock_openai):
        # Setup mock response
        mock_embedding = [0.1] * 3072
        mock_response = MagicMock()
        mock_response.data = [MagicMock(embedding=mock_embedding)]
        mock_openai.embeddings.create.return_value = mock_response

        # Call the function
        result = generate_embedding("Test text")

        # Assertions
        self.assertEqual(len(result), 3072)
        self.assertEqual(result, mock_embedding)
        mock_openai.embeddings.create.assert_called_once()

    def test_chunk_text_empty(self):
        # Test with empty text
        chunks = list(chunk_text(""))
        self.assertEqual(chunks, [])

    def test_chunk_text_short(self):
        # Test with text shorter than max_tokens
        short_text = "This is a short text"
        chunks = list(chunk_text(short_text, max_tokens=10))
        self.assertEqual(chunks, [short_text])

    def test_chunk_text_long(self):
        # Test with text longer than max_tokens
        words = ["word"] * 500
        long_text = " ".join(words)
        chunks = list(chunk_text(long_text, max_tokens=100))
        
        # Should be split into 5 chunks of 100 words each
        self.assertEqual(len(chunks), 5)
        for chunk in chunks:
            # Each chunk should have 100 words except potentially the last one
            self.assertLessEqual(len(chunk.split()), 100)


if __name__ == '__main__':
    unittest.main() 