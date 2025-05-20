import unittest
from unittest.mock import patch, MagicMock
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from event_research.services.deduplication import check_duplicates


class TestDeduplication(unittest.TestCase):

    def setUp(self):
        self.test_embedding = [0.1] * 1536
        self.test_metadata = {"title": "Test Event", "summary": "This is a test event"}

    def test_check_duplicates_match_found(self):
        # Create a mock index with a matching result
        mock_index = MagicMock()
        mock_match = MagicMock(score=0.85)  # Above threshold
        mock_response = MagicMock(matches=[mock_match])
        mock_index.query.return_value = mock_response

        # Call function and assert result
        result = check_duplicates(mock_index, self.test_embedding, self.test_metadata)
        self.assertTrue(result)
        mock_index.query.assert_called_once()

    def test_check_duplicates_no_match(self):
        # Create a mock index with a below-threshold match
        mock_index = MagicMock()
        mock_match = MagicMock(score=0.75)  # Below threshold
        mock_response = MagicMock(matches=[mock_match])
        mock_index.query.return_value = mock_response

        # Call function and assert result
        result = check_duplicates(mock_index, self.test_embedding, self.test_metadata)
        self.assertFalse(result)
        mock_index.query.assert_called_once()

    def test_check_duplicates_no_results(self):
        # Create a mock index with no results
        mock_index = MagicMock()
        mock_response = MagicMock(matches=[])
        mock_index.query.return_value = mock_response

        # Call function and assert result
        result = check_duplicates(mock_index, self.test_embedding, self.test_metadata)
        self.assertFalse(result)
        mock_index.query.assert_called_once()


if __name__ == '__main__':
    unittest.main() 