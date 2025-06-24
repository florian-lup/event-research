import unittest
from unittest.mock import patch, MagicMock
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from services import search_events
from services.discovery import _extract_json_from_response  # private function
from config import CURRENT_DATE


class TestDiscovery(unittest.TestCase):

    @patch('event_research.services.discovery.sanitize_llm_text')
    @patch('event_research.services.discovery.get_perplexity_session')
    def test_search_events_success(self, mock_get_session, mock_sanitize_llm_text):
        # Setup mock response
        mock_session = MagicMock()
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "choices": [
                {
                    "message": {
                        "content": """
                        ```json
                        {
                            "events": [
                                {
                                    "title": "Test Event 1",
                                    "summary": "This is the first test event."
                                },
                                {
                                    "title": "Test Event 2",
                                    "summary": "This is the second test event."
                                }
                            ]
                        }
                        ```
                        """
                    }
                }
            ]
        }
        mock_session.post.return_value = mock_response
        mock_get_session.return_value = mock_session
        
        # Setup mock sanitize function
        mock_sanitize_llm_text.side_effect = lambda summary, **kwargs: f"Cleaned: {summary}"

        # Call the function
        events = search_events()

        # Assertions
        self.assertEqual(len(events), 2)
        self.assertEqual(events[0]["title"], "Test Event 1")
        self.assertEqual(events[0]["summary"], "Cleaned: This is the first test event.")
        self.assertEqual(events[1]["title"], "Test Event 2")
        self.assertEqual(events[1]["summary"], "Cleaned: This is the second test event.")
        
        # Check that each event has the required fields with correct format
        for event in events:
            self.assertIn("date", event)
            self.assertEqual(event["date"], CURRENT_DATE)
            self.assertIn("title", event)
            self.assertIn("summary", event)
            self.assertIn("research", event)
            self.assertEqual(event["research"], "")
            self.assertIn("sources", event)
            self.assertEqual(event["sources"], [])
            
        # Verify sanitize_llm_text was called for each event summary
        self.assertEqual(mock_sanitize_llm_text.call_count, 2)
        
        # Verify the post request contains the correct schema
        called_data = mock_session.post.call_args.kwargs["json"]
        self.assertEqual(called_data["response_format"]["type"], "json_schema")
        self.assertIn("json_schema", called_data["response_format"])
        
        # Verify the schema structure matches what's expected in discovery.py
        schema = called_data["response_format"]["json_schema"]["schema"]
        self.assertIn("properties", schema)
        self.assertIn("events", schema["properties"])
        self.assertIn("items", schema["properties"]["events"])
        self.assertIn("properties", schema["properties"]["events"]["items"])
        self.assertIn("title", schema["properties"]["events"]["items"]["properties"])
        self.assertIn("summary", schema["properties"]["events"]["items"]["properties"])
        self.assertIn("required", schema["properties"]["events"]["items"])
        self.assertListEqual(schema["properties"]["events"]["items"]["required"], ["title", "summary"])

    @patch('event_research.services.discovery.get_perplexity_session')
    def test_search_events_api_error(self, mock_get_session):
        # Setup mock to return an error response
        mock_session = MagicMock()
        mock_response = MagicMock()
        mock_response.status_code = 400
        mock_response.text = "Bad request"
        mock_session.post.return_value = mock_response
        mock_get_session.return_value = mock_session

        # Assert that RuntimeError is raised
        with self.assertRaises(RuntimeError):
            search_events()

    @patch('event_research.services.discovery.strip_think_blocks')
    def test_extract_json_from_response_fenced(self, mock_strip_think_blocks):
        # Setup mock for strip_think_blocks
        mock_strip_think_blocks.side_effect = lambda text: text
        
        # Test extracting JSON from a fenced code block
        test_response = """
        Here's the result:
        ```json
        {"events": [{"title": "Test", "summary": "Summary"}]}
        ```
        """
        result = _extract_json_from_response(test_response)
        self.assertEqual(result, {"events": [{"title": "Test", "summary": "Summary"}]})
        mock_strip_think_blocks.assert_called_once()

    @patch('event_research.services.discovery.strip_think_blocks')
    def test_extract_json_from_response_braces(self, mock_strip_think_blocks):
        # Setup mock for strip_think_blocks
        mock_strip_think_blocks.side_effect = lambda text: text
        
        # Test extracting JSON from braces
        test_response = """
        Here's the result:
        {"events": [{"title": "Test", "summary": "Summary"}]}
        """
        result = _extract_json_from_response(test_response)
        self.assertEqual(result, {"events": [{"title": "Test", "summary": "Summary"}]})
        mock_strip_think_blocks.assert_called_once()

    @patch('event_research.services.discovery.strip_think_blocks')
    def test_extract_json_from_response_no_json(self, mock_strip_think_blocks):
        # Setup mock for strip_think_blocks
        mock_strip_think_blocks.side_effect = lambda text: text
        
        # Test when no JSON is found
        test_response = "No JSON here, just text."
        with self.assertRaises(ValueError):
            _extract_json_from_response(test_response)
        mock_strip_think_blocks.assert_called_once()


if __name__ == '__main__':
    unittest.main() 