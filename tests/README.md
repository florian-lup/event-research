# Event Research Tests

This directory contains unit tests for the Event Research application.

## Test Structure

The tests are organized by module, following the same structure as the main application:

- `test_embeddings.py` - Tests for embedding generation and text chunking
- `test_deduplication.py` - Tests for duplicate detection logic
- `test_research.py` - Tests for event research and enrichment
- `test_discovery.py` - Tests for event discovery via Perplexity API
- `test_storage.py` - Tests for storing events in Pinecone and MongoDB
- `test_event_pipeline.py` - End-to-end tests for the entire event pipeline workflow

## Running Tests

### Run all tests

```bash
python tests/run_tests.py
```

### Run a specific test file

```bash
python tests/test_embeddings.py
python tests/test_deduplication.py
python tests/test_research.py
python tests/test_discovery.py
python tests/test_storage.py
python tests/test_event_pipeline.py
```

## Adding New Tests

When adding new functionality to the application, please add corresponding tests following these guidelines:

1. Create a new test file named `test_yourmodule.py` for new modules
2. Use meaningful test method names that describe what is being tested
3. Mock external dependencies (API calls, etc.) to ensure tests run quickly and reliably
4. Follow the established pattern of setting up test data in `setUp()`

## Code Coverage

To run tests with code coverage (requires the `coverage` package):

```bash
# Install coverage if not already installed
pip install coverage

# Run tests with coverage
coverage run tests/run_tests.py

# Generate coverage report
coverage report
``` 