#!/usr/bin/env python3
import unittest
import os
import sys

# Add project root to path so imports work properly
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

if __name__ == '__main__':
    # Discover all tests in the tests directory
    test_suite = unittest.defaultTestLoader.discover(
        start_dir=os.path.dirname(__file__), 
        pattern='test_*.py'
    )
    
    # Run the tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(test_suite)
    
    # Exit with non-zero code if tests failed
    sys.exit(not result.wasSuccessful()) 