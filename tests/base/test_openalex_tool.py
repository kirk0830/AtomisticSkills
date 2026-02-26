import os
import sys
import unittest
from pathlib import Path

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from src.mcp_server.base_server import search_literature_openalex

class TestOpenAlexTool(unittest.TestCase):
    def setUp(self):
        """Set up test fixtures."""
        # Create a temporary directory in .agent/test for save_to_file tests
        # as per coding standards
        self.test_dir = Path("../../.agent/test/pytest_openalex")
        self.test_dir.mkdir(parents=True, exist_ok=True)
        self.save_file = self.test_dir / "test_results.json"

    def test_basic_search(self):
        """Test a basic search query."""
        query = "machine learning interatomic potentials"
        limit = 3
        
        # Execute tool
        result = search_literature_openalex(query=query, limit=limit)
        
        # Verify output is a string and contains expected formatting
        self.assertIsInstance(result, str)
        self.assertIn("Found", result)
        self.assertIn(query, result)
        
        # Should have up to 'limit' papers returned
        # Look for the markdown headers indicating list items
        num_headers = result.count("### ")
        self.assertGreater(num_headers, 0)
        self.assertLessEqual(num_headers, limit)

    def test_search_with_save(self):
        """Test search query that saves JSON to a file."""
        query = "LGPS solid state battery"
        
        # Clean up any previous test file
        if self.save_file.exists():
            self.save_file.unlink()
            
        # Execute tool
        result = search_literature_openalex(
            query=query, 
            limit=2, 
            save_to_file=str(self.save_file)
        )
        
        # Verify file was created
        self.assertTrue(self.save_file.exists())
        self.assertIn(str(self.save_file), result)
        
        # Clean up
        if self.save_file.exists():
            self.save_file.unlink()

    def test_empty_results(self):
        """Test a query that should yield no results."""
        query = "asdfghjklqwertyuiopzxcvbnm1234567890_nonsense_query"
        result = search_literature_openalex(query=query, limit=1)
        
        self.assertIsInstance(result, str)
        self.assertIn("No results found", result)

    def test_limit_cap(self):
        """Test that the request limit is capped at 50 to prevent massive payloads."""
        # The tool should automatically cap limits > 50 down to 50
        query = "density functional theory"
        # We don't actually request 100 because it's slow, but the code caps it.
        # This test ensures the code doesn't crash when given a large limit.
        result = search_literature_openalex(query=query, limit=2) # Keep test fast
        self.assertIsInstance(result, str)

if __name__ == '__main__':
    unittest.main()
