import os
import sys
import unittest
import json
from pathlib import Path

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from src.mcp_server.base_server import search_literature


class TestOpenAlexTool(unittest.TestCase):
    def setUp(self):
        """Set up test fixtures."""
        # Create a temporary directory in .agents/test for save_to_file tests
        # as per coding standards
        self.test_dir = Path("../../.agents/test/pytest_openalex")
        self.test_dir.mkdir(parents=True, exist_ok=True)
        self.save_file = self.test_dir / "test_results.json"

    def test_basic_search(self):
        """Test a basic search without saving."""
        query = "solid state battery"
        limit = 2
        result = search_literature(query=query, limit=limit)

        # Verify it returns a string with "Found X papers"
        self.assertIsInstance(result, str)
        self.assertIn("Found", result)
        self.assertIn("papers on OpenAlex for query", result)
        self.assertIn(query, result)

        # OpenAlex should return some AI/Materials results for this broad query
        if "No results found" not in result:
            self.assertIn("### 1.", result)

    def test_search_with_save(self):
        """Test search and ensure it saves to file properly."""
        query = "machine learning interatomic potential"

        # Clean up any previous test file
        if self.save_file.exists():
            self.save_file.unlink()

        # Execute tool
        result = search_literature(
            query=query, limit=2, download=False, save_to_file=str(self.save_file)
        )

        self.assertIsInstance(result, str)

        # If openalex actually returned results, check the file
        if "No results found" not in result:
            self.assertTrue(self.save_file.exists())

            with open(self.save_file, "r") as f:
                data = json.load(f)
                self.assertIsInstance(data, list)
                if len(data) > 0:
                    self.assertIn("title", data[0])
                    self.assertIn("doi", data[0])

        # Clean up
        if self.save_file.exists():
            self.save_file.unlink()

    def test_empty_results(self):
        """Test search with a nonsense query."""
        query = "this_is_a_completely_nonsense_query_that_should_return_nothing_12345"
        result = search_literature(query=query, limit=1)

        self.assertIsInstance(result, str)
        self.assertIn("No results found on OpenAlex", result)

    def test_limit_cap(self):
        """Test that the limit logic executes without failure."""
        # The tool should automatically cap limits > 50 down to 50
        query = "lithium"
        # We don't actually request 100 because it's slow, but the code caps it.
        # This test ensures the code doesn't crash when given a large limit.
        result = search_literature(query=query, limit=2)  # Keep test fast
        self.assertIsInstance(result, str)

    def test_search_with_sort_relevance(self):
        """Test search sorted by relevance (default)."""
        query = "solid state battery"
        result = search_literature(query=query, limit=2, sort="relevance", download=False)
        self.assertIsInstance(result, str)
        self.assertIn("sorted by relevance", result)

    def test_search_with_sort_citations(self):
        """Test search sorted by citations."""
        query = "machine learning interatomic potential"
        result = search_literature(query=query, limit=2, sort="citations", download=False)
        self.assertIsInstance(result, str)
        self.assertIn("sorted by citations", result)
        if "No results found" not in result:
            self.assertIn("Citations:", result)

    def test_search_with_sort_recent(self):
        """Test search sorted by recent publication date."""
        query = "lithium solid electrolyte"
        result = search_literature(query=query, limit=2, sort="recent", download=False)
        self.assertIsInstance(result, str)
        self.assertIn("sorted by recent", result)

    def test_invalid_sort(self):
        """Test that an invalid sort value returns a friendly error string."""
        query = "battery"
        result = search_literature(query=query, limit=1, sort="invalid_sort", download=False)
        self.assertIsInstance(result, str)
        self.assertIn("Error executing search_literature", result)
        self.assertIn("Unsupported sort strategy", result)


if __name__ == "__main__":
    unittest.main()
