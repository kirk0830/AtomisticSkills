import requests
from typing import Dict, Any, List, Optional
import logging

logger = logging.getLogger(__name__)


def query_openalex(
    query: str, limit: int = 10, sort: Optional[str] = "relevance"
) -> List[Dict[str, Any]]:
    """
    Search the OpenAlex API for scholarly works based on a search query.

    Args:
        query: The search term (e.g., "solid state battery LGPS").
        limit: Maximum number of results to return (max 200).
        sort: Sort strategy for results. Supported values:
            - "relevance" (default): most semantically relevant papers.
            - "citations": most cited papers (descending by cited_by_count).
            - "recent": most recently published papers.

    Returns:
        A list of dictionaries containing parsed metadata for each work.

    Raises:
        ValueError: If ``sort`` is not one of the supported strategies.
    """
    base_url = "https://api.openalex.org/works"

    import os

    openalex_email = os.getenv("OPENALEX_EMAIL")
    if not openalex_email:
        logger.info(
            "OPENALEX_EMAIL is not set. query_openalex will use the fallback email "
            "support@openalex.org. Setting OPENALEX_EMAIL=your_email@example.com is "
            "recommended to use the polite pool and comply with OpenAlex usage guidelines "
            "(https://openalex.org/). See docs/api_key_guide.md."
        )
        openalex_email = "support@openalex.org"

    sort_mapping = {
        "relevance": "relevance_score:desc",
        "citations": "cited_by_count:desc",
        "recent": "publication_date:desc",
    }

    if sort not in sort_mapping:
        raise ValueError(
            f"Unsupported sort strategy '{sort}'. "
            f"Supported values: {', '.join(sort_mapping.keys())}."
        )

    # We use the search parameter for full-text and title search
    params = {
        "search": query,
        "per-page": min(limit, 200),
        # Sort by the chosen strategy
        "sort": sort_mapping[sort],
        # Access the Polite Pool for faster/more reliable responses
        "mailto": openalex_email,
    }

    try:
        response = requests.get(base_url, params=params, timeout=30)
        response.raise_for_status()
        data = response.json()

        results = []
        for work in data.get("results", []):
            # Extract authors
            authors = []
            for authorship in work.get("authorships", []):
                author = authorship.get("author", {})
                name = author.get("display_name")
                if name:
                    authors.append(name)

            # Extract primary concepts
            concepts = []
            for concept in work.get("concepts", []):
                if concept.get("score", 0) > 0.6:  # Only highly relevant concepts
                    concepts.append(concept.get("display_name"))

            # Check open access
            open_access = work.get("open_access", {})

            # Parse useful metadata
            parsed_work = {
                "id": work.get("id", "").replace("https://openalex.org/", ""),
                "title": work.get("title", "Untitled"),
                "publication_year": work.get("publication_year"),
                "doi": work.get("doi", "").replace("https://doi.org/", "")
                if work.get("doi")
                else None,
                "authors": authors,
                "cited_by_count": work.get("cited_by_count", 0),
                "is_oa": open_access.get("is_oa", False),
                "oa_url": open_access.get("oa_url"),
                "abstract_inverted_index": work.get(
                    "abstract_inverted_index"
                ),  # Raw format, needs reconstruction
                "concepts": concepts[:5],  # Top 5 concepts
                "type": work.get("type", "unknown"),
            }
            results.append(parsed_work)

        return results

    except requests.exceptions.RequestException as e:
        logger.error(f"Error querying OpenAlex API: {e}")
        return []


def reconstruct_abstract(
    abstract_inverted_index: Optional[Dict[str, List[int]]],
) -> str:
    """
    OpenAlex returns abstracts as an inverted index to avoid copyright issues.
    This function reconstructs the abstract string from the index.
    """
    if not abstract_inverted_index:
        return "No abstract available."

    try:
        # Find the maximum index to size the array
        max_idx = 0
        for indices in abstract_inverted_index.values():
            if indices:
                max_idx = max(max_idx, max(indices))

        # Create an array of correct length initialized with empty strings
        words = [""] * (max_idx + 1)

        # Fill the array
        for word, indices in abstract_inverted_index.items():
            for idx in indices:
                words[idx] = word

        # Join into a string, filtering out any empty slots (though ideally there shouldn't be any)
        return " ".join(w for w in words if w)
    except Exception as e:
        logger.error(f"Error reconstructing abstract: {e}")
        return "Error reconstructing abstract."
