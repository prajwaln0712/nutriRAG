"""Retriever: searches the FAISS index, falling back to a live USDA lookup.

Given a user query, it searches the vector store and keeps the matches whose
cosine similarity meets a relevance threshold. If nothing is relevant enough,
it triggers a live USDA lookup (which appends the new food to
food_item_list.json), rebuilds the FAISS index so the new item is searchable,
and returns that result instead.
"""

import os
import sys

# Make the project root importable so `data.ingest` resolves whether this file
# is run directly (`python rag/retriever.py`) or imported as `rag.retriever`.
_BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _BASE_DIR not in sys.path:
    sys.path.insert(0, _BASE_DIR)

try:
    from rag.vectorstore import VectorStore, DGA_INDEX_PATH, DGA_MAPPING_PATH
except ImportError:
    from vectorstore import VectorStore, DGA_INDEX_PATH, DGA_MAPPING_PATH

# The live USDA fallback - fetches a food and updates food_item_list.json.
from data.ingest import lookup_food_item


# Matches scoring below this cosine similarity are treated as "not relevant".
DEFAULT_THRESHOLD = 0.57


class Retriever:
    """Retrieves food/guidance info from two indexes, with a live USDA fallback."""

    def __init__(self, food_store=None, dga_store=None, threshold=DEFAULT_THRESHOLD):
        # Two vector stores: USDA food facts and DGA guidance text.
        self.food_store = food_store or VectorStore()
        self.dga_store = dga_store or VectorStore(
            index_path=DGA_INDEX_PATH, mapping_path=DGA_MAPPING_PATH
        )
        self.threshold = threshold

    def _search_dga(self, query, k):
        """Search the DGA index, returning [] if it has not been built yet."""
        try:
            return self.dga_store.search(query, k=k)
        except (FileNotFoundError, RuntimeError):
            # The DGA index files do not exist yet - degrade gracefully.
            return []

    def retrieve(self, query, k=5):
        """Return relevant index matches, or a live USDA lookup result.

        Searches both the food and DGA indexes and keeps matches whose cosine
        similarity is at or above the threshold. If at least one qualifies, the
        top-k are returned. Otherwise a live USDA lookup is triggered (which
        updates food_item_list.json), the food index is rebuilt, and the lookup
        result is returned.

        Returns a dict: {"mode_of_retreive": "pre_built_idx" | "usda", "results": ...}.
        The top-level "mode_of_retreive" is the retrieval path (a pre-built index
        vs a live USDA lookup); each result carries its own "source" (USDA vs DGA).
        """
        # Search both indexes (each embeds the query and loads its own index).
        matches = self.food_store.search(query, k=k) + self._search_dga(query, k=k)

        # Keep matches at/above the threshold, best first, capped at k.
        relevant = [m for m in matches if m["cosine_similarity"] >= self.threshold]
        relevant.sort(key=lambda m: m["cosine_similarity"], reverse=True)
        relevant = relevant[:k]

        # Relevant results found in either index - return them.
        if relevant:
            return {"mode_of_retreive": "pre_built_idx", "results": relevant}

        # Nothing relevant enough -> live USDA lookup, which updates the JSON.
        print(f"No index match >= {self.threshold:.3f} for '{query}'; "
              f"falling back to a live USDA lookup...")
        lookup = lookup_food_item(query)

        # Normalize the USDA result into the SAME shape as the index results:
        # a list of dicts with "text", "source", "ref" and "cosine_similarity".
        if isinstance(lookup, dict):
            # Valid food (JSON may have a new entry): rebuild the food index so
            # it is searchable next time, expose the description as `text`, and
            # treat the direct lookup as an exact match (cosine similarity 1.0).
            print("Rebuilding the FAISS index with the updated food list...")
            self.food_store.build_from_food_json()
            results = [{
                "text": lookup.get("description", ""),
                "source": "USDA FoodData Central",
                "ref": lookup.get("query", ""),
                "cosine_similarity": 1.0,
            }]
        else:
            # A string message or None (network failure / not found): no match.
            results = [{
                "text": "Sorry could not find any data",
                "source": "",
                "ref": "",
                "cosine_similarity": 0.0,
            }]

        return {"mode_of_retreive": "usda", "results": results}


def _smoke_test():
    """Demonstrate an index hit (no network). The fallback path hits USDA."""
    retriever = Retriever()

    query = "What should I look out for in snack food"
    result = retriever.retrieve(query, k=10)
    print(f"Query: {query!r}  ->  mode_of_retreive: {result['mode_of_retreive']}")
    if result["mode_of_retreive"] == "pre_built_idx":
        for m in result["results"]:
            print(f"  {m['cosine_similarity']:.3f}  {m['text'][:60]}...")

    # To test the fallback, try an out-of-domain query such as:
    #   retriever.retrieve("dragon fruit")
    # That triggers a live USDA call, appends to food_item_list.json, and
    # rebuilds the FAISS index.


if __name__ == "__main__":
    # pass
    _smoke_test()
