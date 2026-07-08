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
    from rag.vectorstore import VectorStore
except ImportError:
    from vectorstore import VectorStore

# The live USDA fallback - fetches a food and updates food_item_list.json.
from data.ingest import lookup_food_item


# Matches scoring below this cosine similarity are treated as "not relevant".
DEFAULT_THRESHOLD = 0.29


class Retriever:
    """Retrieves food info from the index, falling back to a live USDA lookup."""

    def __init__(self, vectorstore=None, threshold=DEFAULT_THRESHOLD):
        # The vector store handles embedding the query and loading the index.
        self.vectorstore = vectorstore or VectorStore()
        self.threshold = threshold

    def retrieve(self, query, k=5):
        """Return relevant index matches, or a live USDA lookup result.

        Searches the FAISS index and keeps matches whose cosine similarity is
        at or above the threshold. If at least one qualifies, those are
        returned. Otherwise a live USDA lookup is triggered (which updates
        food_item_list.json), the FAISS index is rebuilt, and the lookup
        result is returned.

        Returns a dict: {"source": "index" | "usda", "results": ...}.
        """
        # Search the vector store (embeds the query, loads the index from disk).
        matches = self.vectorstore.search(query, k=k)

        # Keep only the matches at or above the relevance threshold.
        relevant = [m for m in matches if m["cosine_similarity"] >= self.threshold]

        # Relevant results found in the index - return them.
        if relevant:
            return {"source": "index", "results": relevant}

        # Nothing relevant enough -> live USDA lookup, which updates the JSON.
        print(f"No index match >= {self.threshold:.3f} for '{query}'; "
              f"falling back to a live USDA lookup...")
        lookup = lookup_food_item(query)

        # A dict means a valid food was returned (and the JSON may have a new
        # entry). Rebuild the index so the new item is searchable next time.
        # A string ("Sorry network error" / "Did not find any such food item")
        # or None means nothing was added, so there is nothing to rebuild.
        if isinstance(lookup, dict):
            print("Rebuilding the FAISS index with the updated food list...")
            self.vectorstore.build_from_food_json()

        return {"source": "usda", "results": lookup}


def _smoke_test():
    """Demonstrate an index hit (no network). The fallback path hits USDA."""
    retriever = Retriever()

    query = "pizza"
    result = retriever.retrieve(query, k=3)
    print(f"Query: {query!r}  ->  source: {result['source']}")
    if result["source"] == "index":
        for m in result["results"]:
            print(f"  {m['cosine_similarity']:.3f}  {m['text'][:60]}...")

    # To test the fallback, try an out-of-domain query such as:
    #   retriever.retrieve("dragon fruit")
    # That triggers a live USDA call, appends to food_item_list.json, and
    # rebuilds the FAISS index.


if __name__ == "__main__":
    _smoke_test()
