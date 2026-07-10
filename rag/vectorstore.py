"""FAISS-backed vector store with cosine similarity and on-disk persistence.

Builds an IndexFlatIP index from text embeddings (produced by Embedder),
stores a positional id->text mapping in a JSON file, and persists both to disk.
A search query vector is matched against the loaded index and the top-k texts
are returned together with their cosine similarity.
"""

import os
import json

import faiss
import numpy as np

# Support both `python rag/vectorstore.py` and `import rag.vectorstore`.
try:
    from rag.embeddings import Embedder
except ImportError:
    from embeddings import Embedder


# Default on-disk locations, under data/processed/ next to the food JSON.
_BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_PROCESSED_DIR = os.path.join(_BASE_DIR, "data", "processed")
DEFAULT_INDEX_PATH = os.path.join(_PROCESSED_DIR, "food_index.faiss")
DEFAULT_MAPPING_PATH = os.path.join(_PROCESSED_DIR, "food_index_mapping.json")
DEFAULT_FOOD_JSON_PATH = os.path.join(_PROCESSED_DIR, "food_item_list.json")


class VectorStore:
    """A cosine-similarity FAISS store that persists its index and id->text map."""

    def __init__(self, embedder=None, index_path=DEFAULT_INDEX_PATH,
                 mapping_path=DEFAULT_MAPPING_PATH):
        # Reuse a shared Embedder (model is cached/loaded only once).
        self.embedder = embedder or Embedder()
        self.index_path = index_path
        self.mapping_path = mapping_path
        self.index = None    # the FAISS index (built or loaded from disk)
        self.mapping = {}    # position (int) -> text (str)

    def build(self, texts):
        """Embed `texts`, build a cosine FAISS index, and save it to disk."""
        # normalize=True gives unit vectors, so inner product == cosine similarity.
        embeddings = self.embedder.embed(texts, normalize=True)  # (n, dim) float32

        # IndexFlatIP = inner product; on unit vectors that is cosine similarity.
        dim = embeddings.shape[1]
        self.index = faiss.IndexFlatIP(dim)
        self.index.add(embeddings)

        # Positional mapping: row i in the index corresponds to texts[i].
        self.mapping = {i: text for i, text in enumerate(texts)}

        self.save()
        return self

    def build_from_food_json(self, json_path=DEFAULT_FOOD_JSON_PATH):
        """Build the index from the food items JSON produced by ingest.py.

        Each food's nutrition `description` and richer `details` are combined
        into one string before embedding, so the vectors capture both the
        numbers and the category/colour/texture/use facts.
        """
        with open(json_path, "r", encoding="utf-8") as f:
            items = json.load(f)

        # Embed description + details together (details is optional per item).
        texts = [
            f"{item['description']} {item.get('details', '')}".strip()
            for item in items
        ]
        return self.build(texts)

    def save(self):
        """Write the FAISS index and the id->text mapping JSON to disk."""
        os.makedirs(os.path.dirname(self.index_path), exist_ok=True)
        faiss.write_index(self.index, self.index_path)
        with open(self.mapping_path, "w", encoding="utf-8") as f:
            json.dump(self.mapping, f, indent=2, ensure_ascii=False)

    def load(self):
        """Load the FAISS index and mapping from disk into memory."""
        self.index = faiss.read_index(self.index_path)
        with open(self.mapping_path, "r", encoding="utf-8") as f:
            raw = json.load(f)
        # JSON object keys are strings; convert them back to int positions.
        self.mapping = {int(k): v for k, v in raw.items()}
        return self

    def search(self, query, k=5):
        """Return the top-k texts and cosine similarities for a query.

        `query` may be a string (it gets embedded) or a precomputed vector.
        Loads the index and mapping from disk first if they are not in memory.
        """
        # Make sure the index and mapping are available.
        if self.index is None or not self.mapping:
            self.load()

        # Turn the query into a normalized (1, dim) float32 vector.
        if isinstance(query, str):
            query_vec = self.embedder.embed(query, normalize=True)   # (1, dim)
        else:
            query_vec = np.asarray(query, dtype=np.float32)
            if query_vec.ndim == 1:
                query_vec = query_vec.reshape(1, -1)
            faiss.normalize_L2(query_vec)   # raw vector: normalize for cosine

        # With unit vectors + IndexFlatIP, the returned score is cosine similarity.
        scores, indices = self.index.search(query_vec, k)

        # Build readable results, skipping the -1 padding FAISS uses when k > n.
        results = []
        for position, score in zip(indices[0], scores[0]):
            if position == -1:
                continue
            results.append({
                "text": self.mapping.get(int(position), ""),
                "cosine_similarity": float(score),
            })
        return results


# def _smoke_test():
#     """Build the index from the food JSON (description + details) and search."""
#     store = VectorStore()
#     if os.path.exists(DEFAULT_FOOD_JSON_PATH):
#         store.build_from_food_json()
#     else:
#         # Fallback samples if the food JSON has not been generated yet.
#         store.build([
#             "Apple has 52 calories and 14 g of carbs.",
#             "Chicken breast has 165 calories and 20 g of protein.",
#             "Almonds have 600 calories and 50 g of fats.",
#         ])
#     print(f"Built index with {store.index.ntotal} vectors; saved to disk.")
#
#     # Search with a fresh store to prove it loads from disk.
#     results = VectorStore().search("food items which are meat", k=5)
#     print("Top matches for 'food items which are meat':")
#     for r in results:
#         print(f"  {r['cosine_similarity']:.3f}  {r['text']}")
#

if __name__ == "__main__":
    pass
    # _smoke_test()
