"""Embedding model wrapper for the RAG pipeline.

Uses the sentence-transformers model `all-mpnet-base-v2` to turn text into
dense vectors that can be stored in / queried against the FAISS vector store.
"""

import numpy as np
from sentence_transformers import SentenceTransformer


# Default embedding model. all-mpnet-base-v2 produces 768-dimensional vectors
# and is a strong general-purpose sentence-embedding model.
DEFAULT_MODEL_NAME = "all-mpnet-base-v2"

# Process-wide cache of loaded models, keyed by model name. A model is loaded
# into memory only once and then reused by every Embedder and every embed()
# call. (sentence-transformers also caches the downloaded files on disk, so the
# model is downloaded from the internet only the very first time.)
_MODEL_CACHE = {}


def _get_model(model_name: str) -> SentenceTransformer:
    """Return the loaded model, loading and caching it only on first request."""
    if model_name not in _MODEL_CACHE:
        _MODEL_CACHE[model_name] = SentenceTransformer(model_name)
    return _MODEL_CACHE[model_name]


class Embedder:
    """Wraps a sentence-transformers model to produce embeddings."""

    def __init__(self, model_name: str = DEFAULT_MODEL_NAME):
        self.model_name = model_name

    @property
    def model(self) -> SentenceTransformer:
        # Reuse the process-wide cached model (loaded once, reused thereafter).
        return _get_model(self.model_name)

    def embed(self, texts):
        """Convert text into embedding vectors.

        `texts` may be a single string (one line) or a list of strings (for
        example the food descriptions from food_item_list.json). Returns a
        2D float32 numpy array of shape (n_texts, embedding_dim) - even a
        single string yields a (1, dim) array, which is what FAISS expects.
        """
        # Accept a single string by wrapping it in a list, so the output is
        # always 2D and consistent for downstream code (e.g. FAISS).
        if isinstance(texts, str):
            texts = [texts]

        # Encode the texts into vectors. FAISS requires float32, so cast to it.
        embeddings = self.model.encode(texts, convert_to_numpy=True)
        return embeddings.astype(np.float32)


def _smoke_test():
    """Quick check that embedding works and the model is reused, not reloaded."""
    embedder = Embedder()
    print(f"Loading model '{embedder.model_name}' (downloads on first run)...")

    # 1) A single line of text.
    single = embedder.embed("Orange has 47 calories and 0.9 g of protein.")
    print(f"Single string         -> shape {single.shape}, dtype {single.dtype}")

    # 2) A list of strings, like the food descriptions in the JSON.
    foods = [
        "Apple has 52 calories and 14 g of carbs.",
        "Chicken breast has 165 calories and 20 g of protein.",
        "Almonds have 600 calories and 50 g of fats.",
        "Peanut butter",
        "Icecream"
    ]
    batch = embedder.embed(foods)
    print(f"List of {len(foods)} strings    -> shape {batch.shape}, dtype {batch.dtype}")

    # Basic sanity checks: 2D output, one row per input, float32, same dim.
    assert single.shape[0] == 1
    assert batch.shape[0] == len(foods)
    assert single.shape[1] == batch.shape[1]
    assert batch.dtype == np.float32

    # Reuse check: a second Embedder shares the same cached model object,
    # confirming the model is loaded only once.
    assert Embedder().model is embedder.model
    print(f"OK: dimension {batch.shape[1]}, and model reused from cache.")


if __name__ == "__main__":
    _smoke_test()
