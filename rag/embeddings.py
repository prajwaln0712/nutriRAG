"""Embedding model wrapper for the RAG pipeline."""

from sentence_transformers import SentenceTransformer


class Embedder:
    """Wraps a sentence-transformers model to produce embeddings."""

    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self.model_name = model_name
        self._model = None

    @property
    def model(self) -> SentenceTransformer:
        if self._model is None:
            self._model = SentenceTransformer(self.model_name)
        return self._model

    def embed(self, texts):
        """Return embeddings for a list of texts."""
        return self.model.encode(texts)
