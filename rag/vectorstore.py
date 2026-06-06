"""FAISS-backed vector store."""

import faiss


class VectorStore:
    """Simple FAISS vector store for dense retrieval."""

    def __init__(self, dim: int):
        self.dim = dim
        self.index = faiss.IndexFlatL2(dim)
        self.documents = []

    def add(self, embeddings, documents):
        """Add embeddings and their associated documents."""
        self.index.add(embeddings)
        self.documents.extend(documents)

    def search(self, query_embedding, k: int = 5):
        """Return the top-k most similar documents."""
        distances, indices = self.index.search(query_embedding, k)
        return [self.documents[i] for i in indices[0] if i < len(self.documents)]
