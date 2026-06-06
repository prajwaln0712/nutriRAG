"""Retriever combining the embedder and vector store."""


class Retriever:
    """Retrieves relevant documents for a query."""

    def __init__(self, embedder=None, vectorstore=None):
        self.embedder = embedder
        self.vectorstore = vectorstore

    def retrieve(self, query: str, k: int = 5):
        """Embed the query and return the top-k documents."""
        query_embedding = self.embedder.embed([query])
        return self.vectorstore.search(query_embedding, k=k)
