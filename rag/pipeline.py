"""RAG pipeline: orchestrates retrieval and generation."""


class RAGPipeline:
    """End-to-end retrieval-augmented generation pipeline."""

    def __init__(self, retriever=None, llm=None):
        self.retriever = retriever
        self.llm = llm

    def run(self, query: str) -> str:
        """Retrieve relevant context and generate an answer."""
        raise NotImplementedError
