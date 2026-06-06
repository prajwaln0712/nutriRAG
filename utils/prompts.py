"""Prompt templates for the RAG pipeline."""

SYSTEM_PROMPT = (
    "You are NutriRAG, an AI nutrition assistant. Answer questions about "
    "food and nutrition using the provided context. If the context does not "
    "contain the answer, say so honestly."
)

RAG_PROMPT_TEMPLATE = (
    "Use the following context to answer the question.\n\n"
    "Context:\n{context}\n\n"
    "Question: {question}\n\n"
    "Answer:"
)
