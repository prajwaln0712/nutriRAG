"""RAG pipeline: orchestrates retrieval and generation with Claude."""

import os
import sys

import anthropic
from dotenv import load_dotenv

# Make the project root importable so `rag.retriever` and `utils.prompts`
# resolve whether this file is run directly or imported as a package.
_BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _BASE_DIR not in sys.path:
    sys.path.insert(0, _BASE_DIR)

try:
    from rag.retriever import Retriever
except ImportError:
    from retriever import Retriever

from utils.prompts import SYSTEM_PROMPT, RAG_PROMPT_TEMPLATE


# The Claude model and per-answer output cap used by the pipeline.
MODEL = "claude-sonnet-4-6"
MAX_TOKENS = 600


class RAGPipeline:
    """End-to-end retrieval-augmented generation pipeline."""

    def __init__(self, retriever=None, client=None):
        # The retriever produces the context; default to a fresh one.
        self.retriever = retriever or Retriever()
        # Load .env and create the Anthropic client (reads ANTHROPIC_API_KEY).
        load_dotenv()
        self.client = client or anthropic.Anthropic()

    def run(self, query: str) -> str:
        """Retrieve relevant context and generate an answer with Claude."""
        # 1. Retrieve context - an index match or a live USDA fallback.
        retrieval = self.retriever.retrieve(query)
        results = retrieval.get("results", [])

        # 2. Join the retrieved texts into a single context string, prefixing
        #    each with its source/reference so Claude can attribute or cite it.
        context = "\n\n".join(
            f"[{r.get('source', '')} {r.get('ref', '')}] {r['text']}".strip()
            for r in results
        ) if results else ""

        # 3. Fill the prompt template with the context and the question.
        user_prompt = RAG_PROMPT_TEMPLATE.format(context=context, question=query)

        # 4. Call Claude with the system prompt + the built user prompt.
        try:
            response = self.client.messages.create(
                model=MODEL,
                max_tokens=MAX_TOKENS,
                system=SYSTEM_PROMPT,
                messages=[{"role": "user", "content": user_prompt}],
            )
        except anthropic.APIError as exc:
            # Network, rate-limit, or server errors - fail gracefully.
            print(f"[ERROR] Claude API call failed: {exc}")
            return "Sorry, I couldn't generate an answer right now. Please try again."

        # 5. Return the answer text (the first text block of the response).
        return next(
            (block.text for block in response.content if block.type == "text"),
            "",
        )
    

# def _smoke_test():
#     """Ask one question end-to-end (needs a built index and a valid API key)."""
#     pipeline = RAGPipeline()
#     question = "Give me a vegetarian source of fibre"
#     print(f"Q: {question}")
#     print(f"A: {pipeline.run(question)}")


if __name__ == "__main__":
    # Read the question from the command line:
    #   python rag/pipeline.py "How much protein is in chicken breast?"
    if len(sys.argv) > 1:
        question = " ".join(sys.argv[1:])
        print(f"Q: {question}")
        print(f"A: {RAGPipeline().run(question)}")
    else:
        print('Usage: python rag/pipeline.py "your question here"')
