"""Shared helper utilities."""

import os

from dotenv import load_dotenv

load_dotenv()


def get_api_key() -> str:
    """Return the Anthropic API key from the environment."""
    return os.getenv("ANTHROPIC_API_KEY", "")
