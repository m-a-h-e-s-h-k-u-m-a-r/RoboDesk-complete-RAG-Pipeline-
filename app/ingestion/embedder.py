"""Embedding model wrapper for RoboDesk."""

import os
from langchain_openai import OpenAIEmbeddings


def get_embeddings():
    """Return the configured embedding model."""
    api_key = os.getenv("OPENAI_API_KEY")
    model = os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small")
    embeddings = OpenAIEmbeddings(openai_api_key=api_key, model=model)
    print(f"  Embedding model: {model}")
    return embeddings
