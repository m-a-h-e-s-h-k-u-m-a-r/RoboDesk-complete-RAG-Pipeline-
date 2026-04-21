"""Semantic retrieval service for RoboDesk."""

import os
from typing import List, Dict, Any
from langchain_chroma import Chroma
from app.ingestion.embedder import get_embeddings
from app.ingestion.indexer import COLLECTION_NAME


def get_retrieval_vectorstore():
    """Connect to the existing Chroma collection."""
    embeddings = get_embeddings()
    chroma_dir = os.getenv("CHROMA_DB_DIR", "./chroma_db")
    return Chroma(
        collection_name=COLLECTION_NAME,
        embedding_function=embeddings,
        persist_directory=chroma_dir,
    )


def retrieve(query: str, top_k: int = 5) -> List[Dict[str, Any]]:
    """
    Perform cosine similarity search and return structured results.
    Each result contains: chunk_text, source, chunk_index, score.
    """
    vectorstore = get_retrieval_vectorstore()
    count = vectorstore._collection.count()

    if count == 0:
        return []

    # Clamp top_k to available documents
    k = min(top_k, count)
    results = vectorstore.similarity_search_with_score(query, k=k)

    output = []
    for doc, score in results:
        output.append({
            "chunk_text": doc.page_content,
            "source": doc.metadata.get("source", "unknown"),
            "chunk_index": doc.metadata.get("chunk_index", 0),
            "document_type": doc.metadata.get("document_type", "unknown"),
            "score": float(score),
        })

    return output
