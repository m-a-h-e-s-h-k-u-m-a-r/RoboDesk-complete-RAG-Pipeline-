"""Vector store upsert logic for RoboDesk."""

import os
from typing import List
from langchain_core.documents import Document
from langchain_chroma import Chroma

COLLECTION_NAME = "roboDesk-kb"


def get_vectorstore(embeddings):
    """Return a Chroma vector store instance."""
    chroma_dir = os.getenv("CHROMA_DB_DIR", "./chroma_db")
    vectorstore = Chroma(
        collection_name=COLLECTION_NAME,
        embedding_function=embeddings,
        persist_directory=chroma_dir,
    )
    return vectorstore


def upsert_chunks(vectorstore, chunks: List[Document]) -> int:
    """
    Upsert chunks into the vector store.
    Uses deterministic IDs (source + chunk_index) to prevent duplicates.
    Returns the number of chunks indexed.
    """
    # Build deterministic IDs to enable upsert (no duplicates on re-run)
    ids = []
    for chunk in chunks:
        source = chunk.metadata.get("source", "unknown")
        idx = chunk.metadata.get("chunk_index", 0)
        # Sanitise: replace path separators and spaces
        safe_source = source.replace("/", "_").replace("\\", "_").replace(" ", "_")
        ids.append(f"{safe_source}__chunk_{idx}")

    # Chroma add_documents with explicit IDs performs upsert behaviour
    vectorstore.add_documents(documents=chunks, ids=ids)
    return len(chunks)


def get_indexed_count(vectorstore) -> int:
    """Return the number of documents currently in the collection."""
    return vectorstore._collection.count()
