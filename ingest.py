#!/usr/bin/env python3
"""
ingest.py — Standalone ingestion runner for RoboDesk.
Loads, chunks, embeds, and indexes the knowledge base into Chroma.
"""

import os
import sys
import time
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

# Allow running from the capstone/ directory
sys.path.insert(0, str(Path(__file__).parent))

from app.ingestion.loader import load_documents, chunk_documents
from app.ingestion.embedder import get_embeddings
from app.ingestion.indexer import get_vectorstore, upsert_chunks, get_indexed_count

DATASET_PATH = os.path.join(os.path.dirname(__file__), "dataset")


def main():
    print("=" * 60)
    print("RoboDesk — Ingestion Pipeline")
    print("=" * 60)

    start = time.time()

    # Step 1: Load
    print("\n[Step 1] Loading documents...")
    documents = load_documents(DATASET_PATH)
    print(f"  Total documents loaded: {len(documents)}")

    if not documents:
        print("ERROR: No documents found. Check the dataset/ directory.")
        sys.exit(1)

    # Step 2: Chunk
    print("\n[Step 2] Chunking documents...")
    chunks = chunk_documents(documents)
    print(f"  Total chunks created: {len(chunks)}")

    # Step 3: Embed + Index
    print("\n[Step 3] Generating embeddings and indexing...")
    embeddings = get_embeddings()
    vs = get_vectorstore(embeddings)
    upsert_chunks(vs, chunks)
    indexed = get_indexed_count(vs)

    duration = round(time.time() - start, 2)

    # Summary report
    print("\n" + "=" * 60)
    print("INGESTION SUMMARY")
    print("=" * 60)
    print(f"  Files loaded:    {len(documents)}")
    print(f"  Chunks created:  {len(chunks)}")
    print(f"  Chunks indexed:  {indexed}")
    print(f"  Time taken:      {duration}s")
    print("=" * 60)

    if indexed != len(chunks):
        print(f"[WARN] Indexed count ({indexed}) differs from chunks created ({len(chunks)}).")
        print("       This may be expected if some chunks already existed (upsert).")


if __name__ == "__main__":
    main()
