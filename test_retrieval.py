#!/usr/bin/env python3
"""
test_retrieval.py — Retrieval validation script for RoboDesk.
Tests at least 5 representative customer queries and prints top-3 results.
"""

import sys
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()
sys.path.insert(0, str(Path(__file__).parent))

from app.retrieval.retriever import retrieve

TEST_QUERIES = [
    "What is the payload capacity of the AX-200?",
    "How do I fix error E002 collision detected?",
    "What does the standard warranty cover?",
    "Does the AX-100 require safety fencing?",
    "How long does delivery take for a new robot?",
    "What programming languages are supported by Axiom robots?",
    "How do I contact emergency support?",
]


def main():
    print("=" * 60)
    print("RoboDesk — Retrieval Validation")
    print("=" * 60)

    for i, query in enumerate(TEST_QUERIES, 1):
        print(f"\n[Query {i}] {query}")
        print("-" * 50)

        results = retrieve(query, top_k=3)

        if not results:
            print("  No results found.")
            continue

        for rank, r in enumerate(results, 1):
            source_name = Path(r["source"]).name
            print(f"  Rank {rank} | Score: {r['score']:.4f} | Source: {source_name} | Chunk: {r['chunk_index']}")
            preview = r["chunk_text"][:200].replace("\n", " ")
            print(f"  Preview: {preview}...")

    print("\n" + "=" * 60)
    print("Validation complete.")


if __name__ == "__main__":
    main()
