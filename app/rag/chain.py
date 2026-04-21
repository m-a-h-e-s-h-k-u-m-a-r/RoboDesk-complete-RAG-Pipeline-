"""LCEL RAG chain for RoboDesk."""

import os
import re
from pathlib import Path
from typing import Dict, Any, List
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from app.retrieval.retriever import retrieve
from app.rag.prompt import RAG_PROMPT


def format_context(chunks: List[Dict[str, Any]]) -> str:
    """Format retrieved chunks into a numbered context block."""
    if not chunks:
        return "No relevant context found."
    parts = []
    for i, chunk in enumerate(chunks, 1):
        source_name = Path(chunk["source"]).name
        parts.append(f"[{i}] (Source: {source_name})\n{chunk['chunk_text']}")
    return "\n\n".join(parts)


def extract_sources(chunks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Extract unique source references from retrieved chunks."""
    seen = set()
    sources = []
    for chunk in chunks:
        key = (chunk["source"], chunk["chunk_index"])
        if key not in seen:
            seen.add(key)
            sources.append({
                "source_file": chunk["source"],
                "chunk_index": chunk["chunk_index"],
            })
    return sources


def run_rag_chain(question: str, top_k: int = 5) -> Dict[str, Any]:
    """
    Full RAG chain: retrieve → format → prompt → LLM → parse.
    Returns answer text and structured source citations.
    """
    # Step 1: Retrieve
    try:
        chunks = retrieve(question, top_k=top_k)
    except Exception as e:
        chunks = []
        print(f"[WARN] Retrieval error: {e}")

    # Step 2: Format context
    context = format_context(chunks)
    sources = extract_sources(chunks)

    # Step 3: Build LLM
    llm = ChatOpenAI(
        openai_api_key=os.getenv("OPENAI_API_KEY"),
        model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
        temperature=0.1,
        max_tokens=1024,
    )

    # Step 4: LCEL chain
    chain = RAG_PROMPT | llm | StrOutputParser()

    try:
        answer = chain.invoke({"context": context, "question": question})
    except Exception as e:
        answer = f"I encountered an error generating a response. Please contact support. (Error: {e})"

    return {
        "answer": answer,
        "sources": sources,
        "model_used": os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
    }
