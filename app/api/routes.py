"""FastAPI route handlers for RoboDesk."""

import os
import time
from datetime import datetime, timezone
from typing import Optional
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from dotenv import load_dotenv

load_dotenv()

router = APIRouter()

# ── Request / Response models ──────────────────────────────────────────────

class IngestRequest(BaseModel):
    dataset_path: Optional[str] = None

class QueryRequest(BaseModel):
    question: str
    top_k: int = 5

class RetrieveRequest(BaseModel):
    query: str
    top_k: int = 5


# ── Helpers ────────────────────────────────────────────────────────────────

def _get_vectorstore():
    from app.ingestion.embedder import get_embeddings
    from app.ingestion.indexer import get_vectorstore
    return get_vectorstore(get_embeddings())


# ── Routes ─────────────────────────────────────────────────────────────────

@router.get("/health")
def health():
    """Return service status and vector store info."""
    try:
        vs = _get_vectorstore()
        count = vs._collection.count()
        vs_status = "connected"
    except Exception as e:
        count = 0
        vs_status = f"error: {e}"

    return {
        "status": "healthy",
        "vector_store": vs_status,
        "indexed_document_count": count,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }


@router.post("/ingest")
def ingest(request: IngestRequest = IngestRequest()):
    """Trigger the full ingestion pipeline."""
    from app.ingestion.loader import load_documents, chunk_documents
    from app.ingestion.embedder import get_embeddings
    from app.ingestion.indexer import get_vectorstore, upsert_chunks, get_indexed_count

    dataset_path = request.dataset_path or os.getenv(
        "DATASET_PATH",
        os.path.join(os.path.dirname(__file__), "..", "..", "..", "dataset"),
    )
    dataset_path = os.path.abspath(dataset_path)

    start = time.time()

    print(f"\n[Ingest] Loading documents from: {dataset_path}")
    documents = load_documents(dataset_path)
    if not documents:
        raise HTTPException(status_code=400, detail="No documents found at the specified path.")

    print("[Ingest] Chunking documents...")
    chunks = chunk_documents(documents)

    print("[Ingest] Indexing chunks...")
    embeddings = get_embeddings()
    vs = get_vectorstore(embeddings)
    upsert_chunks(vs, chunks)
    indexed = get_indexed_count(vs)

    duration = round(time.time() - start, 2)
    print(f"[Ingest] Done in {duration}s — {len(documents)} files, {len(chunks)} chunks, {indexed} indexed.")

    return {
        "files_loaded": len(documents),
        "chunks_created": len(chunks),
        "chunks_indexed": indexed,
        "duration_seconds": duration,
    }


@router.post("/query")
def query(request: QueryRequest):
    """Run the full RAG chain and return a grounded answer with citations."""
    from app.rag.chain import run_rag_chain

    result = run_rag_chain(request.question, top_k=request.top_k)

    return {
        "question": request.question,
        "answer": result["answer"],
        "sources": result["sources"],
        "model_used": result["model_used"],
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }


@router.post("/retrieve")
def retrieve_only(request: RetrieveRequest):
    """Return raw retrieved chunks without invoking the LLM."""
    from app.retrieval.retriever import retrieve

    chunks = retrieve(request.query, top_k=request.top_k)

    if not chunks:
        return {"query": request.query, "results": [], "message": "No relevant chunks found."}

    return {"query": request.query, "results": chunks}
