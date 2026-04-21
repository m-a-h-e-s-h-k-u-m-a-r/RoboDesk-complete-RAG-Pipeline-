# RoboDesk — Axiom Robotics AI Customer Service

A production-ready RAG system that answers customer support queries about Axiom Robotics products using a knowledge base of product catalogs, FAQs, troubleshooting guides, and warranty policies.

## Project Structure

```
capstone/
├── ingest.py                  # Standalone ingestion runner (Task 1)
├── test_retrieval.py          # Retrieval validation script (Task 2)
├── pyproject.toml
├── .env
├── dataset/
│   ├── products/robot_catalog.txt
│   ├── faq/axiom_faq.txt
│   ├── support/troubleshooting_guide.txt
│   └── policies/warranty_and_support_policy.txt
└── app/
    ├── main.py                # FastAPI entry point
    ├── ingestion/
    │   ├── loader.py          # Document loading & chunking
    │   ├── embedder.py        # Embedding model wrapper
    │   └── indexer.py         # Chroma upsert logic
    ├── retrieval/
    │   └── retriever.py       # Cosine similarity search
    ├── rag/
    │   ├── chain.py           # LCEL RAG chain
    │   └── prompt.py          # Prompt templates
    └── api/
        └── routes.py          # FastAPI route handlers
```

## Setup

```bash
uv sync
cp .env.example .env   # add your OPENAI_API_KEY
```

## Running

### Step 1 — Ingest the knowledge base
```bash
uv run python ingest.py
```

### Step 2 — Validate retrieval
```bash
uv run python test_retrieval.py
```

### Step 3 — Start the API server
```bash
uv run uvicorn app.main:app --host 0.0.0.0 --port 8000
```

## API Endpoints

| Method | Path        | Description                              |
|--------|-------------|------------------------------------------|
| GET    | `/health`   | Service status + indexed document count  |
| POST   | `/ingest`   | Trigger full ingestion pipeline          |
| POST   | `/query`    | RAG: retrieve + LLM answer + citations   |
| POST   | `/retrieve` | Retrieval only (no LLM)                  |

### POST /query
```json
{ "question": "What is the payload capacity of the AX-200?", "top_k": 5 }
```
Response:
```json
{
  "question": "What is the payload capacity of the AX-200?",
  "answer": "The payload capacity of the AX-200 is 5 kg.\n\nSources:\n[1] robot_catalog.txt\n[2] axiom_faq.txt",
  "sources": [
    { "source_file": "dataset/products/robot_catalog.txt", "chunk_index": 2 }
  ],
  "model_used": "gpt-4o-mini",
  "timestamp": "2026-04-19T19:32:04Z"
}
```

### POST /retrieve
```json
{ "query": "warranty coverage", "top_k": 3 }
```

### POST /ingest
```json
{}
```
Optional override: `{ "dataset_path": "/path/to/dataset" }`

## Design Decisions

- **Embeddings**: OpenAI `text-embedding-3-small` (1536 dims)
- **Vector store**: ChromaDB (local persistent), collection `roboDesk-kb`
- **Chunking**: 800 chars / 150 overlap with `RecursiveCharacterTextSplitter`
- **Upsert**: Deterministic IDs (`source__chunk_N`) prevent duplicates on re-ingestion
- **LLM**: `gpt-4o-mini`, temperature 0.1 for factual consistency
- **Hallucination prevention**: System prompt instructs LLM to answer only from context and say "I don't know" when context is insufficient
