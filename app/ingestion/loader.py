"""Document loading and chunking for RoboDesk ingestion pipeline."""

import os
from pathlib import Path
from typing import List
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

CHUNK_SIZE = 800
CHUNK_OVERLAP = 150

# Map directory names to document_type metadata
DOCUMENT_TYPE_MAP = {
    "products": "product_catalog",
    "faq": "faq",
    "support": "support",
    "policies": "policy",
}


def load_documents(dataset_path: str) -> List[Document]:
    """Load all .txt files from the dataset directory tree."""
    dataset_path = Path(dataset_path)
    all_docs: List[Document] = []

    for subdir, doc_type in DOCUMENT_TYPE_MAP.items():
        subdir_path = dataset_path / subdir
        if not subdir_path.exists():
            print(f"  [WARN] Directory not found: {subdir_path}")
            continue

        loader = DirectoryLoader(
            str(subdir_path),
            glob="**/*.txt",
            loader_cls=TextLoader,
            loader_kwargs={"encoding": "utf-8"},
            show_progress=False,
        )
        docs = loader.load()

        # Attach document_type metadata
        for doc in docs:
            doc.metadata["document_type"] = doc_type

        print(f"  Loaded {len(docs)} file(s) from {subdir}/")
        all_docs.extend(docs)

    return all_docs


def chunk_documents(documents: List[Document]) -> List[Document]:
    """Split documents into chunks, preserving and enriching metadata."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        length_function=len,
    )

    chunks: List[Document] = []
    source_counts: dict = {}

    for doc in documents:
        doc_chunks = splitter.split_documents([doc])
        source = doc.metadata.get("source", "unknown")

        for idx, chunk in enumerate(doc_chunks):
            chunk.metadata["chunk_index"] = idx
            # Normalise source to relative path basename for readability
            chunk.metadata["source"] = source
            chunks.append(chunk)

        source_counts[source] = len(doc_chunks)
        print(f"  {Path(source).name}: {len(doc_chunks)} chunks")

    return chunks
