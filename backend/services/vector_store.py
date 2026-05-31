import logging
from fastapi import FastAPI
import chromadb

logger = logging.getLogger("documind")
COLLECTION_NAME = "documents"


def init_vector_store(chroma_client: chromadb.PersistentClient, app: FastAPI) -> None:
    collection = chroma_client.get_or_create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"}
    )
    app.state.chroma_collection = collection
    logger.info("Vector store ready — %d chunks in collection", collection.count())


def store_chunks(collection, chunks, embeddings: list[list[float]]) -> int:
    if len(chunks) != len(embeddings):
        raise ValueError("Chunks and embeddings count mismatch")

    ids, texts, metadatas = [], [], []

    for chunk, embedding in zip(chunks, embeddings):
        chunk_id = f"{chunk.source}__chunk_{chunk.chunk_index}"
        ids.append(chunk_id)
        texts.append(chunk.text)
        metadatas.append({
            "page_number": chunk.page_number,
            "source": chunk.source,
            "chunk_index": chunk.chunk_index,
        })

    collection.upsert(ids=ids, embeddings=embeddings, documents=texts, metadatas=metadatas)
    logger.info("Stored %d chunks", len(chunks))
    return len(chunks)


def search_similar(
    collection,
    query_embedding: list[float],
    top_k: int = 4,
    filter_source: str | None = None
) -> list[dict]:
    """
    Searches for similar chunks.
    If filter_source is set, only searches within that document.
    If None, searches across all documents.
    """
    where = {"source": filter_source} if filter_source else None

    kwargs = dict(
        query_embeddings=[query_embedding],
        n_results=min(top_k, collection.count()),
        include=["documents", "metadatas", "distances"],
    )
    if where:
        kwargs["where"] = where

    results = collection.query(**kwargs)

    documents = results["documents"][0]
    metadatas = results["metadatas"][0]
    distances = results["distances"][0]

    chunks = []
    for text, metadata, distance in zip(documents, metadatas, distances):
        chunks.append({
            "text": text,
            "page_number": metadata["page_number"],
            "source": metadata["source"],
            "score": round(1 - distance, 4),
        })

    return chunks


def list_documents(collection) -> list[dict]:
    """
    Returns all unique documents stored in ChromaDB with their chunk counts.
    Called on app load so the sidebar shows persisted documents immediately.

    We fetch all metadatas (no embeddings/documents — faster) and
    count chunks per source filename.
    """
    if collection.count() == 0:
        return []

    results = collection.get(include=["metadatas"])
    metadatas = results["metadatas"]

    # Count chunks per source
    doc_chunks: dict[str, int] = {}
    for meta in metadatas:
        source = meta.get("source", "unknown")
        doc_chunks[source] = doc_chunks.get(source, 0) + 1

    return [
        {"filename": source, "chunks": count}
        for source, count in sorted(doc_chunks.items())
    ]


def delete_document(collection, filename: str) -> int:
    """
    Deletes all chunks belonging to a specific document.
    Returns the number of chunks deleted.
    """
    results = collection.get(where={"source": filename})

    if not results["ids"]:
        return 0

    count = len(results["ids"])
    collection.delete(where={"source": filename})
    logger.info("Deleted %d chunks for '%s'", count, filename)
    return count