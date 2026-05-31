# routers/ingest.py
#
# PDF upload + delete endpoints.
# Added: GET /documents — returns list of all stored documents.

import logging
from fastapi import APIRouter, UploadFile, File, Request, HTTPException
from models import IngestResponse
from services.pdf_parser import parse_pdf
from services.embedder import get_embeddings_batch
from services.vector_store import store_chunks, list_documents, delete_document

logger = logging.getLogger("documind")
router = APIRouter()


@router.post("/upload", response_model=IngestResponse)
async def upload_pdf(request: Request, file: UploadFile = File(...)):
    if file.content_type != "application/pdf":
        raise HTTPException(status_code=400, detail=f"Only PDF files accepted. Got: {file.content_type}")

    file_bytes = await file.read()

    if len(file_bytes) == 0:
        raise HTTPException(status_code=400, detail="Uploaded file is empty")

    if len(file_bytes) > 50 * 1024 * 1024:
        raise HTTPException(status_code=400, detail="File too large. Maximum size is 50MB")

    logger.info("PDF upload: %s (%d bytes)", file.filename, len(file_bytes))

    chunks = parse_pdf(file_bytes, file.filename)

    if not chunks:
        raise HTTPException(status_code=422, detail="PDF contained no extractable text")

    texts = [chunk.text for chunk in chunks]

    try:
        embeddings = get_embeddings_batch(texts)
    except RuntimeError as e:
        raise HTTPException(status_code=503, detail=str(e))

    collection = request.app.state.chroma_collection
    chunks_stored = store_chunks(collection, chunks, embeddings)

    logger.info("Ingested '%s': %d chunks", file.filename, chunks_stored)

    return IngestResponse(
        message="PDF successfully processed",
        filename=file.filename,
        chunks_stored=chunks_stored,
    )


@router.get("/documents")
async def get_documents(request: Request):
    """
    Returns all documents currently stored in ChromaDB.
    Called on frontend load so the sidebar is populated even after a refresh.
    """
    collection = request.app.state.chroma_collection
    docs = list_documents(collection)
    return {"documents": docs}


@router.delete("/delete/{filename}")
async def delete_pdf(filename: str, request: Request):
    collection = request.app.state.chroma_collection
    count = delete_document(collection, filename)

    if count == 0:
        raise HTTPException(status_code=404, detail=f"No document found: '{filename}'")

    return {"message": f"Deleted {count} chunks for '{filename}'"}