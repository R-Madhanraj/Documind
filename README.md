# DocuMind — RAG-Powered PDF Intelligence Platform

Full-stack Retrieval-Augmented Generation (RAG) application for multi-document
PDF querying. Returns cited answers with page references and cosine similarity
scores at average 3.3s end-to-end latency.

## Architecture

**Ingestion Pipeline**
1. PDF parsed with PyMuPDF, split into overlapping text chunks
2. Each chunk embedded via Gemini Embedding API
3. Vectors stored in ChromaDB with page metadata

**Query Pipeline**
1. User query embedded with the same model
2. ChromaDB cosine similarity search retrieves top-k chunks (< 10ms)
3. Retrieved chunks + query passed to LLM with context-constrained prompt
4. Response returned with per-citation page number and similarity score

96% of end-to-end latency is LLM generation. Retrieval itself is sub-10ms.

**Hallucination Mitigation**
- Confidence threshold filtering: chunks below similarity cutoff are excluded
- Context-constrained prompting: LLM is instructed to answer only from
  provided context, not general knowledge
- Per-answer citations expose the exact source chunk, page, and score

## Stack
- **Backend**: FastAPI (Python)
- **Vector DB**: ChromaDB
- **Embeddings**: Gemini Embedding API
- **LLM**: Switchable via dynamic model selector
- **Frontend**: React + Vite

## Features
- Multi-document upload and vector storage
- Dynamic LLM model switching without restarting the server
- Per-answer citations: page number + cosine similarity score
- Sub-10ms retrieval profiled and verified

## Setup
```bash
# Backend
cd backend
pip install -r requirements.txt
uvicorn main:app --reload

# Frontend
cd frontend
npm install
npm run dev
```

Set your Gemini API key in the backend `.env` file.
