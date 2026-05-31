import os
import logging
import google.generativeai as genai
from fastapi import APIRouter, Request, HTTPException
from dotenv import load_dotenv
from models import ChatRequest, ChatResponse, SourceChunk
from services.embedder import get_embedding
from services.vector_store import search_similar

load_dotenv()

logger = logging.getLogger("documind")
router = APIRouter()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    raise RuntimeError("GEMINI_API_KEY not found in environment.")

genai.configure(api_key=GEMINI_API_KEY)

GENERATION_CONFIG = {
    "temperature": 0.1,
    "max_output_tokens": 1024,
}

SYSTEM_PROMPT = """You are a helpful assistant that answers questions about PDF documents.

Rules you must follow:
1. ONLY use information from the context provided below. Do not use any outside knowledge.
2. Always cite the page number(s) where you found the information, like: (Page 4)
3. If the context doesn't contain enough information to answer, say:
   "I couldn't find relevant information in the document for this question."
4. Be concise and direct. Do not repeat the question back.
5. If information appears on multiple pages, cite all of them."""


def get_llm_answer(model_name: str, system_with_context: str, question: str) -> str:
    supported = ["gemini-2.5-flash", "gemini-2.5-pro", "gemini-2.0-flash"]
    if model_name not in supported:
        raise HTTPException(
            status_code=400,
            detail=f"Model '{model_name}' not available yet. Supported: {supported}"
        )

    try:
        model = genai.GenerativeModel(
            model_name=model_name,
            generation_config=GENERATION_CONFIG,
            system_instruction=system_with_context,
        )
        response = model.generate_content(question)
        return response.text

    except Exception as e:
        error_msg = str(e)
        if "quota" in error_msg.lower():
            raise HTTPException(status_code=429, detail="Gemini API quota exceeded.")
        if "API_KEY" in error_msg:
            raise HTTPException(status_code=401, detail="Invalid Gemini API key.")
        logger.exception("Gemini generation failed")
        raise HTTPException(status_code=503, detail=f"LLM error: {error_msg}")


@router.post("/ask", response_model=ChatResponse)
async def ask_question(body: ChatRequest, request: Request):
    # Validate question isn't empty — friendly error instead of 422
    if not body.question or not body.question.strip():
        raise HTTPException(status_code=400, detail="Please type a question first.")

    if len(body.question.strip()) < 2:
        raise HTTPException(status_code=400, detail="Question is too short — please add more detail.")

    logger.info("Question: '%s' | Model: %s | Filter: %s",
                body.question, body.model, body.filter_source)

    collection = request.app.state.chroma_collection

    if collection.count() == 0:
        raise HTTPException(status_code=400, detail="No documents uploaded yet.")

    try:
        query_embedding = get_embedding(body.question, task_type="retrieval_query")
    except RuntimeError as e:
        raise HTTPException(status_code=503, detail=str(e))

    # Pass filter_source — None means search all docs
    similar_chunks = search_similar(
        collection,
        query_embedding,
        top_k=4,
        filter_source=body.filter_source
    )

    if not similar_chunks:
        raise HTTPException(
            status_code=404,
            detail="No relevant content found. Try rephrasing or selecting a different document."
        )

    context_parts = []
    for chunk in similar_chunks:
        context_parts.append(
            f"[Page {chunk['page_number']} — {chunk['source']}]\n{chunk['text']}"
        )
    context = "\n\n---\n\n".join(context_parts)
    system_with_context = f"{SYSTEM_PROMPT}\n\nCONTEXT FROM DOCUMENT:\n{context}"

    answer_text = get_llm_answer(body.model, system_with_context, body.question)

    sources = [
        SourceChunk(page_number=c["page_number"], text=c["text"], score=c["score"])
        for c in similar_chunks
    ]

    logger.info("Answer generated — %d sources", len(sources))
    return ChatResponse(answer=answer_text, sources=sources)