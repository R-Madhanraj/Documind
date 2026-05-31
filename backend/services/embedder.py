import os
import logging
import google.generativeai as genai
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger("documind")

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    raise RuntimeError(
        "GEMINI_API_KEY not found in environment. "
        "Add it to your backend/.env file."
    )

genai.configure(api_key=GEMINI_API_KEY)

EMBED_MODEL = "models/gemini-embedding-001"

def get_embedding(text: str, task_type: str = "retrieval_document") -> list[float]:
    try:
        result = genai.embed_content(
            model=EMBED_MODEL,
            content=text,
        )
        return result["embedding"]

    except Exception as e:
        error_msg = str(e)
        if "API_KEY" in error_msg or "api key" in error_msg.lower():
            raise RuntimeError("Invalid Gemini API key. Check your .env file.")
        if "quota" in error_msg.lower():
            raise RuntimeError("Gemini API quota exceeded. Free tier: 1500 requests/day.")
        raise RuntimeError(f"Gemini embedding failed: {error_msg}")

def get_embeddings_batch(texts: list[str]) -> list[list[float]]:
    """
    Embeds a list of texts (PDF chunks) for storage in ChromaDB.

    All chunks use task_type="retrieval_document" because they're
    going into the vector store to be retrieved later.

    Why loop instead of batch API?
    Gemini's embed_content does support batching but has a limit of
    100 texts per request. Looping is simpler and the free tier
    rate limit (15 req/min) is the real bottleneck anyway.
    We add a small delay if we're processing a large document.
    """
    import time

    embeddings = []

    for i, text in enumerate(texts):
        logger.debug("Embedding chunk %d/%d", i + 1, len(texts))

        embedding = get_embedding(text, task_type="retrieval_document")
        embeddings.append(embedding)

        # Gemini free tier: 15 requests per minute = 1 request per 4 seconds.
        # We add a small sleep every 10 chunks to stay under the limit.
        # This only matters for large PDFs (100+ chunks).
        if (i + 1) % 10 == 0:
            logger.debug("Pausing briefly to respect Gemini rate limits...")
            time.sleep(2)

    logger.info("Generated %d embeddings via Gemini", len(embeddings))
    return embeddings