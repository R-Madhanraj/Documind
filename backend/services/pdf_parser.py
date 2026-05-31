
import fitz  # PyMuPDF — "fitz" is the legacy name, don't let it confuse you
import logging
from dataclasses import dataclass

logger = logging.getLogger("documind")

# ── Data structure ─────────────────────────────────────────────────────────────

@dataclass
class TextChunk:
    """
    A single chunk of text extracted from the PDF.

    We use a dataclass here (not a Pydantic model) because this is
    internal data — it never gets sent over the API. Pydantic models
    are for API boundaries. Dataclasses are for internal data structures.
    They're simpler and faster.

    text:        the actual content of this chunk
    page_number: which PDF page it came from (1-indexed, like real page numbers)
    chunk_index: position of this chunk within the entire document
                 (useful for debugging — "chunk 17 of 43")
    source:      the original filename, stored so we can show the user
                 which document the answer came from
    """
    text: str
    page_number: int
    chunk_index: int
    source: str


# ── Main function ──────────────────────────────────────────────────────────────

def parse_pdf(file_bytes: bytes, filename: str, chunk_size: int = 500, overlap: int = 50) -> list[TextChunk]:
    """
    Takes raw PDF bytes and returns a list of TextChunks.

    Why bytes instead of a file path?
    Because FastAPI gives us uploaded files as bytes in memory.
    We never save the PDF to disk — we process it on the fly and
    store only the chunks in ChromaDB. Less disk usage, simpler code.

    Parameters:
        file_bytes:  the raw PDF content (what FastAPI gives us on upload)
        filename:    original filename, stored in metadata for citations
        chunk_size:  how many characters per chunk (default 500)
                     500 chars ≈ 80-100 words ≈ a solid paragraph
        overlap:     how many characters to repeat between chunks (default 50)
                     50 chars ≈ 1-2 sentences of overlap

    Returns:
        list of TextChunk objects ready to be embedded and stored
    """

    # ── Step 1: Open the PDF from bytes ───────────────────────────────────────

    try:
        pdf_document = fitz.open(stream=file_bytes, filetype="pdf")
    except Exception as e:

        raise ValueError(f"Could not open PDF '{filename}': {e}")

    total_pages = len(pdf_document)
    logger.info("Opened PDF '%s' — %d pages", filename, total_pages)

    if total_pages == 0:
        raise ValueError(f"PDF '{filename}' has no pages")

    # ── Step 2: Extract text from each page ───────────────────────────────────

    pages_text = []

    for page_num in range(total_pages):
        page = pdf_document[page_num]


        text = page.get_text("text")

        text = text.strip()

        if not text:

            logger.warning("Page %d of '%s' has no extractable text — skipping", page_num + 1, filename)
            continue

        pages_text.append((page_num + 1, text))

    pdf_document.close()  

    if not pages_text:
        raise ValueError(
            f"No extractable text found in '{filename}'. "
            "This may be a scanned PDF — OCR support is not yet implemented."
        )

    logger.info("Extracted text from %d pages in '%s'", len(pages_text), filename)

    # ── Step 3: Chunk each page's text ────────────────────────────────────────

    all_chunks = []
    chunk_index = 0

    for page_number, page_text in pages_text:
        page_chunks = _chunk_text(page_text, chunk_size, overlap)

        for chunk_text in page_chunks:
            all_chunks.append(TextChunk(
                text=chunk_text,
                page_number=page_number,
                chunk_index=chunk_index,
                source=filename,
            ))
            chunk_index += 1

    logger.info(
        "Chunked '%s' into %d chunks (size=%d, overlap=%d)",
        filename, len(all_chunks), chunk_size, overlap
    )

    return all_chunks


# ── Chunking helper ────────────────────────────────────────────────────────────

def _chunk_text(text: str, chunk_size: int, overlap: int) -> list[str]:
    """
    Splits a string into overlapping chunks of chunk_size characters.

    Why a separate function?
    Because chunking logic is easy to get wrong and easy to test in isolation.
    We can write a unit test for just this function without needing a PDF.

    How it works (example with chunk_size=20, overlap=5):

    Text:     "The quick brown fox jumps over the lazy dog"
    Chunk 1:  "The quick brown fox " (chars 0-19)
    Chunk 2:  "fox jumps over the l" (chars 15-34)  ← starts 5 chars back
    Chunk 3:  "the lazy dog"         (chars 29-end)  ← starts 5 chars back

    The overlap (5 chars) means chunk 2 starts at position 15, not 20.
    "fox" appears in both chunk 1 and chunk 2 — no sentence gets cut off
    at a boundary without context on either side.

    Why character-based and not word or sentence based?
    Simpler and more predictable. Word-based chunking requires a tokenizer.
    Sentence-based requires a sentence detector. Character-based just works,
    and 500 characters is a consistent, predictable size for embeddings.
    For a first version this is the right tradeoff.
    """
    chunks = []
    start = 0

    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end]

        if len(chunk.strip()) > 20:
            chunks.append(chunk.strip())

        start += chunk_size - overlap

    return chunks