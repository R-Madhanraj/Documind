import logging
import time
from contextlib import asynccontextmanager

import chromadb
from chromadb.config import Settings
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from routers import ingest, chat
from services.vector_store import init_vector_store

# ── Logging ────────────────────────────────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s — %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("documind")


# ── Lifespan: startup / shutdown ───────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Runs once on startup before the server accepts requests,
    and once on shutdown after the last request is handled.

    Startup responsibilities:
    - Initialise the persistent ChromaDB client and create the
      'documents' collection if it doesn't already exist.
    - Store the client on app.state so every request handler can
      reach it without importing a module-level singleton.

    We use FastAPI's lifespan context manager (preferred over the
    deprecated @app.on_event("startup") decorator).
    """
    logger.info("DocuMind starting up…")

    chroma_client = chromadb.PersistentClient(
    path="./chroma_data"
    )

    init_vector_store(chroma_client, app)
    logger.info("ChromaDB initialised — collection ready")

    yield  

    # Shutdown: flush any buffered writes to disk.
    logger.info("DocuMind shutting down — persisting ChromaDB…")
    chroma_client.persist()
    logger.info("Shutdown complete")


# ── App factory ────────────────────────────────────────────────────────────────

def create_app() -> FastAPI:
    """
    Creates and fully configures the FastAPI application instance.
    Separating app creation into a factory function makes the app
    trivially testable — tests can call create_app() without
    side effects from module-level imports.
    """
    app = FastAPI(
        title="DocuMind",
        description=(
            "Chat with your PDF documents. "
            "Upload a PDF, ask questions, get cited answers — "
            "all running locally via Ollama."
        ),
        version="1.0.0",
        lifespan=lifespan,
        # Disable the default /docs redirect so we can customise it
        docs_url="/api/docs",
        redoc_url="/api/redoc",
        openapi_url="/api/openapi.json",
    )

    _configure_cors(app)
    _mount_routers(app)
    _register_error_handlers(app)
    _add_request_logging(app)

    return app


# ── CORS ───────────────────────────────────────────────────────────────────────

def _configure_cors(app: FastAPI) -> None:
    """
    Allow the React dev server (Vite default: port 5173) to call the API.

    In production, replace the wildcard origin with your actual frontend domain.
    Never ship allow_origins=["*"] with allow_credentials=True — that's a
    security misconfiguration. We use explicit origins here.
    """
    app.add_middleware(
        CORSMiddleware,
        allow_origins=[
            "http://localhost:5173",   # Vite dev server
            "http://localhost:3000",   # CRA / alternative dev port
            "http://127.0.0.1:5173",
        ],
        allow_credentials=True,
        allow_methods=["GET", "POST", "DELETE", "OPTIONS"],
        allow_headers=["Content-Type", "Authorization", "X-Request-ID"],
    )


# ── Routers ────────────────────────────────────────────────────────────────────

def _mount_routers(app: FastAPI) -> None:
    """
    All routes are versioned under /api/v1 so we can introduce
    breaking changes in /api/v2 without removing the old endpoints.

    /api/v1/ingest  — PDF upload & ingestion pipeline
    /api/v1/chat    — RAG query & answer pipeline
    """
    app.include_router(
        ingest.router,
        prefix="/api/v1/ingest",
        tags=["Ingestion"],
    )
    app.include_router(
        chat.router,
        prefix="/api/v1/chat",
        tags=["Chat"],
    )


# ── Global error handlers ──────────────────────────────────────────────────────

def _register_error_handlers(app: FastAPI) -> None:
    """
    Catch-all handlers that convert unhandled exceptions into
    structured JSON responses instead of exposing raw stack traces.

    We distinguish between:
    - ValueError / TypeError  → 400 Bad Request (caller's fault)
    - Everything else         → 500 Internal Server Error (our fault)

    In both cases we log the full exception server-side but return
    only a safe, sanitised message to the client.
    """

    @app.exception_handler(ValueError)
    async def value_error_handler(request: Request, exc: ValueError):
        logger.warning("Bad request — %s %s: %s", request.method, request.url.path, exc)
        return JSONResponse(
            status_code=400,
            content={
                "error": "bad_request",
                "message": str(exc),
                "path": str(request.url.path),
            },
        )

    @app.exception_handler(FileNotFoundError)
    async def file_not_found_handler(request: Request, exc: FileNotFoundError):
        logger.warning("Resource not found — %s", exc)
        return JSONResponse(
            status_code=404,
            content={
                "error": "not_found",
                "message": str(exc),
            },
        )

    @app.exception_handler(Exception)
    async def generic_error_handler(request: Request, exc: Exception):
        # Log the full traceback so engineers can debug it,
        # but never send internal details to the client.
        logger.exception("Unhandled exception on %s %s", request.method, request.url.path)
        return JSONResponse(
            status_code=500,
            content={
                "error": "internal_server_error",
                "message": "An unexpected error occurred. Please try again.",
            },
        )


# ── Request logging middleware ─────────────────────────────────────────────────

def _add_request_logging(app: FastAPI) -> None:
    """
    Logs every request with method, path, status code, and duration.
    Useful for spotting slow endpoints during development.

    Example output:
        POST /api/v1/ingest/upload  →  201  in 843ms
    """

    @app.middleware("http")
    async def log_requests(request: Request, call_next):
        start = time.perf_counter()
        response = await call_next(request)
        duration_ms = (time.perf_counter() - start) * 1000
        logger.info(
            "%s %s  →  %s  in %.0fms",
            request.method,
            request.url.path,
            response.status_code,
            duration_ms,
        )
        return response


# ── Health check ───────────────────────────────────────────────────────────────

app = create_app()


@app.get("/health", tags=["Infrastructure"])
async def health_check(request: Request):
    """
    Lightweight liveness probe for load balancers / Docker health checks.
    Returns 200 if the app is running and ChromaDB collection is accessible.
    Returns 503 if the vector store is unreachable.
    """
    try:

        collection = request.app.state.chroma_collection
        count = collection.count()
        return {
            "status": "ok",
            "service": "documind",
            "vector_store": "reachable",
            "document_chunks": count,
        }
    except Exception as e:
        logger.error("Health check failed: %s", e)
        return JSONResponse(
            status_code=503,
            content={
                "status": "degraded",
                "vector_store": "unreachable",
                "detail": str(e),
            },
        )