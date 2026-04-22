import logging
import time
from contextlib import asynccontextmanager
from venv import logger

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
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

logger = logging.getLogger("documind")

# ── startup / shutdown ───────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):

    logger.info("Starting up the application...")

    chroma_client = chromadb.Client(
        Settings(
            chroma_db_impl="duckdb+parquet",
            persist_directory="./chroma_data",
            anonymized_telemetry=False,
        )
    )

    init_vector_store(chroma_client,app)
    logger.info("ChromaDB initialized.")

    yield

    logger.info("Shutting down the application...")
    chroma_client.persist()
    logger.info("Shutdown complete.")

# ── App factory ────────────────────────────────────────────────────────────────

def create_app() -> FastAPI:

    app=FastAPI(
        title="DocuMind",
        description="rag application",

        version="1.0.0",
        lifespan=lifespan,
    
        docs_url="/api/docs",
        redoc_url="/api/redoc",
        openapi_url="/api/openapi.json",
    )

    configure_cors(app)
    mount_routers(app)
    register_error_handlers(app)
    add_request_logging(app)
 
    return app

# ── CORS ───────────────────────────────────────────────────────────────────────

def configure_cors(app: FastAPI) -> None:
    app.add_middleware(
        CORSMiddleware,
        allow_origins=[
            "http://localhost:5173",   # Vite dev server
            "http://localhost:3000",   # CRA / alternative dev port
            "http://127.0.0.1:5173"],
        allow_credentials=True,
        allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
        allow_headers=["content-type", "authorization", "accept", "origin", "user-agent", "x-requested-with"],
    )

# ── Routers ────────────────────────────────────────────────────────────────────

def mount_routers(app: FastAPI) -> None:
    app.include_router(
        ingest.router,
        prefix="/api/v1/ingest", 
        tags=["Ingestion"])
    app.include_router(
        chat.router, 
        prefix="/api/chat", 
        tags=["Chat"])
    
# ── Error Handlers ───────────────────────────────────────────────────────────

def _register_error_handlers(app: FastAPI) -> None:

    @app.exception_handler(ValueError)
    async def value_error_handler(request: Request, exc: ValueError):
        logger.warning("Bad request — %s %s: %s", request.method, request.url.path, exc)
        return JSONResponse(
            status_code=400,
            content={
                "error": "bad_request",
                 "message": str(exc),
                 "path": str(Request.url.path),
            },
        )
    
    @app.exception_handler(FileNotFoundError)

    async def file_not_found_handler(request: Request, exc: FileNotFoundError):
        logger.warning("Resource not found - %s", exc)
        return JSONResponse(
            content={
                "error": "not_found",
                "message": str(exc),
            },
        )