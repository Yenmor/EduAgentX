from __future__ import annotations

import json
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from backend.api import assessment, chat, classroom, learning_path, model_status, profile, resources
from backend.database.db import init_db
from backend.rag.course_manifest import load_manifest
from backend.rag.ingest import (
    DEFAULT_CHUNKS_PATH,
    DEFAULT_INDEX_DIR,
    DEFAULT_MANIFEST_PATH,
    DEFAULT_RAPTOR_TREE_PATH,
    DEFAULT_RAW_ZIP,
    build_sample_course_index,
    ingest_knowledge,
)
from backend.rag.retriever import RagRetriever
from backend.rag.vector_store import VectorStore
from backend.schemas.knowledge import BuildIndexRequest, IngestRequest, SearchRequest
from backend.services.spark_wrapper import SparkAPIError


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize persistent backend resources before serving requests."""
    init_db()
    yield


def create_app() -> FastAPI:
    """Create and configure the EduAgentX FastAPI application."""
    init_db()
    app = FastAPI(title="EduAgentX Backend", version="0.2.0", lifespan=lifespan)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=[
            "http://localhost:3000",
            "http://127.0.0.1:3000",
            "http://localhost:3001",
            "http://127.0.0.1:3001",
        ],
        allow_origin_regex=r"^http://(localhost|127\.0\.0\.1):\d+$",
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @app.exception_handler(SparkAPIError)
    async def spark_api_error_handler(request, exc: SparkAPIError):
        return JSONResponse(
            status_code=502,
            content={
                "ok": False,
                "error": "LLM API request failed",
                "detail": str(exc),
            },
        )

    @app.get("/health")
    def health() -> dict[str, str]:
        return {"status": "ok", "service": "EduAgentX"}

    @app.get("/api/health")
    def api_health() -> dict[str, str]:
        return health()

    app.include_router(chat.router)
    app.include_router(profile.router)
    app.include_router(resources.router)
    app.include_router(learning_path.router)
    app.include_router(assessment.router)
    app.include_router(model_status.router)
    app.include_router(classroom.router)

    @app.post("/api/knowledge/build_index")
    def build_index_endpoint(request: BuildIndexRequest) -> dict:
        try:
            if request.sample_dir:
                return build_sample_course_index(sample_dir=request.sample_dir)
            return build_sample_course_index()
        except Exception as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @app.post("/api/knowledge/ingest")
    def ingest_endpoint(request: IngestRequest) -> dict:
        zip_path = request.zip_path or str(DEFAULT_RAW_ZIP)
        try:
            return ingest_knowledge(zip_path=zip_path)
        except Exception as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @app.post("/api/knowledge/search")
    def search_endpoint(request: SearchRequest) -> dict:
        retriever = RagRetriever()
        if not retriever.is_ready():
            raise HTTPException(
                status_code=409,
                detail="Knowledge index is missing. Run scripts/ingest_knowledge.py or call /api/knowledge/ingest first.",
            )
        return retriever.search(request.query, top_k=request.top_k)

    @app.get("/api/knowledge/tree")
    def tree_endpoint() -> dict:
        path = Path(DEFAULT_RAPTOR_TREE_PATH)
        if not path.exists():
            raise HTTPException(status_code=404, detail="Knowledge tree is missing. Run ingest first.")
        return json.loads(path.read_text(encoding="utf-8"))

    @app.get("/api/knowledge/status")
    def status_endpoint() -> dict:
        manifest_path = Path(DEFAULT_MANIFEST_PATH)
        tree_path = Path(DEFAULT_RAPTOR_TREE_PATH)
        chunks_path = Path(DEFAULT_CHUNKS_PATH)
        index_dir = Path(DEFAULT_INDEX_DIR)
        metadata_path = index_dir / "metadata.json"

        manifest_loaded = manifest_path.exists()
        tree_loaded = tree_path.exists()
        metadata = {}
        if metadata_path.exists():
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))

        document_count = int(metadata.get("document_count", 0) or 0)
        if not document_count and manifest_loaded:
            document_count = len(load_manifest(manifest_path))

        chunk_count = int(metadata.get("chunk_count", 0) or 0)
        if not chunk_count and metadata.get("chunks"):
            chunk_count = len(metadata.get("chunks", []))
        if not chunk_count and chunks_path.exists():
            with chunks_path.open("r", encoding="utf-8") as handle:
                chunk_count = sum(1 for line in handle if line.strip())

        store = VectorStore()
        return {
            "document_count": document_count,
            "chunk_count": chunk_count,
            "retrieval_backend": metadata.get("backend", "missing" if not metadata else "fallback"),
            "faiss_available": store.faiss_available,
            "manifest_loaded": manifest_loaded,
            "tree_loaded": tree_loaded,
            "active_index_source": metadata.get("active_index_source") or metadata.get("index_source") or "unknown",
        }

    @app.get("/api/knowledge/manifest")
    def manifest_endpoint() -> list[dict]:
        path = Path(DEFAULT_MANIFEST_PATH)
        if not path.exists():
            raise HTTPException(status_code=404, detail="Course manifest is missing. Run ingest first.")
        return load_manifest(path)

    return app


app = create_app()
