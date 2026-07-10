from __future__ import annotations

from typing import Optional

from pydantic import BaseModel, Field


class IngestRequest(BaseModel):
    """Request body for loading a course knowledge zip."""

    zip_path: Optional[str] = None


class SearchRequest(BaseModel):
    """Request body for RAG search."""

    query: str = Field(..., min_length=1)
    top_k: int = Field(default=5, ge=1, le=20)


class BuildIndexRequest(BaseModel):
    """Request body for building a sample-course RAG index."""

    sample_dir: Optional[str] = None
