from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class ResourceGenerateRequest(BaseModel):
    """Request for generating personalized learning resources."""

    student_id: str
    topic: str = Field(..., min_length=1)
    knowledge_point: str | None = None
    difficulty: str = "adaptive"


class SourceChunk(BaseModel):
    """RAG source chunk attached to generated learning resources."""

    id: str
    course_id: str = ""
    course_name: str = ""
    source_file: str = ""
    chapter: str | None = None
    section: str | None = None
    heading_path: list[str] = Field(default_factory=list)
    excerpt: str = ""
    score: float | None = None


class LearningResource(BaseModel):
    """Generated resource card returned to the frontend."""

    id: str = ""
    type: Literal["doc", "mindmap", "quiz", "reading", "code", "animation", "project"] | str
    title: str
    content: str
    course_id: str = ""
    course_name: str = ""
    difficulty: Literal["入门", "中等", "进阶"] | str = "入门"
    estimated_minutes: int
    target_concepts: list[str] = Field(default_factory=list)
    prerequisite_concepts: list[str] = Field(default_factory=list)
    source_chunks: list[SourceChunk] = Field(default_factory=list)
    personalized_reason: str = ""
    judge_score: float | None = None
    judge_feedback: str | None = None
    grounded: bool | None = None
    metadata: dict = Field(default_factory=dict)


class ResourceGenerateResponse(BaseModel):
    """Bundle of generated resource cards and judge feedback."""

    student_id: str
    topic: str
    resources: list[LearningResource]
    judge_score: int
    revision_suggestions: list[str] = Field(default_factory=list)
    model_provider: str = "mock"
    model_name: str = "mock"
    fallback_used: bool = False
