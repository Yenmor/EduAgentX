from __future__ import annotations

from pydantic import BaseModel, Field


class DiagnosisResult(BaseModel):
    """Personalized diagnosis derived from profile, courses, and RAG chunks."""

    topic: str
    related_courses: list[str] = Field(default_factory=list)
    related_concepts: list[str] = Field(default_factory=list)
    weak_points: list[str] = Field(default_factory=list)
    mastery_estimate: dict[str, float] = Field(default_factory=dict)
    recommended_focus: list[str] = Field(default_factory=list)
    reason: str
