from __future__ import annotations

from pydantic import BaseModel, Field


class AssessmentSubmitRequest(BaseModel):
    """Simplified assessment submission for BKT updates."""

    student_id: str
    knowledge_point: str = Field(..., min_length=1)
    is_correct: bool
    score: float = Field(default=1.0, ge=0.0, le=1.0)
    response_text: str | None = None


class AssessmentSubmitResponse(BaseModel):
    """Updated mastery probability after an assessment event."""

    student_id: str
    knowledge_point: str
    previous_mastery_probability: float
    mastery_probability: float
    event_id: int | None = None
    agent: str = "EvaluationAgent"
