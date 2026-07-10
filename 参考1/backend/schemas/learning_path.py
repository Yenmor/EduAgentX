from __future__ import annotations

from pydantic import BaseModel, Field


class LearningPathRequest(BaseModel):
    """Request for a personalized learning path."""

    student_id: str
    goal: str = Field(..., min_length=1)
    horizon_days: int = Field(default=14, ge=1, le=180)


class LearningStep(BaseModel):
    """One step in a generated learning path."""

    order: int
    title: str
    knowledge_point: str
    activity: str
    estimated_minutes: int
    mastery_target: float


class LearningPathResponse(BaseModel):
    """Generated staged learning path."""

    student_id: str
    goal: str
    steps: list[LearningStep]
    mermaid: str
    agent: str = "PlannerAgent"
    model_provider: str = "mock"
    model_name: str = "mock"
    fallback_used: bool = False
