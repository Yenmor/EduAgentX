from __future__ import annotations

from pydantic import BaseModel, Field


class JudgeResult(BaseModel):
    """Quality and grounding review for generated learning resources."""

    score: float
    grounded: bool
    feedback: str
    risks: list[str] = Field(default_factory=list)
    suggestions: list[str] = Field(default_factory=list)
