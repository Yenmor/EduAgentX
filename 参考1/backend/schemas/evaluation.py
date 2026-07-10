from __future__ import annotations

from pydantic import BaseModel, Field


class EvaluationResult(BaseModel):
    """Quiz evaluation result used to update mastery and trigger replanning."""

    score: int
    correct_count: int
    total_count: int
    weak_concepts: list[str] = Field(default_factory=list)
    updated_mastery_map: dict[str, float] = Field(default_factory=dict)
    replanning_suggestions: list[str] = Field(default_factory=list)
