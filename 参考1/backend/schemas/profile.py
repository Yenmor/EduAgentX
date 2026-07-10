from __future__ import annotations

from pydantic import BaseModel, Field


class StudentProfile(BaseModel):
    """API representation of a dialogue-derived student profile."""

    student_id: str
    name: str = "Demo Student"
    goals: list[str] = Field(default_factory=list)
    interests: list[str] = Field(default_factory=list)
    skill_level: str = "beginner"
    preferred_style: str = "interactive"


class ProfileUpdateRequest(BaseModel):
    """Text input used by ProfileAgent to update a student profile."""

    student_id: str
    text: str = Field(..., min_length=1)
    name: str | None = None


class ProfileUpdateResponse(BaseModel):
    """Response returned after profile extraction and persistence."""

    profile: StudentProfile
    extracted: dict
    agent: str = "ProfileAgent"
