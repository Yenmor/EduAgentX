from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class ChatRequest(BaseModel):
    """Input for a student-facing chat turn."""

    message: str = Field(..., min_length=1)
    student_id: str = "demo-student"
    mode: Literal[
        "auto",
        "profile",
        "qa",
        "planner",
        "resource",
        "generate_resource",
        "plan_learning_path",
        "update_profile",
    ] = "auto"


class ChatResponse(BaseModel):
    """Structured mock response returned by the MVP chat agent."""

    reply: str
    agent: str
    citations: list[dict] = Field(default_factory=list)
    suggestions: list[str] = Field(default_factory=list)
    model_provider: str = "mock"
    model_name: str = "mock"
    fallback_used: bool = False
