from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

from backend.schemas.diagnosis import DiagnosisResult
from backend.schemas.evaluation import EvaluationResult
from backend.schemas.judge import JudgeResult
from backend.schemas.learning_path import LearningPathResponse
from backend.schemas.profile import StudentProfile
from backend.schemas.resource import LearningResource, SourceChunk


class AgentTraceStep(BaseModel):
    agent_id: str
    agent_name: str
    role: str
    status: Literal["pending", "running", "done", "error"]
    action: str
    input_summary: str | None = None
    output_summary: str | None = None
    started_at: str | None = None
    ended_at: str | None = None
    duration_ms: int | None = None


class LearningState(BaseModel):
    """Unified backend state for classroom generation."""

    session_id: str
    student_id: str | None = None
    user_message: str
    selected_course_ids: list[str] = Field(default_factory=list)
    profile: StudentProfile | None = None
    course_metadata: list[dict] = Field(default_factory=list)
    retrieved_chunks: list[SourceChunk] = Field(default_factory=list)
    diagnosis: DiagnosisResult | None = None
    learning_path: LearningPathResponse | None = None
    resources: list[LearningResource] = Field(default_factory=list)
    evaluation: EvaluationResult | None = None
    judge: JudgeResult | None = None
    mastery_map: dict[str, float] = Field(default_factory=dict)
    agent_trace: list[AgentTraceStep] = Field(default_factory=list)
    mode: Literal["real", "mock"] = "real"
