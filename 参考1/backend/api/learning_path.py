from __future__ import annotations

import json

from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session

from backend.agents.planner_agent import PlannerAgent
from backend.database.db import get_db
from backend.database.models import LearningEvent, Student, StudentProfile as DBStudentProfile
from backend.schemas.learning_path import LearningPathRequest, LearningPathResponse
from backend.schemas.profile import StudentProfile


router = APIRouter(prefix="/api/path", tags=["learning-path"])
planner_agent = PlannerAgent()


def _dump_model(model) -> dict:
    return model.model_dump() if hasattr(model, "model_dump") else model.dict()


def _json_list(raw: str) -> list[str]:
    try:
        value = json.loads(raw)
        return value if isinstance(value, list) else []
    except json.JSONDecodeError:
        return []


def _load_profile(db: Session, student_id: str) -> StudentProfile | None:
    student = db.get(Student, student_id)
    profile = db.query(DBStudentProfile).filter_by(student_id=student_id).first()
    if student is None or profile is None:
        return None
    return StudentProfile(
        student_id=student.id,
        name=student.name,
        goals=_json_list(profile.goals_json),
        interests=_json_list(profile.interests_json),
        skill_level=profile.skill_level,
        preferred_style=profile.preferred_style,
    )


@router.post("/generate", response_model=LearningPathResponse)
def generate_path(request: LearningPathRequest, db: Session = Depends(get_db)) -> LearningPathResponse:
    if db.get(Student, request.student_id) is None:
        db.add(Student(id=request.student_id, name="Demo Student"))
        db.flush()
    response = planner_agent.generate_path(request, profile=_load_profile(db, request.student_id))
    db.add(
        LearningEvent(
            student_id=request.student_id,
            event_type="path_generated",
            payload_json=json.dumps(_dump_model(response), ensure_ascii=False),
        )
    )
    db.commit()
    return response
