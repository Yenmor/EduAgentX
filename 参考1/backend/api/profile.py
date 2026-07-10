from __future__ import annotations

import json

from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session

from backend.agents.profile_agent import ProfileAgent
from backend.database.db import get_db
from backend.database.models import Student, StudentProfile as DBStudentProfile
from backend.schemas.profile import ProfileUpdateRequest, ProfileUpdateResponse, StudentProfile


router = APIRouter(prefix="/api/profile", tags=["profile"])
agent = ProfileAgent()


def _dump_model(model) -> dict:
    return model.model_dump() if hasattr(model, "model_dump") else model.dict()


def _json_list(raw: str) -> list[str]:
    try:
        value = json.loads(raw)
        return value if isinstance(value, list) else []
    except json.JSONDecodeError:
        return []


def _ensure_student(db: Session, student_id: str, name: str | None = None) -> Student:
    student = db.get(Student, student_id)
    if student is None:
        student = Student(id=student_id, name=name or "Demo Student")
        db.add(student)
        db.flush()
    elif name:
        student.name = name
    return student


def _to_schema(student: Student, profile: DBStudentProfile | None) -> StudentProfile:
    if profile is None:
        return StudentProfile(student_id=student.id, name=student.name)
    return StudentProfile(
        student_id=student.id,
        name=student.name,
        goals=_json_list(profile.goals_json),
        interests=_json_list(profile.interests_json),
        skill_level=profile.skill_level,
        preferred_style=profile.preferred_style,
    )


@router.get("/{student_id}", response_model=StudentProfile)
def get_profile(student_id: str, db: Session = Depends(get_db)) -> StudentProfile:
    student = _ensure_student(db, student_id)
    db.commit()
    profile = db.query(DBStudentProfile).filter_by(student_id=student_id).first()
    return _to_schema(student, profile)


@router.post("/update", response_model=ProfileUpdateResponse)
def update_profile(request: ProfileUpdateRequest, db: Session = Depends(get_db)) -> ProfileUpdateResponse:
    payload = agent.extract_profile_payload(request.student_id, request.text, name=request.name)
    extracted = payload["profile"]
    student = _ensure_student(db, request.student_id, extracted.name)
    profile = db.query(DBStudentProfile).filter_by(student_id=request.student_id).first()
    if profile is None:
        profile = DBStudentProfile(student_id=request.student_id)
        db.add(profile)
    profile.goals_json = json.dumps(extracted.goals, ensure_ascii=False)
    profile.interests_json = json.dumps(extracted.interests, ensure_ascii=False)
    profile.skill_level = extracted.skill_level
    profile.preferred_style = extracted.preferred_style
    db.commit()
    db.refresh(student)
    db.refresh(profile)
    return ProfileUpdateResponse(profile=_to_schema(student, profile), extracted={**_dump_model(extracted), **agent.last_model_info})
