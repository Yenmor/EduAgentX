from __future__ import annotations

import json

from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session

from backend.agents.judge_agent import JudgeAgent
from backend.agents.resource_agent import ResourceAgent
from backend.database.db import get_db
from backend.database.models import LearningResource as DBLearningResource
from backend.database.models import Student
from backend.schemas.resource import ResourceGenerateRequest, ResourceGenerateResponse


router = APIRouter(prefix="/api/resources", tags=["resources"])
resource_agent = ResourceAgent()
judge_agent = JudgeAgent()


def _dump_model(model) -> dict:
    return model.model_dump() if hasattr(model, "model_dump") else model.dict()


def _ensure_student(db: Session, student_id: str) -> None:
    if db.get(Student, student_id) is None:
        db.add(Student(id=student_id, name="Demo Student"))
        db.flush()


@router.post("/generate", response_model=ResourceGenerateResponse)
def generate_resources(request: ResourceGenerateRequest, db: Session = Depends(get_db)) -> ResourceGenerateResponse:
    _ensure_student(db, request.student_id)
    response = resource_agent.generate_resources(request)
    response = judge_agent.review_resources(response, required_terms=[request.knowledge_point or request.topic])
    for resource in response.resources:
        db.add(
            DBLearningResource(
                student_id=request.student_id,
                knowledge_point=request.knowledge_point or request.topic,
                resource_type=resource.type,
                title=resource.title,
                content=json.dumps(_dump_model(resource), ensure_ascii=False),
            )
        )
    db.commit()
    return response
