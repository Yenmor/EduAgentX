from __future__ import annotations

import json

from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session

from backend.agents.evaluation_agent import EvaluationAgent
from backend.database.db import get_db
from backend.database.models import LearningEvent, MasteryRecord, Student
from backend.schemas.assessment import AssessmentSubmitRequest, AssessmentSubmitResponse


router = APIRouter(prefix="/api/assessment", tags=["assessment"])
evaluation_agent = EvaluationAgent()


def _dump_model(model) -> dict:
    return model.model_dump() if hasattr(model, "model_dump") else model.dict()


@router.post("/submit", response_model=AssessmentSubmitResponse)
def submit_assessment(request: AssessmentSubmitRequest, db: Session = Depends(get_db)) -> AssessmentSubmitResponse:
    if db.get(Student, request.student_id) is None:
        db.add(Student(id=request.student_id, name="Demo Student"))
        db.flush()

    record = (
        db.query(MasteryRecord)
        .filter_by(student_id=request.student_id, knowledge_point=request.knowledge_point)
        .first()
    )
    previous = record.mastery_probability if record else 0.25
    updated = evaluation_agent.update_mastery(previous, request.is_correct, request.score)
    if record is None:
        record = MasteryRecord(
            student_id=request.student_id,
            knowledge_point=request.knowledge_point,
            mastery_probability=updated,
        )
        db.add(record)
    else:
        record.mastery_probability = updated

    event = LearningEvent(
        student_id=request.student_id,
        event_type="assessment_submitted",
        knowledge_point=request.knowledge_point,
        payload_json=json.dumps(_dump_model(request), ensure_ascii=False),
    )
    db.add(event)
    db.commit()
    db.refresh(event)
    return AssessmentSubmitResponse(
        student_id=request.student_id,
        knowledge_point=request.knowledge_point,
        previous_mastery_probability=previous,
        mastery_probability=updated,
        event_id=event.id,
    )
