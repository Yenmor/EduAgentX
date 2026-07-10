from __future__ import annotations

from datetime import datetime

from sqlalchemy import DateTime, Float, ForeignKey, Integer, String, Text
from sqlalchemy.orm import Mapped, mapped_column, relationship

from backend.database.base import Base


class Student(Base):
    """A learner account tracked by EduAgentX."""

    __tablename__ = "students"

    id: Mapped[str] = mapped_column(String(128), primary_key=True)
    name: Mapped[str] = mapped_column(String(128), default="Demo Student")
    email: Mapped[str | None] = mapped_column(String(255), nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)

    profile: Mapped["StudentProfile | None"] = relationship(back_populates="student", uselist=False)


class StudentProfile(Base):
    """Dialogue-derived student profile facts."""

    __tablename__ = "student_profiles"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True)
    student_id: Mapped[str] = mapped_column(ForeignKey("students.id"), unique=True, index=True)
    goals_json: Mapped[str] = mapped_column(Text, default="[]")
    interests_json: Mapped[str] = mapped_column(Text, default="[]")
    skill_level: Mapped[str] = mapped_column(String(64), default="beginner")
    preferred_style: Mapped[str] = mapped_column(String(64), default="interactive")
    updated_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    student: Mapped[Student] = relationship(back_populates="profile")


class KnowledgePoint(Base):
    """A course concept or skill node."""

    __tablename__ = "knowledge_points"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True)
    course_id: Mapped[str] = mapped_column(String(128), index=True)
    module_name: Mapped[str] = mapped_column(String(255), index=True)
    title: Mapped[str] = mapped_column(String(255), index=True)
    description: Mapped[str] = mapped_column(Text, default="")
    prerequisites_json: Mapped[str] = mapped_column(Text, default="[]")
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)


class LearningResource(Base):
    """Generated learning resource cards."""

    __tablename__ = "learning_resources"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True)
    student_id: Mapped[str] = mapped_column(ForeignKey("students.id"), index=True)
    knowledge_point: Mapped[str] = mapped_column(String(255), index=True)
    resource_type: Mapped[str] = mapped_column(String(64), index=True)
    title: Mapped[str] = mapped_column(String(255))
    content: Mapped[str] = mapped_column(Text)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)


class MasteryRecord(Base):
    """Current BKT-style mastery estimate for a student and knowledge point."""

    __tablename__ = "mastery_records"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True)
    student_id: Mapped[str] = mapped_column(ForeignKey("students.id"), index=True)
    knowledge_point: Mapped[str] = mapped_column(String(255), index=True)
    mastery_probability: Mapped[float] = mapped_column(Float, default=0.25)
    updated_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)


class LearningEvent(Base):
    """Append-only learning behavior and assessment events."""

    __tablename__ = "learning_events"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True)
    student_id: Mapped[str] = mapped_column(ForeignKey("students.id"), index=True)
    event_type: Mapped[str] = mapped_column(String(64), index=True)
    knowledge_point: Mapped[str | None] = mapped_column(String(255), nullable=True)
    payload_json: Mapped[str] = mapped_column(Text, default="{}")
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)


class ChatMessage(Base):
    """Stores chat turns for later profile and learning analytics work."""

    __tablename__ = "chat_messages"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True)
    student_id: Mapped[str] = mapped_column(String(128), index=True)
    role: Mapped[str] = mapped_column(String(32))
    content: Mapped[str] = mapped_column(Text)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
