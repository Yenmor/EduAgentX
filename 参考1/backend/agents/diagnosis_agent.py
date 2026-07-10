from __future__ import annotations

from backend.schemas.diagnosis import DiagnosisResult
from backend.schemas.profile import StudentProfile
from backend.schemas.resource import SourceChunk


class DiagnosisAgent:
    """Rule-based diagnosis with a clear seam for future LLM enhancement."""

    def diagnose(
        self,
        user_message: str,
        profile: StudentProfile | None,
        courses: list[dict],
        retrieved_chunks: list[SourceChunk],
    ) -> DiagnosisResult:
        concepts = []
        for course in courses:
            concepts.extend(str(item) for item in course.get("concepts", []))
        for chunk in retrieved_chunks:
            concepts.extend(chunk.heading_path)

        text = f"{user_message} {' '.join(concepts)}".lower()
        weak_points: list[str] = []
        for concept in ["Embedding", "Chunking", "Retriever", "LCEL", "Tool Calling", "Multi-Agent", "Agent 编排"]:
            if concept.lower() in text or concept in {"Embedding", "Chunking", "Retriever"}:
                weak_points.append(concept)

        related_courses = [str(course.get("name", course.get("id", ""))) for course in courses]
        related_concepts = list(dict.fromkeys(concepts))[:10] or ["RAG", "LangChain", "AI Agent"]
        mastery = {
            "Python": 0.72,
            "机器学习": 0.58,
            "大语言模型基础": 0.45,
            "RAG": 0.32,
            "AI Agent": 0.28,
        }
        if profile and profile.skill_level == "advanced":
            mastery = {key: min(0.9, value + 0.12) for key, value in mastery.items()}

        return DiagnosisResult(
            topic="RAG 与 AI Agent 项目实战",
            related_courses=related_courses,
            related_concepts=related_concepts,
            weak_points=list(dict.fromkeys(weak_points)),
            mastery_estimate=mastery,
            recommended_focus=["先补 Embedding 与 Chunking，再学习 LangChain Retriever，最后进入 Agent 编排"],
            reason="学生具备一定 Python 基础，但缺少向量检索、RAG 系统和 Agent 项目经验。",
        )
