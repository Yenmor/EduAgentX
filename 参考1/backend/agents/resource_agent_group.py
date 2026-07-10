from __future__ import annotations

from backend.schemas.diagnosis import DiagnosisResult
from backend.schemas.learning_path import LearningPathResponse
from backend.schemas.profile import StudentProfile
from backend.schemas.resource import LearningResource, SourceChunk


class ResourceAgentGroup:
    """Generates multiple classroom resource types through specialized sub-agent methods."""

    def generate(
        self,
        profile: StudentProfile | None,
        diagnosis: DiagnosisResult,
        retrieved_chunks: list[SourceChunk],
        learning_path: LearningPathResponse | None,
        selected_courses: list[dict],
    ) -> list[LearningResource]:
        chunks = retrieved_chunks[:3]
        primary_course = selected_courses[0] if selected_courses else {}
        course_id = str(primary_course.get("id", "rag"))
        course_name = str(primary_course.get("name", "RAG 检索增强生成"))
        concepts = diagnosis.related_concepts[:4] or ["Chunking", "Embedding", "Retriever"]
        prereqs = ["Python", "向量表示"]

        return [
            self._doc(course_id, course_name, concepts, prereqs, chunks),
            self._mindmap(course_id, course_name, concepts, prereqs, chunks),
            self._quiz(course_id, course_name, concepts, prereqs, chunks),
            self._reading(course_id, course_name, concepts, prereqs, chunks),
            self._code(course_id, course_name, concepts, prereqs, chunks),
            self._animation(course_id, course_name, concepts, prereqs, chunks),
            self._project(course_id, course_name, concepts, prereqs, chunks),
        ]

    def _resource(
        self,
        resource_id: str,
        resource_type: str,
        title: str,
        content: str,
        course_id: str,
        course_name: str,
        concepts: list[str],
        prereqs: list[str],
        chunks: list[SourceChunk],
        minutes: int,
        difficulty: str = "入门",
    ) -> LearningResource:
        return LearningResource(
            id=resource_id,
            type=resource_type,
            title=title,
            content=content,
            course_id=course_id,
            course_name=course_name,
            difficulty=difficulty,
            estimated_minutes=minutes,
            target_concepts=concepts,
            prerequisite_concepts=prereqs,
            source_chunks=chunks,
            personalized_reason="匹配学生两周完成课程知识库问答 Agent 的目标，优先补齐 RAG grounding 与项目实践能力。",
        )

    def _doc(self, *args) -> LearningResource:
        return self._resource("res-doc", "doc", "RAG 基础流程讲解文档", "讲解 Chunking、Embedding、Retriever、Generator 与 Judge 检查的完整链路。", *args, minutes=10)

    def _mindmap(self, *args) -> LearningResource:
        return self._resource("res-mindmap", "mindmap", "RAG + LangChain + Agent 思维导图", "mindmap\n  root((RAG 项目))\n    Chunking\n    Retriever\n    LCEL\n    Agent 编排", *args, minutes=6)

    def _quiz(self, *args) -> LearningResource:
        return self._resource("res-quiz", "quiz", "RAG 核心概念诊断测验", "3 道题覆盖 Embedding、Chunking、Retriever 与 JudgeAgent。", *args, minutes=8, difficulty="中等")

    def _reading(self, *args) -> LearningResource:
        return self._resource("res-reading", "reading", "Grounding 与引用质量拓展阅读", "阅读 RAG 资源如何保留 source_chunks 并降低幻觉风险。", *args, minutes=12, difficulty="进阶")

    def _code(self, *args) -> LearningResource:
        return self._resource("res-code", "code", "LCEL Retriever 代码实验", "用最小检索链实现课程片段召回，并输出 chunk_id 与 source_file。", *args, minutes=25, difficulty="中等")

    def _animation(self, *args) -> LearningResource:
        return self._resource("res-animation", "animation", "RAG 检索流程动画脚本", "动画展示 query embedding、top-k retrieval、grounded generation。", *args, minutes=7)

    def _project(self, *args) -> LearningResource:
        return self._resource("res-project", "project", "课程知识库问答 Agent PBL", "整合 RAG、LangChain、Agent 编排、测评反馈与 Judge 审查。", *args, minutes=90, difficulty="进阶")
