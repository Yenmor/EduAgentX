from __future__ import annotations

from backend.schemas.chat import ChatRequest, ChatResponse
from backend.schemas.resource import LearningResource, ResourceGenerateRequest, ResourceGenerateResponse
from backend.services.spark_wrapper import SparkWrapper


RESOURCE_TYPES = ["explanation_doc", "mindmap", "quiz", "reading_material", "code_lab"]
RESOURCE_SYSTEM_PROMPT = (
    "你是高校课程学习资源生成智能体。请基于课程知识库检索结果和学生画像，生成个性化学习资源。"
    "内容必须尽量基于给定资料，不要编造。"
)


class ResourceAgent:
    """Generates learning resources through SparkWrapper."""

    def __init__(self, llm: SparkWrapper | None = None) -> None:
        self.llm = llm or SparkWrapper()
        self.last_model_info = {"model_provider": "mock", "model_name": "mock", "fallback_used": False}

    def run(self, request: ChatRequest) -> ChatResponse:
        response = self.llm.generate(
            [
                {"role": "system", "content": RESOURCE_SYSTEM_PROMPT},
                {"role": "user", "content": request.message},
            ]
        )
        self.last_model_info = self._model_info(response)
        return ChatResponse(
            reply=str(response.get("content", "")),
            agent="ResourceAgent",
            suggestions=["Read the matching course note.", "Try one checkpoint exercise.", "Ask for a quiz."],
            **self.last_model_info,
        )

    def generate_resources(self, request: ResourceGenerateRequest) -> ResourceGenerateResponse:
        topic = request.knowledge_point or request.topic
        resources: list[LearningResource] = []
        merged_info = {"model_provider": "mock", "model_name": "mock", "fallback_used": False}
        for index, resource_type in enumerate(RESOURCE_TYPES):
            response = self.llm.generate(
                [
                    {"role": "system", "content": RESOURCE_SYSTEM_PROMPT},
                    {
                        "role": "user",
                        "content": (
                            f"请为主题 {topic} 生成 {resource_type}。难度：{request.difficulty}。"
                            "输出简洁、可演示、适合高校课程。"
                        ),
                    },
                ]
            )
            info = self._model_info(response)
            merged_info = self._merge_info(merged_info, info)
            resources.append(
                LearningResource(
                    type=resource_type,
                    title=f"{topic} - {resource_type.replace('_', ' ').title()}",
                    content=str(response.get("content", "")),
                    estimated_minutes=15 + index * 5,
                    metadata={"difficulty": request.difficulty, **info},
                )
            )
        self.last_model_info = merged_info
        return ResourceGenerateResponse(
            student_id=request.student_id,
            topic=request.topic,
            resources=resources,
            judge_score=0,
            revision_suggestions=[],
            **self.last_model_info,
        )

    def generate_orchestrated_resources(
        self,
        topic: str,
        sources: list[dict],
        profile: dict | None = None,
        rewrite_instruction: str | None = None,
    ) -> list[dict]:
        """Generate resource dictionaries required by the orchestrator contract."""
        profile = profile or {}
        level = profile.get("skill_level", "beginner")
        source_hint = sources[0].get("source_file") if sources else "course knowledge base"
        source_text = "\n".join(str(item.get("content", ""))[:240] for item in sources[:3])
        resources: list[dict] = []
        merged_info = {"model_provider": "mock", "model_name": "mock", "fallback_used": False}
        for resource_type in RESOURCE_TYPES:
            response = self.llm.generate(
                [
                    {"role": "system", "content": RESOURCE_SYSTEM_PROMPT},
                    {
                        "role": "user",
                        "content": (
                            f"主题：{topic}\n学生画像：{profile}\n资料：{source_text}\n"
                            f"资源类型：{resource_type}\n重写要求：{rewrite_instruction or '无'}"
                        ),
                    },
                ]
            )
            info = self._model_info(response)
            merged_info = self._merge_info(merged_info, info)
            resources.append(
                {
                    "title": f"{topic} - {resource_type.replace('_', ' ').title()}",
                    "type": resource_type,
                    "content": str(response.get("content", "")),
                    "sources": sources,
                    "judge_score": 0,
                    "personalized_reason": f"Matched to a {level} learner and grounded in {source_hint}.",
                    **info,
                }
            )
        self.last_model_info = merged_info
        return resources

    def _model_info(self, result: dict) -> dict:
        return {
            "model_provider": str(result.get("model_provider", "mock")),
            "model_name": str(result.get("model_name", "mock")),
            "fallback_used": bool(result.get("fallback_used", False)),
        }

    def _merge_info(self, current: dict, new: dict) -> dict:
        return {
            "model_provider": new.get("model_provider", current.get("model_provider", "mock")),
            "model_name": new.get("model_name", current.get("model_name", "mock")),
            "fallback_used": bool(current.get("fallback_used", False) or new.get("fallback_used", False)),
        }
