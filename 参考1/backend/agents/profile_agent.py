from __future__ import annotations

from backend.schemas.chat import ChatRequest, ChatResponse
from backend.schemas.profile import StudentProfile
from backend.services.spark_wrapper import SparkWrapper


PROFILE_SYSTEM_PROMPT = (
    "你是学生画像分析智能体。请从学生自然语言中提取学习画像，输出 JSON。"
    "画像至少包含：专业背景、当前课程、知识基础、学习目标、薄弱知识点、认知风格、资源偏好、学习节奏。"
)


class ProfileAgent:
    """Builds a lightweight student profile through SparkWrapper."""

    def __init__(self, llm: SparkWrapper | None = None) -> None:
        self.llm = llm or SparkWrapper()
        self.last_model_info = {"model_provider": "mock", "model_name": "mock", "fallback_used": False}

    def run(self, request: ChatRequest) -> ChatResponse:
        payload = self.extract_profile_payload(request.student_id, request.message)
        profile = payload["profile"]
        return ChatResponse(
            reply=f"Profile updated for {profile.name}.",
            agent="ProfileAgent",
            suggestions=["Tell me your target course.", "Share your current programming level."],
            **self.last_model_info,
        )

    def extract_profile_payload(self, student_id: str, text: str, name: str | None = None) -> dict:
        result = self.llm.generate_json(
            [
                {"role": "system", "content": PROFILE_SYSTEM_PROMPT},
                {"role": "user", "content": text},
            ],
            temperature=0.2,
        )
        self.last_model_info = self._model_info(result)
        profile = self._profile_from_result(student_id, text, result, name=name)
        return {"profile": profile, **self.last_model_info, "raw": result}

    def extract_profile(self, student_id: str, text: str, name: str | None = None) -> StudentProfile:
        return self.extract_profile_payload(student_id, text, name=name)["profile"]

    def _profile_from_result(self, student_id: str, text: str, result: dict, name: str | None = None) -> StudentProfile:
        weak_points = result.get("薄弱知识点") or result.get("weak_points") or []
        interests = result.get("资源偏好") or result.get("interests") or []
        if isinstance(weak_points, str):
            weak_points = [weak_points]
        if isinstance(interests, str):
            interests = [interests]

        lowered = text.lower()
        skill_level = "beginner"
        if any(word in lowered for word in ["advanced", "熟练", "进阶", "研究生"]):
            skill_level = "advanced"
        elif any(word in lowered for word in ["intermediate", "基础", "学过", "了解"]):
            skill_level = "intermediate"

        preferred_style = "interactive"
        preference_text = " ".join(str(item) for item in interests)
        if any(word in lowered + preference_text.lower() for word in ["code", "代码", "实践", "lab"]):
            preferred_style = "code_lab"
        elif any(word in lowered + preference_text.lower() for word in ["图", "mindmap", "可视化"]):
            preferred_style = "visual"

        goals = result.get("学习目标") or result.get("goals") or "Build practical AI and LLM application skills"
        if isinstance(goals, str):
            goals = [goals]

        return StudentProfile(
            student_id=student_id,
            name=name or "Demo Student",
            goals=goals,
            interests=list(dict.fromkeys([str(item) for item in weak_points + interests])) or ["AI Applications"],
            skill_level=str(result.get("知识基础") or skill_level),
            preferred_style=preferred_style,
        )

    def _model_info(self, result: dict) -> dict:
        return {
            "model_provider": str(result.get("model_provider", "mock")),
            "model_name": str(result.get("model_name", "mock")),
            "fallback_used": bool(result.get("fallback_used", False)),
        }
