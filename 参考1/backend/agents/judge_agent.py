from __future__ import annotations

from backend.schemas.chat import ChatResponse
from backend.schemas.judge import JudgeResult
from backend.schemas.resource import LearningResource, SourceChunk
from backend.schemas.resource import ResourceGenerateResponse
from backend.services.safety_service import SafetyService
from backend.services.spark_wrapper import SparkWrapper


JUDGE_SYSTEM_PROMPT = (
    "你是内容质量评审智能体。请从学术准确性、是否基于资料、个性化适配、表达清晰度和安全性五个维度评分，"
    "输出 JSON：{\"pass\": true, \"score\": 0, \"issues\": [], \"rewrite_instruction\": \"\"}"
)


class JudgeAgent:
    """Scores generated content and attaches revision advice when quality is low."""

    def __init__(self, llm: SparkWrapper | None = None) -> None:
        self.safety = SafetyService()
        self.llm = llm or SparkWrapper()
        self.last_model_info = {"model_provider": "mock", "model_name": "mock", "fallback_used": False}

    def review(self, response: ChatResponse) -> ChatResponse:
        review = self.judge_content(response.reply)
        response.reply = f"{response.reply}\n\nJudge score: {review['score']}/100."
        if not review["pass"]:
            response.suggestions.extend(review["issues"])
        response.model_provider = review["model_provider"]
        response.model_name = review["model_name"]
        response.fallback_used = review["fallback_used"]
        return response

    def review_resources(self, response: ResourceGenerateResponse, required_terms: list[str] | None = None) -> ResourceGenerateResponse:
        combined = "\n".join(resource.content for resource in response.resources)
        review = self.judge_content(combined, intent="generate_resource")
        response.judge_score = int(review["score"])
        response.revision_suggestions = review["issues"] if not review["pass"] else []
        response.model_provider = review["model_provider"]
        response.model_name = review["model_name"]
        response.fallback_used = review["fallback_used"]
        return response

    def judge_content(
        self,
        content: str,
        sources: list[dict] | None = None,
        profile: dict | None = None,
        intent: str = "qa",
    ) -> dict:
        """Return a structured quality review for orchestrated agent outputs."""
        sources = sources or []
        profile = profile or {}
        model_result = self.llm.generate_json(
            [
                {"role": "system", "content": JUDGE_SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": f"intent={intent}\nprofile={profile}\nsources={sources[:2]}\ncontent={content[:1600]}",
                },
            ],
            temperature=0.2,
        )
        model_info = self._model_info(model_result)
        rule_review = self._rule_review(content, sources, profile, intent)
        score = int(model_result.get("score", rule_review["score"])) if not model_result.get("parse_error") else rule_review["score"]
        issues = model_result.get("issues", rule_review["issues"])
        if not isinstance(issues, list):
            issues = [str(issues)]
        passed = bool(model_result.get("pass", score >= 80)) if not model_result.get("parse_error") else rule_review["pass"]
        rewrite_instruction = str(
            model_result.get("rewrite_instruction", rule_review["rewrite_instruction"] if not passed else "")
        )
        review = {
            "pass": passed and score >= 80,
            "score": max(0, min(100, score)),
            "issues": issues,
            "rewrite_instruction": "" if passed and score >= 80 else rewrite_instruction,
            **model_info,
        }
        self.last_model_info = model_info
        return review

    def judge_resources(self, resources: list[LearningResource], retrieved_chunks: list[SourceChunk]) -> JudgeResult:
        scores: list[float] = []
        risks: list[str] = []
        for resource in resources:
            score = 0
            if resource.source_chunks:
                score += 30
            if resource.personalized_reason:
                score += 20
            if resource.target_concepts:
                score += 20
            if 40 <= len(resource.content) <= 3000:
                score += 10
            if resource.content.strip():
                score += 20
            grounded = bool(resource.source_chunks or retrieved_chunks)
            resource.judge_score = float(score)
            resource.grounded = grounded
            resource.judge_feedback = "资源已绑定课程片段并通过基础 grounded 检查。" if grounded else "资源缺少课程片段引用，需要补充 source_chunks。"
            if not grounded:
                risks.append(f"{resource.title} lacks source chunks.")
            scores.append(float(score))
        avg_score = round(sum(scores) / len(scores), 1) if scores else 0.0
        return JudgeResult(
            score=avg_score,
            grounded=avg_score >= 80 and not risks,
            feedback="资源整体通过 grounded 与个性化质量审查。" if avg_score >= 80 and not risks else "部分资源需要补充引用或个性化理由。",
            risks=risks,
            suggestions=[] if avg_score >= 80 and not risks else ["为每个资源补充 1-3 个课程 source_chunks", "明确资源适配学生画像的原因"],
        )

    def _rule_review(self, content: str, sources: list[dict], profile: dict, intent: str) -> dict:
        score = 100
        issues: list[str] = []
        if len(content.strip()) < 80:
            score -= 18
            issues.append("Expression is too brief to be useful.")
        if intent in {"qa", "generate_resource"} and not sources:
            score -= 25
            issues.append("No retrieved course source was used.")
        if not profile.get("skill_level"):
            score -= 6
            issues.append("Student profile adaptation is weak.")
        if any(term in content.lower() for term in ["dangerous", "harmful"]):
            score -= 20
            issues.append("Potential safety issue detected.")
        score = max(0, min(100, score))
        return {
            "pass": score >= 80,
            "score": score,
            "issues": issues,
            "rewrite_instruction": "" if score >= 80 else "Rewrite once with clearer structure, student-level language, and explicit grounding in retrieved sources.",
        }

    def _model_info(self, result: dict) -> dict:
        return {
            "model_provider": str(result.get("model_provider", "mock")),
            "model_name": str(result.get("model_name", "mock")),
            "fallback_used": bool(result.get("fallback_used", False)),
        }
