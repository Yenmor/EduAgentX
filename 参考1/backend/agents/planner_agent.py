from __future__ import annotations

from backend.schemas.chat import ChatRequest, ChatResponse
from backend.schemas.learning_path import LearningPathRequest, LearningPathResponse, LearningStep
from backend.schemas.profile import StudentProfile
from backend.services.spark_wrapper import SparkWrapper


PLANNER_SYSTEM_PROMPT = "你是个性化学习路径规划智能体。请根据学生画像、知识掌握度和课程目标生成学习路径，输出 JSON。"


class PlannerAgent:
    """Produces staged learning paths from a profile and learning goal."""

    def __init__(self, llm: SparkWrapper | None = None) -> None:
        self.llm = llm or SparkWrapper()
        self.last_model_info = {"model_provider": "mock", "model_name": "mock", "fallback_used": False}

    def run(self, request: ChatRequest) -> ChatResponse:
        text = self.llm.generate_text(
            [
                {"role": "system", "content": PLANNER_SYSTEM_PROMPT},
                {"role": "user", "content": request.message},
            ],
            role="planner-agent",
        )
        response = self.llm.generate([{"role": "user", "content": request.message}], role="planner-agent")
        self.last_model_info = self._model_info(response)
        return ChatResponse(
            reply=text,
            agent="PlannerAgent",
            suggestions=["Week 1: Prompt", "Week 2: RAG", "Week 3: Agent workflow"],
            **self.last_model_info,
        )

    def classify_intent(self, message: str, explicit_mode: str = "auto") -> str:
        """Classify a student message into one of the orchestrator intents."""
        mode_map = {
            "resource": "generate_resource",
            "planner": "plan_learning_path",
            "profile": "update_profile",
            "generate_resource": "generate_resource",
            "plan_learning_path": "plan_learning_path",
            "update_profile": "update_profile",
            "qa": "qa",
        }
        if explicit_mode in mode_map:
            return mode_map[explicit_mode]
        lowered = message.lower()
        if any(word in lowered for word in ["资料", "resource", "生成", "学习资料", "练习", "quiz"]):
            return "generate_resource"
        if any(word in lowered for word in ["路径", "计划", "plan", "learning path", "规划"]):
            return "plan_learning_path"
        if any(word in lowered for word in ["我的目标", "我的基础", "画像", "profile", "我想学"]):
            return "update_profile"
        return "qa"

    def generate_path(self, request: LearningPathRequest, profile: StudentProfile | None = None) -> LearningPathResponse:
        model_result = self.llm.generate_json(
            [
                {"role": "system", "content": PLANNER_SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": f"学生画像：{profile.model_dump() if profile else {}}；学习目标：{request.goal}",
                },
            ],
            temperature=0.2,
        )
        self.last_model_info = self._model_info(model_result)
        steps = self._steps_from_model(model_result, profile=profile)
        mermaid_edges = "\n".join(
            f"  S{step.order}[{step.title}] --> S{steps[index + 1].order}[{steps[index + 1].title}]"
            for index, step in enumerate(steps[:-1])
        )
        return LearningPathResponse(
            student_id=request.student_id,
            goal=request.goal,
            steps=steps,
            mermaid="graph TD\n" + mermaid_edges,
            **self.last_model_info,
        )

    def _steps_from_model(self, result: dict, profile: StudentProfile | None = None) -> list[LearningStep]:
        raw_steps = result.get("steps") if isinstance(result, dict) else None
        if isinstance(raw_steps, list) and raw_steps:
            steps = []
            for index, item in enumerate(raw_steps[:6]):
                steps.append(
                    LearningStep(
                        order=index + 1,
                        title=str(item.get("title", f"Step {index + 1}")),
                        knowledge_point=str(item.get("knowledge_point", item.get("title", "LLM Application"))),
                        activity=str(item.get("activity", "Complete concept review and practice.")),
                        estimated_minutes=int(item.get("estimated_minutes", 40)),
                        mastery_target=float(item.get("mastery_target", round(0.55 + index * 0.06, 2))),
                    )
                )
            return steps
        level = profile.skill_level if profile else "beginner"
        base_minutes = 45 if level == "beginner" else 35
        topics = [
            ("Prompt Engineering", "Structure role, task, context, output constraints, and examples."),
            ("Embedding and Vector Search", "Understand semantic vectors and similarity retrieval."),
            ("RAG", "Build retrieval-augmented generation with grounded sources."),
            ("Function Calling", "Design tool schemas and backend execution boundaries."),
            ("Agent Workflow", "Coordinate planning, tools, memory, and Judge review."),
            ("Evaluation and Anti-Hallucination", "Use citations, rubrics, and safety checks."),
        ]
        return [
            LearningStep(
                order=index + 1,
                title=title,
                knowledge_point=title,
                activity=activity,
                estimated_minutes=base_minutes + index * 10,
                mastery_target=round(0.55 + index * 0.06, 2),
            )
            for index, (title, activity) in enumerate(topics)
        ]

    def _model_info(self, result: dict) -> dict:
        return {
            "model_provider": str(result.get("model_provider", "mock")),
            "model_name": str(result.get("model_name", "mock")),
            "fallback_used": bool(result.get("fallback_used", False)),
        }
