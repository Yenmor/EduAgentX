from __future__ import annotations

from collections.abc import Iterator
from datetime import datetime, timezone
from time import perf_counter
from uuid import uuid4

from backend.agents.diagnosis_agent import DiagnosisAgent
from backend.agents.evaluation_agent import EvaluationAgent
from backend.agents.judge_agent import JudgeAgent
from backend.agents.planner_agent import PlannerAgent
from backend.agents.profile_agent import ProfileAgent
from backend.agents.resource_agent_group import ResourceAgentGroup
from backend.agents.resource_agent import ResourceAgent
from backend.agents.retriever_agent import RetrieverAgent
from backend.agents.tutor_agent import TutorAgent
from backend.schemas.chat import ChatRequest
from backend.schemas.learning_state import AgentTraceStep, LearningState
from backend.schemas.learning_path import LearningPathRequest
from backend.schemas.resource import SourceChunk
from backend.services.course_catalog_service import get_all_courses, get_course_by_id, get_course_source_file


class EduAgentOrchestrator:
    """Coordinates EduAgentX agents from student message to final response."""

    def __init__(
        self,
        profile_agent: ProfileAgent | None = None,
        planner_agent: PlannerAgent | None = None,
        retriever_agent: RetrieverAgent | None = None,
        resource_agent: ResourceAgent | None = None,
        tutor_agent: TutorAgent | None = None,
        judge_agent: JudgeAgent | None = None,
    ) -> None:
        self.profile_agent = profile_agent or ProfileAgent()
        self.planner_agent = planner_agent or PlannerAgent()
        self.retriever_agent = retriever_agent or RetrieverAgent()
        self.resource_agent = resource_agent or ResourceAgent()
        self.resource_agent_group = ResourceAgentGroup()
        self.tutor_agent = tutor_agent or TutorAgent()
        self.judge_agent = judge_agent or JudgeAgent()
        self.diagnosis_agent = DiagnosisAgent()
        self.evaluation_agent = EvaluationAgent()

    def generate_classroom_state(
        self,
        goal: str,
        selected_course_ids: list[str] | None = None,
        session_id: str | None = None,
        student_id: str | None = None,
    ) -> LearningState:
        """Run the classroom multi-agent chain and return a unified LearningState."""
        selected_course_ids = selected_course_ids or ["rag", "langchain", "agent"]
        state = LearningState(
            session_id=session_id or f"cls-{uuid4().hex[:10]}",
            student_id=student_id or "demo-student",
            user_message=goal,
            selected_course_ids=selected_course_ids,
            mastery_map={},
            mode="real",
        )

        profile_payload = self._run_classroom_step(
            state,
            agent_id="profile",
            agent_name="ProfileAgent",
            role="对话式学习画像",
            action="分析学习目标并生成学生画像",
            fn=lambda: self.profile_agent.extract_profile(state.student_id or "demo-student", goal),
            fallback=lambda: None,
        )
        state.profile = profile_payload

        courses = self._run_classroom_step(
            state,
            agent_id="course_catalog",
            agent_name="CourseCatalogService",
            role="课程目录服务",
            action="加载目标课程、先修课程与知识点",
            fn=lambda: [get_course_by_id(course_id) for course_id in selected_course_ids if get_course_by_id(course_id)],
            fallback=lambda: [course for course in get_all_courses() if course.get("id") in selected_course_ids],
        )
        state.course_metadata = courses or []

        retrieved_chunks = self._run_classroom_step(
            state,
            agent_id="retriever",
            agent_name="RetrieverAgent",
            role="RAG 检索与课程 grounding",
            action="检索课程知识库片段",
            fn=lambda: self._retrieve_course_chunks(goal, selected_course_ids, state.course_metadata),
            fallback=lambda: self._fallback_course_chunks(selected_course_ids, state.course_metadata),
        )
        state.retrieved_chunks = retrieved_chunks

        diagnosis = self._run_classroom_step(
            state,
            agent_id="diagnosis",
            agent_name="DiagnosisAgent",
            role="知识短板诊断",
            action="诊断薄弱点和推荐学习重点",
            fn=lambda: self.diagnosis_agent.diagnose(goal, state.profile, state.course_metadata, state.retrieved_chunks),
            fallback=lambda: self.diagnosis_agent.diagnose(goal, state.profile, state.course_metadata, state.retrieved_chunks),
        )
        state.diagnosis = diagnosis
        state.mastery_map = diagnosis.mastery_estimate if diagnosis else {}

        learning_path = self._run_classroom_step(
            state,
            agent_id="planner",
            agent_name="PlannerAgent",
            role="个性化学习路径规划",
            action="规划 RAG + LangChain + Agent 学习路径",
            fn=lambda: self.planner_agent.generate_path(
                LearningPathRequest(student_id=state.student_id or "demo-student", goal=goal, horizon_days=14),
                profile=state.profile,
            ),
            fallback=lambda: self.planner_agent.generate_path(
                LearningPathRequest(student_id=state.student_id or "demo-student", goal=goal, horizon_days=14),
                profile=None,
            ),
        )
        state.learning_path = learning_path

        resources = self._run_classroom_step(
            state,
            agent_id="resource_group",
            agent_name="ResourceAgentGroup",
            role="多类型资源子 Agent",
            action="生成讲解、思维导图、测验、拓展阅读、代码实验、动画脚本和 PBL 项目",
            fn=lambda: self.resource_agent_group.generate(state.profile, state.diagnosis, state.retrieved_chunks, state.learning_path, state.course_metadata),
            fallback=lambda: [],
        )
        state.resources = resources

        judge = self._run_classroom_step(
            state,
            agent_id="judge",
            agent_name="JudgeAgent",
            role="质量审查与防幻觉检查",
            action="检查资源 grounding、质量分和个性化适配",
            fn=lambda: self.judge_agent.judge_resources(state.resources, state.retrieved_chunks),
            fallback=lambda: self.judge_agent.judge_resources(state.resources, state.retrieved_chunks),
        )
        state.judge = judge
        return state

    def stream_classroom_state(
        self,
        goal: str,
        selected_course_ids: list[str] | None = None,
        session_id: str | None = None,
        student_id: str | None = None,
    ) -> Iterator[dict]:
        """Synchronous generator of SSE-ready classroom events."""
        state = self.generate_classroom_state(goal, selected_course_ids, session_id, student_id)
        event_map = {
            "profile": "profile_updated",
            "course_catalog": "course_catalog_loaded",
            "retriever": "retrieval_done",
            "diagnosis": "diagnosis_done",
            "planner": "plan_done",
            "resource_group": "resource_generated",
            "judge": "judge_done",
        }
        for step in state.agent_trace:
            yield {"event": "agent_done", "data": self._dump_model(step)}
            if step.agent_id in event_map:
                yield {"event": event_map[step.agent_id], "data": {"session_id": state.session_id, "trace_step": self._dump_model(step)}}
        yield {"event": "final", "data": self._dump_model(state)}

    def _run_classroom_step(self, state: LearningState, agent_id: str, agent_name: str, role: str, action: str, fn, fallback):
        started = datetime.now(timezone.utc)
        start = perf_counter()
        try:
            result = fn()
            status = "done"
            output_summary = self._summary(result)
        except Exception as exc:
            result = fallback()
            status = "error"
            output_summary = f"fallback used: {exc}"
            state.mode = "mock"
        ended = datetime.now(timezone.utc)
        state.agent_trace.append(
            AgentTraceStep(
                agent_id=agent_id,
                agent_name=agent_name,
                role=role,
                status=status,
                action=action,
                input_summary=state.user_message[:180],
                output_summary=output_summary,
                started_at=started.isoformat(),
                ended_at=ended.isoformat(),
                duration_ms=int((perf_counter() - start) * 1000),
            )
        )
        return result

    def _retrieve_course_chunks(self, goal: str, selected_course_ids: list[str], courses: list[dict]) -> list[SourceChunk]:
        retrieved = self.retriever_agent.retrieve(goal, top_k=6)
        raw_results = retrieved.get("results", [])
        chunks = [self._source_chunk_from_raw(item, courses) for item in raw_results]
        return chunks or self._fallback_course_chunks(selected_course_ids, courses)

    def _fallback_course_chunks(self, selected_course_ids: list[str], courses: list[dict]) -> list[SourceChunk]:
        course_lookup = {str(course.get("id")): course for course in courses}
        chunks: list[SourceChunk] = []
        for index, course_id in enumerate(selected_course_ids[:3], start=1):
            course = course_lookup.get(course_id) or get_course_by_id(course_id) or {}
            source_path = get_course_source_file(course_id)
            excerpt = ""
            if source_path and source_path.exists():
                text = source_path.read_text(encoding="utf-8", errors="ignore")
                excerpt = " ".join(line.strip() for line in text.splitlines() if line.strip() and not line.startswith("---"))[:320]
            chunks.append(
                SourceChunk(
                    id=f"{course_id}-fallback-{index}",
                    course_id=course_id,
                    course_name=str(course.get("name") or self._course_display_name(course_id)),
                    source_file=str(course.get("file") or (source_path.name if source_path else "")),
                    chapter="课程核心章节",
                    section="知识库片段",
                    heading_path=[str(course.get("name") or self._course_display_name(course_id)), "知识库片段"],
                    excerpt=excerpt or f"{self._course_display_name(course_id)} 的课程知识片段，用于课堂资源 grounding。",
                    score=0.82,
                )
            )
        return chunks

    def _source_chunk_from_raw(self, item: dict, courses: list[dict]) -> SourceChunk:
        source_file = str(item.get("source_file", ""))
        matched = next((course for course in courses if str(course.get("file")) in source_file), {})
        heading_path = item.get("heading_path") or []
        if not isinstance(heading_path, list):
            heading_path = [str(heading_path)]
        return SourceChunk(
            id=str(item.get("chunk_id") or item.get("id") or uuid4().hex[:8]),
            course_id=str(item.get("course_id") or matched.get("id", "")),
            course_name=str(item.get("course_name") or matched.get("name") or "高校 AI 课程知识库"),
            source_file=source_file,
            chapter=str(item.get("chapter") or (heading_path[0] if heading_path else "")),
            section=str(item.get("section") or (heading_path[-1] if heading_path else "")),
            heading_path=[str(part) for part in heading_path],
            excerpt=str(item.get("excerpt") or item.get("content") or "")[:420],
            score=float(item.get("score", 0.8) or 0.8),
        )

    def _course_display_name(self, course_id: str) -> str:
        names = {
            "rag": "RAG 检索增强生成",
            "langchain": "RAG 与 LangChain 应用开发",
            "agent": "AI Agent 与多智能体系统",
            "llm": "大语言模型基础",
            "python": "Python 程序设计",
        }
        return names.get(course_id, course_id)

    def _summary(self, result) -> str:
        if isinstance(result, list):
            return f"{len(result)} item(s)"
        if hasattr(result, "topic"):
            return str(getattr(result, "topic"))
        if hasattr(result, "score"):
            return f"score={getattr(result, 'score')}"
        if hasattr(result, "name"):
            return str(getattr(result, "name"))
        return "completed"

    def _dump_model(self, model):
        return model.model_dump() if hasattr(model, "model_dump") else model.dict() if hasattr(model, "dict") else model

    def run(self, request: ChatRequest) -> dict:
        state: dict | None = None
        for event in self.stream(request, emit_tokens=False):
            if event["event"] == "final":
                state = event["data"]
        if state is None:
            raise RuntimeError("Orchestrator did not produce a final response.")
        return state

    def stream(self, request: ChatRequest, emit_tokens: bool = True) -> Iterator[dict]:
        """Yield named workflow events for the SSE chat endpoint."""
        trace: list[dict] = []
        message = request.message

        profile_result = self.profile_agent.extract_profile_payload(request.student_id, message)
        profile = profile_result["profile"]
        profile_payload = profile.model_dump() if hasattr(profile, "model_dump") else profile.dict()
        profile_step = self._trace_step("ProfileAgent", "updated_profile", profile_result, "ok", {"profile": profile_payload})
        trace.append(profile_step)
        yield {
            "event": "profile_update",
            "data": {"profile": profile_payload, "trace_step": profile_step, "agent_trace": trace},
        }

        intent = self._classify_intent(message, request.mode)
        intent_step = self._trace_step(
            "PlannerAgent",
            "classified_intent",
            {"model_provider": "rule", "model_name": "intent-rules", "fallback_used": False},
            "ok",
            {"intent": intent},
        )
        trace.append(intent_step)
        yield {
            "event": "intent_detected",
            "data": {"intent": intent, "trace_step": intent_step, "agent_trace": trace},
        }

        yield {"event": "retrieve_start", "data": {"query": message, "agent_trace": trace}}
        retrieved = self.retriever_agent.retrieve(message, top_k=4)
        sources = retrieved.get("results", [])
        retrieve_step = self._trace_step(
            "RetrieverAgent",
            "retrieved_sources",
            {"model_provider": "rag", "model_name": "mock-embedding", "fallback_used": False},
            "ok" if sources else "empty",
            {"count": len(sources), "sources": [item.get("source_file") for item in sources]},
        )
        trace.append(retrieve_step)
        yield {
            "event": "retrieve_done",
            "data": {"results": sources, "trace_step": retrieve_step, "agent_trace": trace},
        }

        generator_name = self._generator_name(intent)
        yield {"event": "generate_start", "data": {"agent": generator_name, "intent": intent, "agent_trace": trace}}
        result = self._generate(intent, message, request.student_id, profile_payload, sources)
        generator_info = self._result_model_info(result)
        generate_step = self._trace_step(generator_name, "generated_content", generator_info, "ok")
        trace.append(generate_step)
        reply = self._reply_from_result(intent, result)
        if emit_tokens:
            yield from self._token_events(reply)

        content_for_judge = self._content_for_judge(result)
        yield {"event": "judge_start", "data": {"agent": "JudgeAgent", "agent_trace": trace}}
        review = self.judge_agent.judge_content(content_for_judge, sources=sources, profile=profile_payload, intent=intent)
        judge_step = self._trace_step("JudgeAgent", "reviewed", review, "pass" if review["pass"] else "rewrite", review)
        trace.append(judge_step)
        yield {"event": "judge_done", "data": {"judge": review, "trace_step": judge_step, "agent_trace": trace}}

        rewritten = False
        if not review["pass"]:
            rewritten = True
            yield {
                "event": "generate_start",
                "data": {
                    "agent": generator_name,
                    "intent": intent,
                    "rewrite": True,
                    "instruction": review["rewrite_instruction"],
                    "agent_trace": trace,
                },
            }
            result = self._generate(
                intent,
                message,
                request.student_id,
                profile_payload,
                sources,
                rewrite_instruction=review["rewrite_instruction"],
            )
            generator_info = self._result_model_info(result)
            rewrite_step = self._trace_step(generator_name, "rewrote_content", generator_info, "ok")
            trace.append(rewrite_step)
            reply = self._reply_from_result(intent, result)
            if emit_tokens:
                yield from self._token_events("\n\nRewritten: " + reply)

            content_for_judge = self._content_for_judge(result)
            yield {"event": "judge_start", "data": {"agent": "JudgeAgent", "rewrite": True, "agent_trace": trace}}
            review = self.judge_agent.judge_content(content_for_judge, sources=sources, profile=profile_payload, intent=intent)
            judge_step = self._trace_step(
                "JudgeAgent",
                "reviewed_after_rewrite",
                review,
                "pass" if review["pass"] else "failed",
                review,
            )
            trace.append(judge_step)
            yield {"event": "judge_done", "data": {"judge": review, "trace_step": judge_step, "agent_trace": trace}}

        self._attach_judge_score(result, review["score"])
        final_reply = self._reply_from_result(intent, result)
        yield {
            "event": "final",
            "data": {
                "reply": final_reply,
                "agent": "EduAgentOrchestrator",
                "intent": intent,
                "result": result,
                "judge": review,
                "rewritten": rewritten,
                "agent_trace": trace,
                **self._result_model_info(result),
            },
        }

    def _classify_intent(self, message: str, explicit_mode: str = "auto") -> str:
        intent = self.planner_agent.classify_intent(message, explicit_mode)
        if intent != "qa":
            return intent
        lowered = message.lower()
        if any(key in lowered for key in ["学习资料", "学习资源", "生成资料", "练习题", "测验", "资源", "resource"]):
            return "generate_resource"
        if any(key in lowered for key in ["学习路径", "路线", "规划", "计划", "path", "plan"]):
            return "plan_learning_path"
        if any(key in lowered for key in ["画像", "偏好", "基础", "目标", "profile"]):
            return "update_profile"
        return "qa"

    def _generate(
        self,
        intent: str,
        message: str,
        student_id: str,
        profile: dict,
        sources: list[dict],
        rewrite_instruction: str | None = None,
    ) -> dict | list[dict]:
        if intent == "generate_resource":
            topic = self._topic_from_message(message, sources)
            return self.resource_agent.generate_orchestrated_resources(topic, sources, profile, rewrite_instruction)
        if intent == "plan_learning_path":
            path = self.planner_agent.generate_path(
                LearningPathRequest(student_id=student_id, goal=message, horizon_days=14),
                profile=None,
            )
            return path.model_dump() if hasattr(path, "model_dump") else path.dict()
        if intent == "update_profile":
            return {
                "profile": profile,
                "message": "Profile updated from the latest student message.",
                **self.profile_agent.last_model_info,
            }
        return self.tutor_agent.answer_question(message, sources, profile, rewrite_instruction)

    def _topic_from_message(self, message: str, sources: list[dict]) -> str:
        known_topics = ["RAG", "Self-Attention", "Embedding", "Function Calling", "LangGraph", "Agent", "Transformer", "RLHF"]
        for key in known_topics:
            if key.lower() in message.lower():
                return key
        if sources:
            heading = sources[0].get("heading_path") or []
            if heading:
                return str(heading[-1])
        return "大模型应用开发"

    def _content_for_judge(self, result: dict | list[dict]) -> str:
        if isinstance(result, list):
            return "\n".join(str(item.get("content", "")) for item in result)
        return str(result.get("answer") or result.get("message") or result)

    def _attach_judge_score(self, result: dict | list[dict], score: int) -> None:
        if isinstance(result, list):
            for item in result:
                item["judge_score"] = score
        else:
            result["judge_score"] = score

    def _reply_from_result(self, intent: str, result: dict | list[dict]) -> str:
        if intent == "generate_resource" and isinstance(result, list):
            titles = "、".join(str(item.get("title", item.get("type", "resource"))) for item in result[:5])
            return f"已生成 {len(result)} 类个性化学习资源：{titles}"
        if isinstance(result, dict) and "answer" in result:
            return str(result["answer"])
        if isinstance(result, dict) and "steps" in result:
            return f"已生成包含 {len(result['steps'])} 个步骤的个性化学习路径。"
        if isinstance(result, dict) and "message" in result:
            return str(result["message"])
        return "EduAgentX completed the requested workflow."

    def _generator_name(self, intent: str) -> str:
        if intent == "generate_resource":
            return "ResourceAgent"
        if intent == "plan_learning_path":
            return "PlannerAgent"
        if intent == "update_profile":
            return "ProfileAgent"
        return "TutorAgent"

    def _result_model_info(self, result: dict | list[dict]) -> dict:
        if isinstance(result, list) and result:
            return {
                "model_provider": str(result[0].get("model_provider", "mock")),
                "model_name": str(result[0].get("model_name", "mock")),
                "fallback_used": any(bool(item.get("fallback_used", False)) for item in result),
            }
        if isinstance(result, dict):
            return {
                "model_provider": str(result.get("model_provider", "mock")),
                "model_name": str(result.get("model_name", "mock")),
                "fallback_used": bool(result.get("fallback_used", False)),
            }
        return {"model_provider": "mock", "model_name": "mock", "fallback_used": False}

    def _trace_step(
        self,
        agent_name: str,
        action: str,
        model_info: dict,
        status: str,
        output: dict | None = None,
    ) -> dict:
        return {
            "agent": agent_name,
            "agent_name": agent_name,
            "action": action,
            "model_provider": model_info.get("model_provider", "unknown"),
            "model_name": model_info.get("model_name", "unknown"),
            "fallback_used": bool(model_info.get("fallback_used", False)),
            "status": status,
            "output": output or {},
        }

    def _token_events(self, text: str) -> Iterator[dict]:
        words = text.split()
        if not words:
            return
        for index, word in enumerate(words):
            suffix = "" if index == len(words) - 1 else " "
            yield {"event": "token", "data": {"content": word + suffix}}
