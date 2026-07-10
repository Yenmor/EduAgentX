from __future__ import annotations

import json
from collections.abc import Iterator

from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from backend.agents.evaluation_agent import EvaluationAgent
from backend.agents.orchestrator import EduAgentOrchestrator
from backend.schemas.learning_path import LearningStep
from backend.schemas.learning_state import LearningState


router = APIRouter(prefix="/api/classroom", tags=["classroom"])
orchestrator = EduAgentOrchestrator()
evaluation_agent = EvaluationAgent()
SESSION_STORE: dict[str, LearningState] = {}


class ClassroomCreateRequest(BaseModel):
    goal: str = Field(..., min_length=1)
    selected_course_ids: list[str] = Field(default_factory=lambda: ["rag", "langchain", "agent"])
    student_id: str | None = None


class QuizSubmitRequest(BaseModel):
    answers: dict[str, int] = Field(default_factory=dict)


class ReplanRequest(BaseModel):
    reason: str | None = None
    weak_concepts: list[str] = Field(default_factory=list)


@router.post("/sessions")
def create_session(request: ClassroomCreateRequest) -> dict:
    try:
        state = orchestrator.generate_classroom_state(
            goal=request.goal,
            selected_course_ids=request.selected_course_ids,
            student_id=request.student_id,
        )
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Failed to create classroom session: {exc}") from exc
    SESSION_STORE[state.session_id] = state
    return _learning_state_to_classroom_state(state)


@router.get("/sessions/{session_id}")
def get_session(session_id: str) -> dict:
    state = SESSION_STORE.get(session_id)
    if state is None and session_id == "demo":
        state = orchestrator.generate_classroom_state(
            goal="我学过深度学习，但不了解 RAG 和 LangChain，想完成一个课程知识库问答项目",
            selected_course_ids=["rag", "langchain", "agent"],
            session_id="demo",
        )
        SESSION_STORE[session_id] = state
    if state is None:
        raise HTTPException(status_code=404, detail="Classroom session not found. Sessions are stored in memory and may be lost after backend restart.")
    return _learning_state_to_classroom_state(state)


@router.get("/sessions/{session_id}/stream")
def stream_session(session_id: str) -> StreamingResponse:
    state = SESSION_STORE.get(session_id)
    if state is None:
        raise HTTPException(status_code=404, detail="Classroom session not found.")

    def event_iter() -> Iterator[str]:
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
            yield _sse("agent_start", {"session_id": state.session_id, "agent_id": step.agent_id, "agent_name": step.agent_name, "role": step.role, "action": step.action})
            yield _sse("agent_done", _dump(step))
            if step.agent_id in event_map:
                yield _sse(event_map[step.agent_id], {"session_id": state.session_id, "trace_step": _dump(step)})
        yield _sse("final", _learning_state_to_classroom_state(state))

    return StreamingResponse(event_iter(), media_type="text/event-stream")


@router.post("/sessions/{session_id}/quiz")
def submit_quiz(session_id: str, request: QuizSubmitRequest) -> dict:
    state = SESSION_STORE.get(session_id)
    if state is None:
        raise HTTPException(status_code=404, detail="Classroom session not found.")
    answer_key = {
        "q1": {"answer_index": 0, "concept": "Embedding"},
        "q2": {"answer_index": 1, "concept": "Chunking"},
        "q3": {"answer_index": 1, "concept": "JudgeAgent"},
    }
    result = evaluation_agent.evaluate_quiz(request.answers, answer_key, state.mastery_map)
    state.evaluation = result
    state.mastery_map = result.updated_mastery_map
    return {
        "score": result.score,
        "correctCount": result.correct_count,
        "totalCount": result.total_count,
        "weakConcepts": result.weak_concepts,
        "updatedMasteryMap": {key: round(value * 100) if value <= 1 else value for key, value in result.updated_mastery_map.items()},
        "replanningSuggestions": result.replanning_suggestions,
    }


@router.post("/sessions/{session_id}/replan")
def replan(session_id: str, request: ReplanRequest | None = None) -> dict:
    state = SESSION_STORE.get(session_id)
    if state is None:
        raise HTTPException(status_code=404, detail="Classroom session not found.")
    weak_concepts = request.weak_concepts if request else []
    if weak_concepts and state.learning_path:
        state.learning_path.steps.insert(0,
            LearningStep(
                order=0,
                title="补强薄弱知识点",
                knowledge_point=" / ".join(weak_concepts),
                activity="完成针对性讲解、测验订正和 FAISS 检索 CodeLab",
                estimated_minutes=45,
                mastery_target=0.68,
            ),
        )
    return _learning_state_to_classroom_state(state)


def _learning_state_to_classroom_state(state: LearningState) -> dict:
    courses = state.course_metadata or []
    current_courses = [_course_to_front(course) for course in courses]
    related = [_course_to_front(course) for course in courses]
    resources = [_resource_to_front(resource) for resource in state.resources]
    first_chunks = [_source_to_front(chunk) for chunk in state.retrieved_chunks[:2]]
    return {
        "session_id": state.session_id,
        "sessionId": state.session_id,
        "mode": state.mode,
        "title": "RAG + LangChain + AI Agent 个性化课堂",
        "courseGroupName": "高校人工智能专业能力进阶课程群",
        "currentCourses": current_courses,
        "relatedCourses": related,
        "profile": {
            "major": "计算机 / 软件工程 / 人工智能",
            "foundation": "Python 中等，机器学习基础一般",
            "goal": state.user_message,
            "weakPoints": state.diagnosis.weak_points if state.diagnosis else ["Embedding", "Chunking", "Retriever"],
            "preferences": ["案例学习", "代码实验", "项目实践"],
            "pace": "每天 1 小时",
        },
        "progress": 38,
        "agentTrace": [_trace_to_front(step) for step in state.agent_trace],
        "slides": [
            {
                "title": "为什么高校 AI 课程需要 RAG？",
                "subtitle": "让大模型回答受课程知识库约束，而不是泛泛生成。",
                "bullets": [
                    "课程 Markdown 被切分为 chunk，并保留 course_id、章节和 heading_path。",
                    "Retriever 命中课程片段后交给 Generator 组织回答。",
                    "JudgeAgent 检查资源是否 grounded 且适合学生画像。",
                ],
                "example": "课程知识库问答 Agent：引用 RAG、LangChain 与 Agent 课程片段回答学习问题。",
                "source": first_chunks[0]["sourceFile"] if first_chunks else "rag-retrieval-augmented-generation.md",
                "personalizedReason": state.diagnosis.reason if state.diagnosis else "",
                "sourceChunks": first_chunks,
                "judgeInfo": {
                    "score": state.judge.score if state.judge else 88,
                    "grounded": state.judge.grounded if state.judge else True,
                    "feedback": state.judge.feedback if state.judge else "已通过基础 grounded 检查。",
                },
            }
        ],
        "whiteboardChart": "graph TD\n  A[课程 Markdown] --> B[Chunking]\n  B --> C[Embedding]\n  C --> D[Retriever]\n  D --> E[Generator]\n  E --> F[JudgeAgent]\n  F --> G[mastery_map 与重规划]",
        "mindmapChart": "mindmap\n  root((高校 AI 课程群))\n    RAG\n      Chunking\n      Embedding\n      Retriever\n    LangChain\n      LCEL\n      Tool\n    AI Agent\n      Planning\n      Multi-Agent",
        "quiz": [
            {"id": "q1", "question": "在 RAG 中，Embedding 的主要作用是什么？", "options": ["把文本映射为可检索的向量表示", "直接替代大模型生成答案", "负责 UI 渲染", "只用于压缩图片"], "answerIndex": 0, "explanation": "Embedding 让文本和查询进入同一语义向量空间。", "masteryImpact": "Embedding +8%", "concept": "Embedding"},
            {"id": "q2", "question": "Chunk size 过大最可能带来什么问题？", "options": ["模型无法运行", "检索片段包含过多无关信息", "数据库无法存储", "Prompt 不需要来源"], "answerIndex": 1, "explanation": "过大的 chunk 容易带入噪声，降低 grounding 精度。", "masteryImpact": "Chunking +6%", "concept": "Chunking"},
            {"id": "q3", "question": "JudgeAgent 的核心职责是什么？", "options": ["替学生完成项目", "检查资源 grounding、质量和画像适配", "删除知识库", "只负责动画"], "answerIndex": 1, "explanation": "JudgeAgent 负责质量审查与防幻觉检查。", "masteryImpact": "JudgeAgent +7%", "concept": "JudgeAgent"},
        ],
        "codeLab": {
            "title": "用最小向量检索实现课程知识库问答原型",
            "goal": "理解 chunk、embedding、top-k retrieval 与 grounded answer 的连接方式。",
            "prerequisites": ["Python 基础", "向量表示", "RAG 检索流程"],
            "steps": ["准备课程片段", "生成向量", "执行 top-k 检索", "拼接 Prompt", "记录 source_chunks"],
            "code": "docs = ['Chunking controls retrieval granularity', 'LCEL composes chains']\nprint(docs[0])",
            "runHint": "真实实现可替换为 sentence-transformers + FAISS / Milvus。",
            "reflection": ["top-k 从 1 改为 3 会怎样？", "chunk_id 如何帮助 JudgeAgent？"],
        },
        "project": {
            "title": "PBL 项目：高校 AI 课程知识库问答 Agent",
            "background": "围绕 RAG、LangChain 与 AI Agent 三门课程完成可演示项目。",
            "goals": ["完成 ingestion 与 chunk manifest", "实现 grounded answer", "生成多类型资源", "加入 Judge 审查"],
            "milestones": ["第 1 周：RAG 检索链路", "第 2 周：LCEL 与 Agent 编排", "答辩：闭环演示"],
            "deliverables": ["问答 Agent", "source_chunks 展示", "测验反馈", "重规划建议"],
            "rubric": ["grounding 清晰", "路径匹配画像", "资源完整", "Judge 可解释"],
        },
        "resources": resources,
        "learningPath": _path_to_front(state),
        "masteryMap": {key: round(value * 100) if value <= 1 else value for key, value in state.mastery_map.items()},
    }


def _course_to_front(course: dict) -> dict:
    return {
        "id": str(course.get("id", "")),
        "name": str(course.get("name", course.get("id", ""))),
        "file": str(course.get("file", "")),
        "level": str(course.get("level", "")),
        "description": str(course.get("description", "")),
        "concepts": list(course.get("concepts", [])),
        "prerequisites": list(course.get("prerequisites", [])),
        "nextCourses": list(course.get("nextCourses", [])),
        "resourceTypes": list(course.get("resourceTypes", [])),
    }


def _source_to_front(chunk) -> dict:
    return {
        "id": chunk.id,
        "courseId": chunk.course_id,
        "courseName": chunk.course_name,
        "sourceFile": chunk.source_file,
        "chapter": chunk.chapter,
        "section": chunk.section,
        "headingPath": chunk.heading_path,
        "excerpt": chunk.excerpt,
        "score": chunk.score,
    }


def _resource_to_front(resource) -> dict:
    difficulty = "进阶" if resource.difficulty == "中等" else resource.difficulty
    return {
        "id": resource.id,
        "type": resource.type,
        "title": resource.title,
        "difficulty": difficulty,
        "estimatedMinutes": resource.estimated_minutes,
        "personalizedReason": resource.personalized_reason,
        "source": resource.source_chunks[0].source_file if resource.source_chunks else "course knowledge base",
        "status": "recommended",
        "courseName": resource.course_name,
        "targetConcepts": resource.target_concepts,
        "prerequisiteConcepts": resource.prerequisite_concepts,
        "sourceChapter": resource.source_chunks[0].chapter if resource.source_chunks else "课程核心章节",
        "sourceChunks": [_source_to_front(chunk) for chunk in resource.source_chunks],
        "judge": {
            "score": resource.judge_score or 0,
            "grounded": bool(resource.grounded),
            "feedback": resource.judge_feedback or "",
        },
    }


def _trace_to_front(step) -> dict:
    return {
        "agentId": step.agent_id,
        "agentName": step.agent_name,
        "role": step.role,
        "status": step.status,
        "action": step.action,
        "inputSummary": step.input_summary,
        "outputSummary": step.output_summary,
        "startedAt": step.started_at,
        "endedAt": step.ended_at,
        "durationMs": step.duration_ms,
    }


def _path_to_front(state: LearningState) -> list[dict]:
    if not state.learning_path:
        return []
    return [
        {
            "id": f"step-{step.order}",
            "days": f"Step {step.order}",
            "title": step.title,
            "courseName": step.knowledge_point,
            "focus": step.knowledge_point,
            "activity": step.activity,
            "status": "active" if index == 0 else "pending",
        }
        for index, step in enumerate(state.learning_path.steps)
    ]


def _dump(model) -> dict:
    return model.model_dump() if hasattr(model, "model_dump") else model.dict()


def _sse(event: str, data: dict) -> str:
    return f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"
