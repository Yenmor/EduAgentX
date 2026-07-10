# EduAgentX Agent Architecture

EduAgentX is a personalized resource generation and multi-agent learning system for an AI-major course group. The backend keeps the existing agents and adds a classroom-oriented orchestration path around `LearningState`.

## Agent Roles

- `ProfileAgent`: builds a student profile from the learning goal and historical profile hints.
- `CourseCatalogService`: loads selected course metadata, prerequisites, next courses, and concepts.
- `RetrieverAgent`: retrieves course knowledge chunks with `course_id`, `course_name`, `source_file`, `chapter`, `section`, and `heading_path`.
- `DiagnosisAgent`: diagnoses topic, related courses, weak concepts, mastery estimates, and recommended focus.
- `PlannerAgent`: creates a staged `LearningPathResponse`.
- `ResourceAgentGroup`: generates doc, mindmap, quiz, reading, code lab, animation, and project resources.
- `TutorAgent`: remains available for classroom explanation and existing chat flows.
- `JudgeAgent`: checks resource grounding, quality, and profile fit, then writes `judge_score`, `judge_feedback`, and `grounded`.
- `EvaluationAgent`: scores quiz answers, updates `mastery_map`, detects weak concepts, and emits replanning suggestions.

## Orchestrator Flow

`EduAgentOrchestrator.generate_classroom_state()` runs:

1. `ProfileAgent`
2. `CourseCatalogService`
3. `RetrieverAgent`
4. `DiagnosisAgent`
5. `PlannerAgent`
6. `ResourceAgentGroup`
7. `JudgeAgent`

Each step is wrapped by `_run_classroom_step()`. Failures are recorded as `error` trace entries and use fallback outputs so the demo can still return a usable `LearningState`.

## LearningState

`backend/schemas/learning_state.py` defines `LearningState` with session id, selected courses, profile, retrieved chunks, diagnosis, learning path, resources, evaluation, judge result, mastery map, mode, and agent trace.

The classroom API currently uses an in-memory session store. Sessions are lost after backend restart.

## AgentTraceStep

Each `AgentTraceStep` includes `agent_id`, `agent_name`, `role`, `status`, `action`, optional input/output summaries, timestamps, and duration. The frontend consumes these as `agentTrace`.

## ResourceAgentGroup

`ResourceAgentGroup` is additive and does not remove the original `ResourceAgent`. It exposes multiple sub-agent methods:

- DocAgent
- MindmapAgent
- QuizAgent
- ReadingAgent
- CodeLabAgent
- AnimationAgent
- ProjectAgent

Every generated `LearningResource` includes target concepts, prerequisite concepts, source chunks, and a personalized reason.

## Grounding And Judge

`JudgeAgent.judge_resources()` uses rule scoring:

- source chunks present
- personalized reason present
- target concepts present
- reasonable content length
- non-empty content

The review is summarized as `JudgeResult` and written back to each resource.

## Evaluation

`EvaluationAgent.evaluate_quiz()` returns `EvaluationResult` with score, correct count, total count, weak concepts, updated mastery map, and replanning suggestions.

## SSE Events

`GET /api/classroom/sessions/{session_id}/stream` returns `text/event-stream` and emits:

- `agent_start`
- `agent_done`
- `profile_updated`
- `course_catalog_loaded`
- `retrieval_done`
- `diagnosis_done`
- `plan_done`
- `resource_generated`
- `judge_done`
- `final`

The frontend wrapper `subscribeGenerationEvents()` is prepared to consume these events with `EventSource`.
