import type { ClassroomGenerationInput, ClassroomState, QuizResult } from "./classroom-types";
import { createMockClassroomState, defaultSelectedCourseIds, demoClassroomState, sampleMasteryMap } from "./mock-classroom";

const API_BASE = process.env.NEXT_PUBLIC_API_BASE_URL || "http://localhost:8000";
const mockSessions = new Map<string, ClassroomState>([["demo", demoClassroomState]]);

async function requestJson<T>(path: string, init?: RequestInit): Promise<T> {
  const response = await fetch(`${API_BASE}${path}`, {
    ...init,
    headers: {
      "Content-Type": "application/json",
      ...(init?.headers || {})
    }
  });
  if (!response.ok) {
    throw new Error(`Request failed: ${response.status}`);
  }
  return response.json() as Promise<T>;
}

export async function createMockSession(input: ClassroomGenerationInput): Promise<ClassroomState> {
  const state = createMockClassroomState(input, input.selectedCourseIds?.length ? input.selectedCourseIds : defaultSelectedCourseIds);
  rememberSession(state);
  return state;
}

export async function startClassroomGeneration(input: ClassroomGenerationInput): Promise<ClassroomState> {
  try {
    const raw = await requestJson<unknown>("/api/classroom/sessions", {
      method: "POST",
      body: JSON.stringify({
        goal: input.goal,
        selected_course_ids: input.selectedCourseIds
      })
    });
    const state = learningStateToClassroomState(raw, input);
    rememberSession(state);
    return state;
  } catch {
    return createMockSession(input);
  }
}

export async function getClassroomState(sessionId: string): Promise<ClassroomState> {
  try {
    const raw = await requestJson<unknown>(`/api/classroom/sessions/${encodeURIComponent(sessionId)}`);
    const state = learningStateToClassroomState(raw);
    rememberSession(state);
    return state;
  } catch {
    const cached = mockSessions.get(sessionId);
    if (cached) return cached;
    if (typeof window !== "undefined") {
      const stored = window.sessionStorage.getItem(`eduagentx:classroom:${sessionId}`);
      if (stored) {
        const parsed = JSON.parse(stored) as ClassroomState;
        mockSessions.set(sessionId, parsed);
        return parsed;
      }
    }
    return sessionId === "demo" ? demoClassroomState : createMockClassroomState({ goal: "高校 AI 课程群学习", selectedCourseIds: defaultSelectedCourseIds });
  }
}

export async function submitQuizAnswers(sessionId: string, answers: Record<string, number>): Promise<QuizResult> {
  try {
    const raw = await requestJson<unknown>(`/api/classroom/sessions/${encodeURIComponent(sessionId)}/quiz`, {
      method: "POST",
      body: JSON.stringify({ answers })
    });
    return evaluationToQuizResult(raw);
  } catch {
    const state = await getClassroomState(sessionId);
    const correctCount = state.quiz.reduce((sum, question) => sum + (answers[question.id] === question.answerIndex ? 1 : 0), 0);
    const weakConcepts = state.quiz.filter((question) => answers[question.id] !== question.answerIndex).map((question) => question.concept);
    return {
      score: Math.round((correctCount / state.quiz.length) * 100),
      correctCount,
      totalCount: state.quiz.length,
      weakConcepts,
      updatedMasteryMap: {
        ...sampleMasteryMap,
        Embedding: Math.min(100, sampleMasteryMap.Embedding + 8),
        Chunking: Math.min(100, sampleMasteryMap.Chunking + (weakConcepts.includes("Chunking") ? 3 : 8)),
        Retriever: Math.min(100, sampleMasteryMap.Retriever + 6)
      },
      replanningSuggestions: weakConcepts.length
        ? [`补强 ${weakConcepts.join("、")} 后再进入 LangChain LCEL 代码实验。`, "追加 1 个 RAG 检索流程动画脚本帮助建立直觉。"]
        : ["测验表现良好，可提前进入 LCEL Retriever 代码实验。"]
    };
  }
}

export async function replanLearningPath(sessionId: string): Promise<ClassroomState> {
  try {
    const raw = await requestJson<unknown>(`/api/classroom/sessions/${encodeURIComponent(sessionId)}/replan`, {
      method: "POST",
      body: JSON.stringify({ reason: "manual" })
    });
    const state = learningStateToClassroomState(raw);
    rememberSession(state);
    return state;
  } catch {
    return getClassroomState(sessionId);
  }
}

export function subscribeGenerationEvents(sessionId: string, onEvent: (event: MessageEvent) => void): EventSource | null {
  if (typeof window === "undefined") return null;
  const source = new EventSource(`${API_BASE}/api/classroom/sessions/${encodeURIComponent(sessionId)}/stream`);
  ["agent_start", "agent_update", "agent_done", "profile_updated", "course_catalog_loaded", "retrieval_done", "diagnosis_done", "plan_done", "resource_generated", "judge_done", "final"].forEach((eventName) => {
    source.addEventListener(eventName, onEvent);
  });
  return source;
}

export function learningStateToClassroomState(raw: unknown, input?: Partial<ClassroomGenerationInput>): ClassroomState {
  const data = raw as Partial<ClassroomState> & Record<string, unknown>;
  if (data.sessionId && data.currentCourses && data.agentTrace) {
    return data as ClassroomState;
  }
  if (data.session_id) {
    const fallback = createMockClassroomState({ goal: String(data.user_message || input?.goal || ""), selectedCourseIds: input?.selectedCourseIds || defaultSelectedCourseIds });
    return {
      ...fallback,
      sessionId: String(data.session_id),
      title: "RAG + LangChain + AI Agent 个性化课堂"
    };
  }
  if (data.session_id && data.classroom_state) {
    return data.classroom_state as ClassroomState;
  }
  return createMockClassroomState({ goal: input?.goal || "高校 AI 课程群学习", selectedCourseIds: input?.selectedCourseIds || defaultSelectedCourseIds });
}

export function evaluationToQuizResult(raw: unknown): QuizResult {
  const data = raw as Record<string, unknown>;
  return {
    score: Number(data.score || 0),
    correctCount: Number(data.correctCount ?? data.correct_count ?? 0),
    totalCount: Number(data.totalCount ?? data.total_count ?? 0),
    weakConcepts: (data.weakConcepts || data.weak_concepts || []) as string[],
    updatedMasteryMap: (data.updatedMasteryMap || data.updated_mastery_map || {}) as Record<string, number>,
    replanningSuggestions: (data.replanningSuggestions || data.replanning_suggestions || []) as string[]
  };
}

function rememberSession(state: ClassroomState) {
  mockSessions.set(state.sessionId, state);
  if (typeof window !== "undefined") {
    window.sessionStorage.setItem(`eduagentx:classroom:${state.sessionId}`, JSON.stringify(state));
  }
}
