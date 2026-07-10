export type AgentTraceStep = {
  agent: string;
  agent_name?: string;
  action: string;
  status?: string;
  model_provider?: string;
  model_name?: string;
  fallback_used?: boolean;
  output?: unknown;
};

export type SourceRef = {
  chunk_id?: string;
  content?: string;
  source_file?: string;
  module_name?: string;
  heading_path?: string[];
  score?: number;
};

export type OrchestratorResource = {
  title: string;
  type: "explanation_doc" | "mindmap" | "quiz" | "reading_material" | "code_lab" | string;
  content: string;
  sources?: SourceRef[];
  judge_score?: number;
  personalized_reason?: string;
};

export type ChatResponse = {
  reply: string;
  agent: string;
  intent: string;
  result: OrchestratorResource[] | Record<string, unknown>;
  judge?: {
    pass: boolean;
    score: number;
    issues: string[];
    rewrite_instruction: string;
  };
  rewritten?: boolean;
  agent_trace?: AgentTraceStep[];
};

export type LearningStep = {
  order: number;
  title: string;
  knowledge_point: string;
  activity: string;
  estimated_minutes: number;
  mastery_target: number;
};

export type StudentProfile = {
  student_id: string;
  name: string;
  goals: string[];
  interests: string[];
  skill_level: string;
  preferred_style: string;
};
