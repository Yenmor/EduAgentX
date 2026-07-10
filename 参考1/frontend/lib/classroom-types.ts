export type CourseMeta = {
  id: string;
  name: string;
  file: string;
  level: string;
  description: string;
  concepts: string[];
  prerequisites: string[];
  nextCourses: string[];
  resourceTypes: string[];
};

export type SourceChunk = {
  id: string;
  courseId: string;
  courseName: string;
  sourceFile: string;
  chapter?: string;
  section?: string;
  headingPath?: string[];
  excerpt: string;
  score?: number;
};

export type JudgeInfo = {
  score: number;
  grounded: boolean;
  feedback: string;
};

export type AgentTraceStep = {
  agentId: string;
  agentName: string;
  role: string;
  status: "pending" | "running" | "done" | "error";
  action: string;
  inputSummary?: string;
  outputSummary?: string;
  startedAt?: string;
  endedAt?: string;
  durationMs?: number;
  accent?: string;
};

export type StudentProfile = {
  major: string;
  foundation: string;
  goal: string;
  weakPoints: string[];
  preferences: string[];
  pace: string;
};

export type Slide = {
  title: string;
  subtitle?: string;
  bullets: string[];
  example?: string;
  source?: string;
  personalizedReason?: string;
  sourceChunks?: SourceChunk[];
  judgeInfo?: JudgeInfo;
};

export type QuizQuestion = {
  id: string;
  question: string;
  options: string[];
  answerIndex: number;
  explanation: string;
  masteryImpact: string;
  concept: string;
};

export type QuizResult = {
  score: number;
  correctCount?: number;
  totalCount?: number;
  weakConcepts: string[];
  updatedMasteryMap: Record<string, number>;
  replanningSuggestions: string[];
};

export type CodeLabTask = {
  title: string;
  goal: string;
  prerequisites: string[];
  steps: string[];
  code: string;
  runHint: string;
  reflection: string[];
};

export type ProjectTask = {
  title: string;
  background: string;
  goals: string[];
  milestones: string[];
  deliverables: string[];
  rubric: string[];
};

export type ResourceScene = {
  id: string;
  type: "doc" | "mindmap" | "quiz" | "reading" | "code" | "animation" | "project";
  title: string;
  difficulty: "入门" | "进阶" | "挑战";
  estimatedMinutes: number;
  personalizedReason: string;
  source: string;
  status?: "locked" | "recommended" | "done";
  courseName: string;
  targetConcepts: string[];
  prerequisiteConcepts: string[];
  sourceChapter: string;
  sourceChunks: SourceChunk[];
  judge: JudgeInfo;
};

export type LearningPathItem = {
  id: string;
  title: string;
  days: string;
  courseName: string;
  focus: string;
  activity: string;
  status?: "done" | "active" | "pending";
};

export type ClassroomState = {
  sessionId: string;
  title: string;
  courseGroupName: string;
  currentCourses: CourseMeta[];
  relatedCourses: CourseMeta[];
  profile: StudentProfile;
  progress: number;
  agentTrace: AgentTraceStep[];
  slides: Slide[];
  whiteboardChart: string;
  mindmapChart: string;
  quiz: QuizQuestion[];
  codeLab: CodeLabTask;
  project: ProjectTask;
  resources: ResourceScene[];
  learningPath: LearningPathItem[];
  masteryMap: Record<string, number>;
};

export type ClassroomGenerationInput = {
  goal: string;
  selectedCourseIds: string[];
};
