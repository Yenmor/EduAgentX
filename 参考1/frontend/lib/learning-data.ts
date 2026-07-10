export type ResourceType = "讲义" | "速览课件" | "思维导图" | "题库" | "拓展阅读" | "代码案例" | "HTML 交互模拟";

export type Course = {
  id: string;
  title: string;
  summary: string;
  category: string;
  difficulty: "入门" | "进阶" | "高级";
  goal: string;
  duration: string;
  outcomes: string[];
  resourceTypes: ResourceType[];
  recommendedFor: string[];
  score: number;
};

export type Resource = {
  id: string;
  title: string;
  type: ResourceType;
  courseId: string;
  description: string;
  status: "已生成" | "可生成" | "待复核";
};

export type ProfileDimension = {
  label: string;
  value: string;
};

export type RoadmapStageData = {
  id: string;
  phase: string;
  goal: string;
  courses: string[];
  resources: string[];
  quiz: string;
  estimatedTime: string;
};

export type QuizType = {
  id: string;
  title: string;
  description: string;
  fit: string;
};

export type QuizQuestion = {
  id: string;
  stem: string;
  options: string[];
  answer: number;
  weakPoint: string;
  analysis: string;
};

export type AgentRunStep = {
  agent: string;
  input: string;
  output: string;
  duration: string;
  status: "完成" | "复核中" | "已过滤";
};

export const apiPlaceholders = [
  "POST /api/profile/chat",
  "POST /api/courses/recommend",
  "POST /api/resources/generate",
  "POST /api/roadmap/generate",
  "POST /api/quiz/generate",
  "POST /api/quiz/submit",
  "POST /api/progress/score",
  "GET /api/agent-runs/:runId",
  "GET /api/agent-runs/:runId/events"
];

export const courses: Course[] = [
  {
    id: "intro-ai",
    title: "人工智能导论",
    summary: "从问题建模、机器学习、搜索、知识表示到生成式 AI，建立完整的 AI 学科地图。",
    category: "人工智能",
    difficulty: "入门",
    goal: "课程补强",
    duration: "6 周",
    outcomes: ["解释 AI 核心分支", "读懂典型模型流程", "完成一个小型分类项目"],
    resourceTypes: ["讲义", "速览课件", "思维导图", "题库", "HTML 交互模拟"],
    recommendedFor: ["零基础转入 AI", "课程预习", "期末复习"],
    score: 96
  },
  {
    id: "machine-learning",
    title: "机器学习基础",
    summary: "围绕监督学习、无监督学习、模型评估与特征工程，训练可落地的建模能力。",
    category: "人工智能",
    difficulty: "进阶",
    goal: "项目实践",
    duration: "8 周",
    outcomes: ["搭建训练流程", "解释过拟合与正则化", "完成模型评估报告"],
    resourceTypes: ["讲义", "题库", "代码案例", "拓展阅读"],
    recommendedFor: ["有 Python 基础", "科研入门", "就业面试"],
    score: 91
  },
  {
    id: "python-data",
    title: "Python 数据分析",
    summary: "用 NumPy、Pandas 与可视化工具完成数据清洗、分析与报告表达。",
    category: "编程基础",
    difficulty: "入门",
    goal: "课程补强",
    duration: "4 周",
    outcomes: ["清洗表格数据", "完成统计摘要", "制作可复现实验 notebook"],
    resourceTypes: ["讲义", "代码案例", "题库"],
    recommendedFor: ["编程薄弱", "数据课程先修", "项目实践"],
    score: 88
  },
  {
    id: "deep-learning",
    title: "深度学习与神经网络",
    summary: "从 MLP、CNN、优化器到训练技巧，理解深度模型的结构与实验方法。",
    category: "人工智能",
    difficulty: "高级",
    goal: "科研入门",
    duration: "10 周",
    outcomes: ["读懂网络结构图", "调试训练曲线", "复现实验 baseline"],
    resourceTypes: ["讲义", "速览课件", "代码案例", "拓展阅读"],
    recommendedFor: ["机器学习后续", "科研入门", "竞赛训练"],
    score: 86
  },
  {
    id: "llm-apps",
    title: "大模型应用开发",
    summary: "学习提示词、RAG、工具调用与评测，把大模型能力封装为可验证的应用。",
    category: "生成式 AI",
    difficulty: "进阶",
    goal: "就业面试",
    duration: "5 周",
    outcomes: ["设计 RAG 流程", "实现工具调用", "构建评测样例"],
    resourceTypes: ["讲义", "代码案例", "HTML 交互模拟", "拓展阅读"],
    recommendedFor: ["项目作品集", "工程实践", "就业面试"],
    score: 89
  }
];

export const resources: Resource[] = [
  { id: "r1", courseId: "intro-ai", title: "AI 学科地图讲义", type: "讲义", description: "按搜索、学习、知识、生成式 AI 梳理核心概念。", status: "已生成" },
  { id: "r2", courseId: "intro-ai", title: "人工智能导论 15 分钟速览", type: "速览课件", description: "面向课前预习和期末串讲的短课件。", status: "已生成" },
  { id: "r3", courseId: "intro-ai", title: "AI 核心概念思维导图", type: "思维导图", description: "展示算法、数据、模型、评测之间的关系。", status: "待复核" },
  { id: "r4", courseId: "intro-ai", title: "诊断题库 A", type: "题库", description: "覆盖定义、判断、案例分析和轻量计算。", status: "已生成" },
  { id: "r5", courseId: "machine-learning", title: "线性模型代码案例", type: "代码案例", description: "包含数据划分、训练、评估与错误分析。", status: "可生成" },
  { id: "r6", courseId: "llm-apps", title: "RAG 检索流程交互模拟", type: "HTML 交互模拟", description: "演示切分、召回、重排、生成与引用检查。", status: "可生成" },
  { id: "r7", courseId: "deep-learning", title: "优化器拓展阅读包", type: "拓展阅读", description: "精选 SGD、Adam 与学习率调度阅读材料。", status: "已生成" }
];

export const profileDimensions: ProfileDimension[] = [
  { label: "专业背景", value: "计算机大二，修过程序设计" },
  { label: "学历年级", value: "本科二年级" },
  { label: "知识基础", value: "Python 较稳，概率统计偏弱" },
  { label: "学习目标", value: "补齐 AI 导论并准备课程项目" },
  { label: "学习历史", value: "近 14 天完成 7 个学习单元" },
  { label: "认知风格", value: "先看结构图，再做题巩固" },
  { label: "易错点", value: "模型评估、过拟合、搜索策略" },
  { label: "学习节奏", value: "工作日 45 分钟，周末 2 小时" },
  { label: "资源偏好", value: "讲义 + 代码案例 + 短测验" }
];

export const roadmapStages: RoadmapStageData[] = [
  {
    id: "stage-1",
    phase: "阶段 1",
    goal: "建立人工智能导论的全局框架",
    courses: ["人工智能导论"],
    resources: ["AI 学科地图讲义", "人工智能导论 15 分钟速览"],
    quiz: "诊断测验",
    estimatedTime: "3 天"
  },
  {
    id: "stage-2",
    phase: "阶段 2",
    goal: "补强机器学习基本概念与评估方法",
    courses: ["机器学习基础", "Python 数据分析"],
    resources: ["诊断题库 A", "线性模型代码案例"],
    quiz: "章节测验：模型评估",
    estimatedTime: "7 天"
  },
  {
    id: "stage-3",
    phase: "阶段 3",
    goal: "完成一个可展示的小项目并沉淀复盘",
    courses: ["大模型应用开发"],
    resources: ["RAG 检索流程交互模拟", "拓展阅读包"],
    quiz: "代码实操题",
    estimatedTime: "10 天"
  }
];

export const quizTypes: QuizType[] = [
  { id: "diagnostic", title: "诊断测验", description: "快速识别基础、误区和适合的学习路径。", fit: "新课程开始前" },
  { id: "chapter", title: "章节测验", description: "围绕当前章节生成 8-12 道题。", fit: "每章结束后" },
  { id: "final", title: "综合测验", description: "跨章节混合考查概念、推理与应用。", fit: "考试复习" },
  { id: "wrong", title: "错题重练", description: "从薄弱点和错题分布生成相似题。", fit: "复习质量提升" },
  { id: "custom", title: "自定义测验", description: "用自然语言描述范围、难度和题型。", fit: "个性化训练" },
  { id: "code", title: "代码实操题", description: "生成可运行的代码任务和评分点。", fit: "项目实践" }
];

export const quizQuestions: QuizQuestion[] = [
  {
    id: "q1",
    stem: "如果一个模型在训练集表现很好、测试集表现明显变差，最可能的问题是？",
    options: ["欠拟合", "过拟合", "数据增强", "早停成功"],
    answer: 1,
    weakPoint: "模型泛化",
    analysis: "训练集与测试集差距大通常意味着模型记住了训练数据细节，需要正则化、更多数据或更简单模型。"
  },
  {
    id: "q2",
    stem: "人工智能导论中，搜索算法最常被用来解决哪类问题？",
    options: ["已知目标但路径未知的问题", "图像像素压缩", "数据库索引维护", "前端样式适配"],
    answer: 0,
    weakPoint: "问题建模",
    analysis: "搜索适合把状态、动作、目标与代价显式建模的问题，例如路径规划或博弈。"
  },
  {
    id: "q3",
    stem: "选择评估指标时，类别极不平衡的分类任务更应关注什么？",
    options: ["文件大小", "Accuracy 之外的 Precision/Recall/F1", "训练日志颜色", "变量命名风格"],
    answer: 1,
    weakPoint: "模型评估",
    analysis: "类别不平衡时 Accuracy 可能虚高，Precision、Recall、F1 更能反映少数类表现。"
  }
];

export const progressMetrics = [
  { label: "完成度", value: 78 },
  { label: "测验表现", value: 84 },
  { label: "学习连续性", value: 71 },
  { label: "复习质量", value: 66 },
  { label: "应用能力", value: 59 }
];

export const agentRunSteps: AgentRunStep[] = [
  { agent: "Orchestrator Agent", input: "学生输入学习需求与当前进度", output: "拆解为画像更新、课程推荐、路径生成、测验生成任务", duration: "0.18s", status: "完成" },
  { agent: "Profile Agent", input: "自然语言需求、历史测验、学习节奏", output: "更新 9 个学习画像维度", duration: "0.42s", status: "完成" },
  { agent: "Course Curator Agent", input: "画像、课程目录、资源元数据", output: "推荐人工智能导论、机器学习基础和 Python 数据分析", duration: "0.36s", status: "完成" },
  { agent: "Content Agent", input: "课程目标与薄弱点", output: "生成导论讲义与复习摘要", duration: "0.64s", status: "完成" },
  { agent: "Mindmap Agent", input: "知识点依赖关系", output: "生成 AI 核心概念思维导图", duration: "0.31s", status: "复核中" },
  { agent: "Quiz Agent", input: "薄弱点：模型评估、过拟合、搜索策略", output: "生成诊断测验、错题重练与综合题", duration: "0.55s", status: "完成" },
  { agent: "Code Lab Agent", input: "项目实践目标与 Python 基础", output: "生成线性模型实操案例", duration: "0.48s", status: "完成" },
  { agent: "Media Agent", input: "讲义摘要与知识结构", output: "生成速览课件与 HTML 交互模拟脚本", duration: "0.71s", status: "完成" },
  { agent: "Roadmap Agent", input: "画像、进度、测验结果", output: "生成 3 阶段动态学习路径", duration: "0.29s", status: "完成" },
  { agent: "Assessment Agent", input: "完成度、测验、连续性、复习、应用数据", output: "计算总评分 74/100 和下一步建议", duration: "0.22s", status: "完成" },
  { agent: "Safety & Fact-check Agent", input: "生成内容、引用和题目答案", output: "通过事实校验；过滤无来源结论 1 条", duration: "0.33s", status: "已过滤" }
];

export const generationEvents = [
  { label: "画像更新", percent: 100, detail: "9 个维度已更新" },
  { label: "课程推荐", percent: 100, detail: "5 门课程完成匹配" },
  { label: "资源生成", percent: 86, detail: "讲义、课件、题库、代码案例、交互模拟已完成" },
  { label: "事实校验", percent: 92, detail: "来源引用与题目答案复核中" }
];
