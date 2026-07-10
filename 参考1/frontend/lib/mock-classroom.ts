import type {
  AgentTraceStep,
  ClassroomGenerationInput,
  ClassroomState,
  CodeLabTask,
  CourseMeta,
  JudgeInfo,
  LearningPathItem,
  ProjectTask,
  QuizQuestion,
  ResourceScene,
  Slide,
  SourceChunk,
  StudentProfile
} from "./classroom-types";

export const courseGroupName = "高校人工智能专业能力进阶课程群";

export const sampleCourseCatalog: CourseMeta[] = [
  {
    id: "git",
    name: "Git 与工程协作",
    file: "getting-started-with-git.md",
    level: "基础工具层",
    description: "面向 AI 项目开发的版本控制、分支协作、提交规范与 GitHub 工作流基础。",
    concepts: ["版本控制", "Commit", "Branch", "Merge", "Pull Request", "协作规范"],
    prerequisites: [],
    nextCourses: ["Python 程序设计"],
    resourceTypes: ["讲解文档", "操作练习", "项目实践", "测验"]
  },
  {
    id: "python",
    name: "Python 程序设计",
    file: "python_for_beginners.md",
    level: "基础工具层",
    description: "建立 AI 学习所需的 Python 语法、函数、数据结构、文件处理与调试基础。",
    concepts: ["变量", "控制流", "函数", "数据结构", "文件读写", "异常处理"],
    prerequisites: [],
    nextCourses: ["NumPy 与 Pandas 数据处理", "机器学习"],
    resourceTypes: ["讲解文档", "代码实验", "测验", "项目实践"]
  },
  {
    id: "numpy-pandas",
    name: "NumPy 与 Pandas 数据处理",
    file: "essential-numpy-pandas.md",
    level: "基础工具层",
    description: "掌握数组计算、DataFrame、数据清洗、统计分析与机器学习前置数据处理流程。",
    concepts: ["NumPy 数组", "向量化", "Pandas DataFrame", "数据清洗", "统计分析"],
    prerequisites: ["Python 程序设计"],
    nextCourses: ["机器学习", "PyTorch 深度学习实践"],
    resourceTypes: ["讲解文档", "代码实验", "数据任务", "测验"]
  },
  {
    id: "machine-learning",
    name: "机器学习",
    file: "introduction-to-machine-learning.md",
    level: "核心课程层",
    description: "理解监督学习、无监督学习、模型训练、评估指标与泛化能力。",
    concepts: ["监督学习", "无监督学习", "损失函数", "分类", "回归", "模型评估"],
    prerequisites: ["Python 程序设计", "NumPy 与 Pandas 数据处理"],
    nextCourses: ["神经网络与深度学习", "PyTorch 深度学习实践"],
    resourceTypes: ["讲解文档", "案例练习", "测验", "代码实验"]
  },
  {
    id: "neural-networks",
    name: "神经网络与深度学习",
    file: "introduction-to-neural-networks.md",
    level: "核心课程层",
    description: "学习前向传播、反向传播、激活函数、优化器与深度学习训练基本范式。",
    concepts: ["神经元", "前向传播", "反向传播", "激活函数", "梯度下降", "过拟合"],
    prerequisites: ["机器学习"],
    nextCourses: ["PyTorch 深度学习实践", "Transformer 架构基础", "计算机视觉与 CNN"],
    resourceTypes: ["讲解文档", "白板推导", "测验", "代码实验"]
  },
  {
    id: "pytorch",
    name: "PyTorch 深度学习实践",
    file: "getting-started-with-pytorch.md",
    level: "核心课程层",
    description: "通过 Tensor、Autograd、nn.Module、Dataset 与训练循环完成神经网络代码实践。",
    concepts: ["Tensor", "Autograd", "nn.Module", "Dataset", "DataLoader", "训练循环"],
    prerequisites: ["Python 程序设计", "神经网络与深度学习"],
    nextCourses: ["大语言模型基础", "计算机视觉与 CNN"],
    resourceTypes: ["代码实验", "讲解文档", "项目实践", "测验"]
  },
  {
    id: "computer-vision",
    name: "计算机视觉基础",
    file: "introduction-to-computer-vision.md",
    level: "高级专题层",
    description: "理解图像表示、特征提取、分类、检测与视觉任务的基本流程。",
    concepts: ["图像表示", "卷积", "特征提取", "图像分类", "目标检测"],
    prerequisites: ["Python 程序设计", "机器学习"],
    nextCourses: ["计算机视觉与 CNN"],
    resourceTypes: ["讲解文档", "视觉案例", "测验", "项目实践"]
  },
  {
    id: "cnn",
    name: "计算机视觉与 CNN",
    file: "cnns-for-computer-vision.md",
    level: "高级专题层",
    description: "学习卷积神经网络、池化、经典 CNN 架构与视觉模型训练实践。",
    concepts: ["卷积", "池化", "CNN", "迁移学习", "图像分类"],
    prerequisites: ["神经网络与深度学习", "PyTorch 深度学习实践"],
    nextCourses: ["Transformer 架构基础"],
    resourceTypes: ["讲解文档", "代码实验", "视觉案例", "测验"]
  },
  {
    id: "transformer",
    name: "Transformer 架构基础",
    file: "foundations-transformers-architecture.md",
    level: "高级专题层",
    description: "掌握注意力机制、自注意力、位置编码、Encoder/Decoder 与 Transformer 结构。",
    concepts: ["Attention", "Self-Attention", "位置编码", "Encoder", "Decoder", "Transformer"],
    prerequisites: ["神经网络与深度学习", "PyTorch 深度学习实践"],
    nextCourses: ["大语言模型基础"],
    resourceTypes: ["讲解文档", "白板推导", "测验", "代码实验"]
  },
  {
    id: "llm",
    name: "大语言模型基础",
    file: "how-to-build-a-large-language-model.md",
    level: "高级专题层",
    description: "理解 Tokenizer、Transformer、预训练、微调、Prompt Engineering 与 LLM 应用基础。",
    concepts: ["Token", "Tokenizer", "Transformer", "预训练", "微调", "Prompt Engineering"],
    prerequisites: ["Transformer 架构基础", "PyTorch 深度学习实践"],
    nextCourses: ["RAG 检索增强生成", "RAG 与 LangChain 应用开发", "AI Agent 与多智能体系统"],
    resourceTypes: ["讲解文档", "测验", "代码实验", "项目实践"]
  },
  {
    id: "rag",
    name: "RAG 检索增强生成",
    file: "rag-retrieval-augmented-generation.md",
    level: "应用实践层",
    description: "学习 Chunking、Embedding、向量数据库、Retriever、Generator、Grounding 与 RAG 评估。",
    concepts: ["Chunking", "Embedding", "Vector DB", "Retriever", "Generator", "Grounding", "RAG 评估"],
    prerequisites: ["Python 程序设计", "大语言模型基础"],
    nextCourses: ["RAG 与 LangChain 应用开发", "AI Agent 与多智能体系统"],
    resourceTypes: ["讲解文档", "思维导图", "代码实验", "测验", "项目实践"]
  },
  {
    id: "langchain",
    name: "RAG 与 LangChain 应用开发",
    file: "langchain-production-llm.md",
    level: "应用实践层",
    description: "学习 LCEL、Retriever、Memory、Tool、Callback 与可观测 LLM 应用开发。",
    concepts: ["LCEL", "Retriever", "Memory", "Tool", "Callback", "RAG", "Observability"],
    prerequisites: ["Python 程序设计", "大语言模型基础", "RAG 检索增强生成"],
    nextCourses: ["AI Agent 与多智能体系统"],
    resourceTypes: ["讲解文档", "代码实验", "应用项目", "测验"]
  },
  {
    id: "agent",
    name: "AI Agent 与多智能体系统",
    file: "intro-llm-agents.md",
    level: "应用实践层",
    description: "学习工具调用、规划、记忆、反思、多智能体协作与 Agent 编排基础。",
    concepts: ["Agent", "Tool Calling", "Planning", "Memory", "Reflection", "Multi-Agent", "Orchestration"],
    prerequisites: ["大语言模型基础", "RAG 与 LangChain 应用开发"],
    nextCourses: [],
    resourceTypes: ["讲解文档", "项目实践", "多智能体案例", "测验"]
  },
  {
    id: "reinforcement-learning",
    name: "强化学习",
    file: "advanced-reinforcement-learning.md",
    level: "高级专题层",
    description: "学习 MDP、状态、动作、奖励、Q-learning、DQN、策略梯度与 Actor-Critic。",
    concepts: ["MDP", "状态", "动作", "奖励", "Q-learning", "Policy Gradient", "Actor-Critic"],
    prerequisites: ["机器学习", "神经网络与深度学习"],
    nextCourses: ["RLHF 人类反馈强化学习"],
    resourceTypes: ["讲解文档", "动画脚本", "测验", "代码实验"]
  },
  {
    id: "rlhf",
    name: "RLHF 人类反馈强化学习",
    file: "rlhf-reinforcement-learning-human-feedback.md",
    level: "应用实践层",
    description: "理解人类反馈、偏好数据、奖励模型、PPO 与大模型对齐流程。",
    concepts: ["AI 对齐", "偏好数据", "奖励模型", "PPO", "RLHF", "安全反馈"],
    prerequisites: ["强化学习", "大语言模型基础"],
    nextCourses: ["AI Agent 与多智能体系统"],
    resourceTypes: ["讲解文档", "流程动画", "测验", "案例分析"]
  },
  {
    id: "gnn",
    name: "图神经网络",
    file: "introduction-to-graph-neural-networks.md",
    level: "高级专题层",
    description: "学习图结构数据、消息传递、GCN 与图表示学习基础。",
    concepts: ["图结构", "节点表示", "消息传递", "GCN", "图分类"],
    prerequisites: ["机器学习", "神经网络与深度学习"],
    nextCourses: [],
    resourceTypes: ["讲解文档", "图案例", "测验", "项目实践"]
  }
];

export const defaultSelectedCourseIds = ["rag", "langchain", "agent"];

export const courseLevels = ["基础工具层", "核心课程层", "高级专题层", "应用实践层"];

export const sampleProfile: StudentProfile = {
  major: "计算机 / 软件工程 / 人工智能",
  foundation: "Python 中等，机器学习基础一般",
  goal: "两周内完成课程知识库问答 Agent 项目",
  weakPoints: ["Embedding", "Chunking", "Retriever", "LangChain LCEL", "Agent 编排"],
  preferences: ["案例学习", "代码实验", "项目实践"],
  pace: "每天 1 小时"
};

const sourceChunks: SourceChunk[] = [
  {
    id: "rag-003-chunking",
    courseId: "rag",
    courseName: "RAG 检索增强生成",
    sourceFile: "rag-retrieval-augmented-generation.md",
    chapter: "第 3 章 RAG 基础流程",
    section: "Chunking 与 Embedding",
    headingPath: ["RAG 基础流程", "Chunking 与 Embedding"],
    excerpt: "将课程资料切分为可检索片段，再用 Embedding 建立向量索引，是 RAG grounding 的前置步骤。",
    score: 0.92
  },
  {
    id: "langchain-005-retriever",
    courseId: "langchain",
    courseName: "RAG 与 LangChain 应用开发",
    sourceFile: "langchain-production-llm.md",
    chapter: "第 5 章 Retriever 与 LCEL",
    section: "Retriever Chain",
    headingPath: ["LangChain 应用开发", "Retriever Chain"],
    excerpt: "LCEL 可以把 Retriever、Prompt、LLM 和 Parser 组合成可观测、可复用的 RAG 应用链。",
    score: 0.88
  },
  {
    id: "agent-004-tools",
    courseId: "agent",
    courseName: "AI Agent 与多智能体系统",
    sourceFile: "intro-llm-agents.md",
    chapter: "第 4 章 工具调用与多智能体协作",
    section: "Agent 编排",
    headingPath: ["LLM Agents", "Tool Calling", "Multi-Agent"],
    excerpt: "Agent 通过规划、工具调用、记忆与反思完成复杂任务，多智能体协作可拆分检索、生成、评测与重规划职责。",
    score: 0.9
  }
];

const groundedJudge: JudgeInfo = {
  score: 92,
  grounded: true,
  feedback: "引用覆盖 Chunking、Retriever 与 Agent 编排关键知识点，适合进入项目化练习。"
};

export const sampleAgentTrace: AgentTraceStep[] = [
  {
    agentId: "teacher",
    agentName: "AI 主讲教师",
    role: "概念讲解与课堂节奏",
    status: "running",
    action: "正在讲解 RAG 的 Retriever-Generator 工作流",
    outputSummary: "生成基于课程片段的讲解页",
    durationMs: 1800,
    accent: "from-sky-400 to-indigo-400"
  },
  {
    agentId: "assistant",
    agentName: "AI 助教",
    role: "代码提示与错因解释",
    status: "done",
    action: "已准备 FAISS 检索代码实验",
    outputSummary: "生成可复制的最小代码实验",
    durationMs: 1300,
    accent: "from-violet-400 to-fuchsia-400"
  },
  {
    agentId: "peer",
    agentName: "AI 学伴",
    role: "类比解释与学习陪伴",
    status: "pending",
    action: "等待学生完成第一轮测验后给出类比反馈",
    accent: "from-emerald-300 to-cyan-400"
  },
  {
    agentId: "planner",
    agentName: "规划师",
    role: "学习路径与重规划",
    status: "done",
    action: "规划 14 天 RAG + LangChain + Agent 项目路径",
    durationMs: 1100,
    accent: "from-blue-300 to-cyan-400"
  },
  {
    agentId: "designer",
    agentName: "资源设计师",
    role: "多类型资源生成",
    status: "done",
    action: "生成讲解文档、思维导图、测验、代码实验与 PBL 项目",
    durationMs: 1700,
    accent: "from-amber-300 to-orange-400"
  },
  {
    agentId: "evaluator",
    agentName: "评测官",
    role: "测验诊断与 mastery_map 更新",
    status: "pending",
    action: "等待测验提交后诊断 Embedding / Chunking / Retriever 掌握度",
    accent: "from-rose-300 to-pink-400"
  },
  {
    agentId: "judge",
    agentName: "质量审查员",
    role: "Grounding 与质量审查",
    status: "done",
    action: "已完成课程知识库引用与 Judge 评分",
    outputSummary: "Grounded 检查通过，质量评分 92/100",
    durationMs: 900,
    accent: "from-slate-200 to-sky-300"
  }
];

export const sampleSlides: Slide[] = [
  {
    title: "为什么高校 AI 课程需要 RAG？",
    subtitle: "让大模型回答受课程知识库约束，而不是泛泛生成。",
    bullets: [
      "课程 Markdown 被切分为 chunk，并保留 course_id、章节、heading_path 等元数据。",
      "学生问题先经过 Retriever 命中课程片段，再交给 Generator 组织回答。",
      "JudgeAgent 检查回答是否引用课程来源、是否适合当前学习画像。"
    ],
    example: "课程知识库问答 Agent：用户问 LangChain Retriever 如何接入，系统引用 RAG 与 LangChain 两门课程片段作答。",
    source: "rag-retrieval-augmented-generation.md",
    personalizedReason: "学生目标是两周完成课程知识库问答 Agent，因此先建立 RAG grounding 的系统视角。",
    sourceChunks: sourceChunks.slice(0, 2),
    judgeInfo: groundedJudge
  },
  {
    title: "RAG 基础流程",
    subtitle: "Indexing / Retrieval / Generation / Evaluation",
    bullets: [
      "Indexing：文档解析、Chunking、Embedding、向量索引。",
      "Retrieval：把学习问题转成向量查询，取回 top-k 课程片段。",
      "Generation：将检索片段注入 Prompt，生成有来源约束的学习资源。"
    ],
    example: "PDF / Markdown -> chunk -> embedding -> vector db -> top-k -> grounded answer",
    source: "rag-retrieval-augmented-generation.md",
    sourceChunks: [sourceChunks[0]],
    judgeInfo: groundedJudge
  },
  {
    title: "从 RAG 过渡到 LangChain",
    subtitle: "用 LCEL 把 Retriever、Prompt、LLM、Parser 串成应用链。",
    bullets: [
      "Retriever 负责课程片段检索，Prompt 负责知识约束与学生画像注入。",
      "LCEL 让链路可组合、可替换、可观测，适合课程项目开发。",
      "Callback 和 tracing 为 JudgeAgent 提供可审查的生成过程。"
    ],
    example: "retriever | prompt | llm | parser",
    source: "langchain-production-llm.md",
    personalizedReason: "学生已具备 Python 基础，代码实验会从最小 LCEL 链开始。",
    sourceChunks: [sourceChunks[1]],
    judgeInfo: groundedJudge
  },
  {
    title: "Agent 如何扩展 RAG 项目？",
    subtitle: "把检索、规划、生成、评测、重规划拆给不同智能体。",
    bullets: [
      "PlannerAgent 根据学习目标生成路径与任务拆解。",
      "RetrieverAgent 负责课程知识库 grounding。",
      "JudgeAgent 对资源质量、引用完整性和个性化适配度打分。"
    ],
    example: "ProfileAgent -> RetrieverAgent -> ResourceAgent -> JudgeAgent -> PlannerAgent",
    source: "intro-llm-agents.md",
    personalizedReason: "学生最终目标是多智能体课程问答项目，因此课堂持续展示 Agent 分工。",
    sourceChunks: [sourceChunks[2]],
    judgeInfo: groundedJudge
  }
];

export const sampleWhiteboardChart = `graph TD
  A["高校 AI 课程 Markdown"] --> B["Chunking"]
  B --> C["Embedding"]
  C --> D["Vector DB"]
  E["学生学习目标"] --> F["Query Embedding"]
  F --> G["Retriever top-k"]
  D --> G
  G --> H["Generator 生成课堂资源"]
  H --> I["JudgeAgent grounding 检查"]
  I --> J["Quiz / mastery_map / 重规划"]`;

export const sampleMindmapChart = `mindmap
  root((高校 AI 课程群))
    基础工具层
      Git
      Python
      NumPy/Pandas
    核心课程层
      机器学习
      神经网络
      PyTorch
    应用实践层
      RAG
        Chunking
        Embedding
        Retriever
      LangChain
        LCEL
        Tool
        Memory
      AI Agent
        Planning
        Tool Calling
        Multi-Agent
    闭环反馈
      Quiz
      mastery_map
      Re-planning`;

export const sampleQuiz: QuizQuestion[] = [
  {
    id: "q1",
    question: "在 RAG 中，Embedding 的主要作用是什么？",
    options: ["把文本映射为可检索的向量表示", "直接替代大模型生成答案", "负责网页 UI 渲染", "只用于压缩图片"],
    answerIndex: 0,
    explanation: "Embedding 将文本片段和查询映射到同一向量空间，使 Retriever 能按语义相似度召回课程片段。",
    masteryImpact: "Embedding 掌握度 +8%",
    concept: "Embedding"
  },
  {
    id: "q2",
    question: "Chunk size 过大最可能带来什么问题？",
    options: ["模型一定无法运行", "检索片段包含过多无关信息，降低 grounding 精度", "数据库无法存储任何数据", "Prompt 不再需要课程来源"],
    answerIndex: 1,
    explanation: "Chunk 过大容易把多个主题混在一起，Retriever 命中后会给 Generator 注入噪声。",
    masteryImpact: "Chunking 掌握度 +6%",
    concept: "Chunking"
  },
  {
    id: "q3",
    question: "JudgeAgent 在本系统中的核心职责是什么？",
    options: ["替学生直接完成项目", "检查资源是否 grounded、是否符合学生画像与课程目标", "删除课程知识库", "只负责页面动画"],
    answerIndex: 1,
    explanation: "JudgeAgent 用于审查资源质量、引用依据和个性化适配度，是比赛演示闭环的一部分。",
    masteryImpact: "Evaluation 掌握度 +7%",
    concept: "JudgeAgent"
  }
];

export const sampleCodeLab: CodeLabTask = {
  title: "用最小向量检索实现课程知识库问答原型",
  goal: "通过一个可读的 Python 实验理解 chunk、embedding、top-k retrieval 与 grounded answer 的连接方式。",
  prerequisites: ["Python 基础", "向量表示", "RAG 检索流程"],
  steps: [
    "准备 3 条课程片段作为最小知识库。",
    "使用 Embedding 模型生成文本向量。",
    "建立向量索引并执行 top-k 检索。",
    "把检索结果拼进 Prompt，生成有来源约束的回答。",
    "记录 chunk_id、course_id、source_file，供 JudgeAgent 审查。"
  ],
  code: `docs = [
    {"chunk_id": "rag-003", "course": "RAG 检索增强生成", "text": "Chunking controls retrieval granularity."},
    {"chunk_id": "langchain-005", "course": "RAG 与 LangChain 应用开发", "text": "LCEL composes retriever, prompt, model and parser."},
    {"chunk_id": "agent-004", "course": "AI Agent 与多智能体系统", "text": "Agents plan, call tools, use memory and collaborate."},
]

query = "RAG 项目里 chunking 有什么作用？"
hits = [doc for doc in docs if "Chunking" in doc["text"] or "chunking" in query.lower()]

context = "\\n".join(f"[{doc['chunk_id']}] {doc['text']}" for doc in hits)
answer = f"基于课程片段：\\n{context}\\n\\n回答：Chunking 决定检索粒度，会影响召回精度与生成答案的依据质量。"
print(answer)`,
  runHint: "真实接入时可替换为 sentence-transformers + FAISS / Milvus，这里保留最小可读版本用于课堂讲解。",
  reflection: ["如果 top-k 从 1 改为 3，答案的依据会怎样变化？", "chunk_id 如何帮助 JudgeAgent 判断回答是否 grounded？"]
};

export const sampleProject: ProjectTask = {
  title: "PBL 项目：高校 AI 课程知识库问答 Agent",
  background: "围绕 RAG、LangChain 与 AI Agent 三门课程，完成一个可检索课程 Markdown、能生成学习资源、能测评反馈的课程问答 Agent。",
  goals: ["完成课程 Markdown ingestion 与 chunk manifest", "实现 grounded answer 与来源展示", "生成测验、讲解、代码实验等资源", "用 JudgeAgent 检查资源质量"],
  milestones: ["第 1 周：完成 RAG 检索链路", "第 2 周前半：接入 LangChain LCEL 与工具调用", "第 2 周后半：实现 Agent 编排、评测与演示页面"],
  deliverables: ["可运行的课程问答 Agent", "RAG chunk manifest 与引用展示", "三类个性化学习资源", "测验反馈与 mastery_map 更新"],
  rubric: ["课程知识库 grounding 清晰", "学习路径与画像匹配", "资源类型完整", "Judge 评分和改进建议可解释"]
};

export const sampleResources: ResourceScene[] = [
  {
    id: "res-doc",
    type: "doc",
    title: "RAG 基础流程讲解文档",
    difficulty: "入门",
    estimatedMinutes: 8,
    personalizedReason: "先补齐 Chunking、Embedding、Retriever 的概念链，适合机器学习基础一般的学生。",
    source: "rag-retrieval-augmented-generation.md",
    status: "done",
    courseName: "RAG 检索增强生成",
    targetConcepts: ["Chunking", "Embedding", "Retriever"],
    prerequisiteConcepts: ["Python", "向量表示"],
    sourceChapter: "第 3 章 RAG 基础流程",
    sourceChunks: [sourceChunks[0]],
    judge: groundedJudge
  },
  {
    id: "res-mindmap",
    type: "mindmap",
    title: "RAG + LangChain + Agent 思维导图",
    difficulty: "入门",
    estimatedMinutes: 5,
    personalizedReason: "用图结构把三门课程关系串起来，降低跨课程迁移成本。",
    source: "course_catalog/catalog.json",
    status: "recommended",
    courseName: "高校人工智能专业能力进阶课程群",
    targetConcepts: ["RAG", "LCEL", "Multi-Agent"],
    prerequisiteConcepts: ["大语言模型基础"],
    sourceChapter: "课程目录",
    sourceChunks,
    judge: groundedJudge
  },
  {
    id: "res-quiz",
    type: "quiz",
    title: "Embedding / Chunking / Retriever 诊断测验",
    difficulty: "进阶",
    estimatedMinutes: 6,
    personalizedReason: "针对薄弱点快速诊断，并触发 mastery_map 更新和路径重规划。",
    source: "EvaluationAgent",
    status: "recommended",
    courseName: "RAG 检索增强生成",
    targetConcepts: ["Embedding", "Chunking", "Retriever"],
    prerequisiteConcepts: ["Python", "机器学习基础"],
    sourceChapter: "第 3 章 RAG 基础流程",
    sourceChunks: [sourceChunks[0]],
    judge: groundedJudge
  },
  {
    id: "res-reading",
    type: "reading",
    title: "Grounding 与引用质量拓展阅读",
    difficulty: "挑战",
    estimatedMinutes: 12,
    personalizedReason: "项目进入答辩前需要解释为什么回答可信，这份阅读强化 Judge 视角。",
    source: "rag-retrieval-augmented-generation.md",
    status: "locked",
    courseName: "RAG 检索增强生成",
    targetConcepts: ["Grounding", "RAG 评估"],
    prerequisiteConcepts: ["Retriever", "Prompt Engineering"],
    sourceChapter: "第 6 章 RAG 评估",
    sourceChunks: [sourceChunks[0]],
    judge: { score: 88, grounded: true, feedback: "引用充分，但建议先完成测验后解锁。" }
  },
  {
    id: "res-code",
    type: "code",
    title: "LCEL Retriever 代码实验",
    difficulty: "进阶",
    estimatedMinutes: 25,
    personalizedReason: "学生偏好代码实验，适合把抽象的 Retriever 链路落成可运行原型。",
    source: "langchain-production-llm.md",
    status: "recommended",
    courseName: "RAG 与 LangChain 应用开发",
    targetConcepts: ["LCEL", "Retriever", "Tool"],
    prerequisiteConcepts: ["Python", "RAG 基础流程"],
    sourceChapter: "第 5 章 Retriever 与 LCEL",
    sourceChunks: [sourceChunks[1]],
    judge: groundedJudge
  },
  {
    id: "res-animation",
    type: "animation",
    title: "RAG 检索流程动画脚本",
    difficulty: "入门",
    estimatedMinutes: 7,
    personalizedReason: "用动画脚本解释 query embedding 到 top-k chunk 的过程，适合课堂演示。",
    source: "rag-retrieval-augmented-generation.md",
    status: "recommended",
    courseName: "RAG 检索增强生成",
    targetConcepts: ["Query Embedding", "Vector DB", "top-k"],
    prerequisiteConcepts: ["Embedding"],
    sourceChapter: "第 3 章 RAG 基础流程",
    sourceChunks: [sourceChunks[0]],
    judge: { score: 90, grounded: true, feedback: "脚本对应检索链路，建议配合白板图展示。" }
  },
  {
    id: "res-project",
    type: "project",
    title: "课程知识库问答 Agent PBL",
    difficulty: "挑战",
    estimatedMinutes: 90,
    personalizedReason: "直接对齐两周目标，把 RAG、LangChain、Agent 编排合成一个期末项目。",
    source: "intro-llm-agents.md",
    status: "recommended",
    courseName: "AI Agent 与多智能体系统",
    targetConcepts: ["Tool Calling", "Memory", "Multi-Agent"],
    prerequisiteConcepts: ["RAG", "LCEL"],
    sourceChapter: "第 4 章 工具调用与多智能体协作",
    sourceChunks: [sourceChunks[2]],
    judge: groundedJudge
  }
];

export const sampleLearningPath: LearningPathItem[] = [
  { id: "d1", days: "Day 1-2", title: "大语言模型基础 / Prompt Engineering", courseName: "大语言模型基础", focus: "理解 LLM 输入输出边界", activity: "阅读 + Prompt 改写练习", status: "done" },
  { id: "d2", days: "Day 3-4", title: "RAG / Embedding 与向量数据库", courseName: "RAG 检索增强生成", focus: "建立 chunk 与向量检索概念", activity: "向量检索最小实验", status: "active" },
  { id: "d3", days: "Day 5-6", title: "RAG / 检索增强生成流程", courseName: "RAG 检索增强生成", focus: "完成 top-k 检索到 grounded answer", activity: "课程片段问答练习", status: "pending" },
  { id: "d4", days: "Day 7-9", title: "LangChain / LCEL、Retriever 与工具调用", courseName: "RAG 与 LangChain 应用开发", focus: "用 LCEL 组织 RAG 链", activity: "Retriever chain 代码实验", status: "pending" },
  { id: "d5", days: "Day 10-12", title: "AI Agent / 工具调用、记忆与多智能体协作", courseName: "AI Agent 与多智能体系统", focus: "拆分规划、检索、生成、评测职责", activity: "多 Agent 编排设计", status: "pending" },
  { id: "d6", days: "Day 13-14", title: "综合项目 / 课程知识库问答 Agent", courseName: "AI Agent 与多智能体系统", focus: "整合演示闭环", activity: "PBL 项目答辩", status: "pending" }
];

export const sampleMasteryMap: Record<string, number> = {
  "Prompt Engineering": 78,
  Embedding: 55,
  Chunking: 48,
  Retriever: 44,
  "LangChain LCEL": 36,
  "Agent 编排": 32
};

export const sampleGenerationSteps = sampleAgentTrace.map((step) => ({
  id: step.agentId,
  title: step.action,
  agentName: step.agentName,
  status: step.status === "running" ? "active" : step.status === "done" ? "done" : "pending",
  description: step.outputSummary || step.inputSummary || step.role,
  duration: step.durationMs ? `${(step.durationMs / 1000).toFixed(1)}s` : "等待中"
}));

export function getCourseById(courseId: string) {
  return sampleCourseCatalog.find((course) => course.id === courseId);
}

export function createMockClassroomState(input?: Partial<ClassroomGenerationInput>, selectedCourseIds = defaultSelectedCourseIds): ClassroomState {
  const chosenIds = input?.selectedCourseIds?.length ? input.selectedCourseIds : selectedCourseIds;
  const currentCourses = chosenIds.map((id) => getCourseById(id)).filter((course): course is CourseMeta => Boolean(course));
  const relatedCourses = sampleCourseCatalog.filter((course) => ["llm", "python", "machine-learning", "neural-networks"].includes(course.id));
  const sessionSuffix = Math.random().toString(36).slice(2, 8);

  return {
    sessionId: input?.goal ? `mock-${sessionSuffix}` : "demo",
    title: "RAG + LangChain + AI Agent 个性化课堂",
    courseGroupName,
    currentCourses,
    relatedCourses,
    profile: { ...sampleProfile, goal: input?.goal || sampleProfile.goal },
    progress: 38,
    agentTrace: sampleAgentTrace,
    slides: sampleSlides,
    whiteboardChart: sampleWhiteboardChart,
    mindmapChart: sampleMindmapChart,
    quiz: sampleQuiz,
    codeLab: sampleCodeLab,
    project: sampleProject,
    resources: sampleResources,
    learningPath: sampleLearningPath,
    masteryMap: sampleMasteryMap
  };
}

export const demoClassroomState = createMockClassroomState({ goal: sampleProfile.goal, selectedCourseIds: defaultSelectedCourseIds }, defaultSelectedCourseIds);
