---
course_id: rag
course_name: RAG 检索增强生成
level: LLM 应用层
source_file: rag-retrieval-augmented-generation.md
knowledge_base: 高校人工智能专业能力进阶课程群
---

# RAG 检索增强生成

## 课程定位

RAG（Retrieval-Augmented Generation，检索增强生成）是连接高校课程知识库与大语言模型应用的重要技术。本课程面向已经具备 Python 和大语言模型基础的学习者，重点训练其构建可追溯、可评估、低幻觉的知识库问答系统的能力。

## 先修课程

- Python 程序设计
- NumPy 与 Pandas 数据处理
- 大语言模型基础

## 学习目标

完成本课程后，学生应能够：

1. 解释 RAG 与纯 LLM 问答的区别。
2. 对课程资料进行合理切分并构建 chunk 元数据。
3. 使用 Embedding 模型将文本转化为向量表示。
4. 使用向量数据库或相似度检索实现 Retriever。
5. 将检索结果组织为 grounded prompt，生成可追溯回答。
6. 设计 JudgeAgent 或评测流程检查事实一致性、引用质量和幻觉风险。
7. 完成一个课程知识库问答 Agent 项目。

## Chapter 1 为什么需要 RAG

大语言模型具备较强的语言理解和生成能力，但它的参数知识并不总是包含某门课程的最新资料、教师讲义、实验说明或学生个性化背景。RAG 的核心思想是：在生成回答之前，先从外部知识库中检索与问题相关的内容，再让模型基于这些内容生成答案。

RAG 适合高校课程场景，因为课程资料通常具有明确来源、章节结构和教学目标。系统可以把课程 Markdown、PPT、实验文档、题库等资料切分成可检索片段，并在回答、资源生成和测评反馈中显示引用来源。

## Chapter 2 文档切分 Chunking

Chunking 是把长文档切分为适合检索和生成的片段。切分粒度过大会降低匹配精度，切分粒度过小会导致上下文不完整。高校课程资料通常适合按以下层级切分：

- course：课程级元数据
- chapter：章节
- section：小节
- heading_path：标题路径
- chunk_text：正文片段
- source_file：原始 Markdown 文件

推荐保留章节标题路径，因为前端需要展示 grounding 信息，例如“来源章节：RAG 检索增强生成 / Chapter 3 / Retriever 检索策略”。

## Chapter 3 Embedding 与向量数据库

Embedding 模型将文本映射到向量空间，使语义相近的文本具有更高相似度。课程知识库可以为每个 chunk 生成向量，并存入 FAISS、Chroma、Milvus 或其他向量数据库中。

典型流程：

```python
from sentence_transformers import SentenceTransformer

model = SentenceTransformer("BAAI/bge-small-zh-v1.5")
texts = [chunk["text"] for chunk in chunks]
embeddings = model.encode(texts, normalize_embeddings=True)
```

## Chapter 4 Retriever 检索策略

Retriever 根据学生问题检索相关课程片段。常见策略包括：

- Top-K 语义检索
- 关键词 + 向量混合检索
- 按课程或章节过滤检索
- MMR 多样性重排
- 基于学生画像的个性化检索

在 EduAgentX 中，RetrieverAgent 不只返回文本，还应返回 course_id、course_name、source_file、chapter、section、score 等元数据。

## Chapter 5 Generator 与 Grounded Prompt

Generator 应基于检索结果生成回答或学习资源。Prompt 中需要明确要求：

1. 优先使用检索到的课程资料。
2. 不确定时说明不确定。
3. 生成资源时给出引用章节。
4. 针对学生画像调整难度和表达方式。

示例 Prompt 模板：

```text
你是高校 AI 课程助教。请基于给定课程片段回答问题。
如果课程片段无法支持结论，请明确说明。
学生画像：{profile}
课程片段：{retrieved_chunks}
学生问题：{question}
```

## Chapter 6 防幻觉与 JudgeAgent

JudgeAgent 可以对生成内容做质量审查，重点检查：

- 是否引用课程知识库
- 是否存在无来源断言
- 是否符合学生当前基础
- 是否覆盖目标知识点
- 是否存在危险或误导性内容

输出可以包括 judge_score、grounded、feedback 和 revision_suggestions。

## Chapter 7 RAG 系统评测

RAG 评测可以分为检索评测和生成评测：

- 检索召回率：是否找到了正确章节
- 引用准确性：引用片段是否支持回答
- 答案相关性：是否解决学生问题
- 个性化适配度：是否符合学生画像
- 测评闭环：学生小测结果是否能反向更新 mastery_map

## Chapter 8 项目实战：课程知识库问答 Agent

项目目标：构建一个面向高校 AI 课程群的知识库问答 Agent。

交付物：

1. 课程 Markdown 知识库
2. Chunking 与索引脚本
3. RetrieverAgent
4. TutorAgent
5. JudgeAgent
6. 前端 grounding 展示
7. 学生测评与学习路径重规划

评价标准：

- 能否基于课程资料回答问题
- 是否显示来源章节
- 是否支持不同学生画像
- 是否能生成讲解、测验、代码实验和项目任务
- 是否能根据测评结果更新学习路径
