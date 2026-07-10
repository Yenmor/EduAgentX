# 高校 AI 课程群知识库

EduAgentX 当前定位为：面向高校人工智能专业课程群的个性化资源生成与多智能体学习系统。知识库围绕 AI 专业学习路径组织，不是泛用聊天资料库。

## 已接入课程

本轮已将 `knowledge_eduagentx_course_catalog.zip` 解压到 `backend/data`，包含 16 个 Markdown 课程文件：Git 与工程协作、Python 程序设计、NumPy 与 Pandas 数据处理、机器学习、神经网络与深度学习、PyTorch 深度学习实践、计算机视觉基础、计算机视觉与 CNN、Transformer 架构基础、大语言模型基础、RAG 检索增强生成、RAG 与 LangChain 应用开发、AI Agent 与多智能体系统、强化学习、RLHF 人类反馈强化学习、图神经网络。

比赛演示主线聚焦：RAG 检索增强生成、RAG 与 LangChain 应用开发、AI Agent 与多智能体系统。

## Markdown 知识库结构

```text
backend/data/
  course_catalog/catalog.json
  courses/*.md
  metadata/processing_report.json
  README.md
```

课程 Markdown 保留 YAML frontmatter，后续 ingestion、chunking 和索引构建应继续保留来源文件与标题层级信息。

## catalog.json 字段

每门课程至少包含 `id`、`name`、`file`、`level`、`description`、`concepts`、`prerequisites`、`nextCourses`、`resourceTypes`。

后端轻量服务位于 `backend/services/course_catalog_service.py`，提供 `load_course_catalog()`、`get_all_courses()`、`get_course_by_id()`、`get_courses_by_level()`、`get_course_source_file()`、`get_course_prerequisites()`、`get_course_next_courses()`。

## RAG Chunk 元数据

后续 RAG manifest / chunks 至少预留：`chunk_id`、`course_id`、`course_name`、`source_file`、`chapter`、`section`、`heading_path`、`content`、`char_count`。

本轮已在 `backend/rag/chunker.py` 中补充 `course_id`、`chapter`、`section`、`char_count` 字段，便于前端展示 grounding 和 Judge 结果。

## RetrieverAgent 与课程目录

课堂生成时，`CourseCatalogService` 会先按 `selected_course_ids` 读取课程目录，再将课程文件、课程名称、先修课程和 concepts 传入 Orchestrator。`RetrieverAgent` 优先使用已有 RAG index；如果本地索引不可用，会从 `backend/data/courses/*.md` 生成 fallback `SourceChunk`，保证演示链路不断。

## source_chunks 进入资源卡片

`ResourceAgentGroup` 生成每个学习资源时，会绑定 1-3 个 `SourceChunk`。`JudgeAgent` 根据这些片段判断资源是否 grounded，并写回 `judge_score`、`judge_feedback`、`grounded`。后端 classroom API 会把这些字段适配成前端 `ResourceSceneCard` 可展示的数据结构。

## 如何支撑资源生成

课程目录用于确定学习目标与课程范围；Markdown chunk 用于 RAG grounding；Agent trace 用于展示多智能体协作过程；Judge 信息用于说明资源是否基于课程片段、是否适合当前学生画像。

资源生成链路：ProfileAgent 生成学习画像，RetrieverAgent 检索课程 chunk，PlannerAgent 规划学习路径，ResourceAgent 生成讲解、测验、代码实验、项目等资源，JudgeAgent 检查 grounding、质量和个性化适配，EvaluationAgent 根据测验更新 mastery_map 并触发重规划。
