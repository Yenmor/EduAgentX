# EduAgentX 高校人工智能课程群知识库

本压缩包由原始 `knowledge.zip` 整理而来，适合作为 EduAgentX 的高校 AI 课程知识库底座。

## 目录结构

```text
course_catalog/catalog.json   # 课程目录与元数据
courses/*.md                  # 已轻量规范化的课程 Markdown
metadata/processing_report.json
```

## 覆盖课程

基础工具层：Git 与工程协作、Python 程序设计、NumPy 与 Pandas 数据处理  
核心课程层：机器学习、神经网络与深度学习、PyTorch 深度学习实践  
高级专题层：Transformer、大语言模型、强化学习、RLHF、图神经网络  
应用实践层：RAG 检索增强生成、RAG 与 LangChain 应用开发、AI Agent 与多智能体系统

## 处理说明

- 为每个 Markdown 增加课程级 front matter。
- 为每门课补充课程定位、先修课程、后续课程、核心知识点和适配资源类型。
- 将原始 H1 降级，确保每个文件只有一个课程级 H1。
- 删除少量明显占位标题或异常短行，例如“这是一个二级标题”“New Feature”“G”。
- 新增独立的 `rag-retrieval-augmented-generation.md`，方便比赛主线演示 RAG + Agent。

## 接入建议

后端 RAG 构建索引时，建议每个 chunk 保留：

```text
course_id, course_name, source_file, level, chapter, section, heading_path, chunk_index, text, token_count
```

前端可直接使用 `course_catalog/catalog.json` 驱动课程选择区、课程归属展示、资源卡片来源展示和比赛 demo 页面。
