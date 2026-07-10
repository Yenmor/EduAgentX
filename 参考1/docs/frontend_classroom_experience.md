# 前端课堂体验

EduAgentX 前端围绕三个核心入口组织：`/studio`、`/classroom`、`/demo`。

## /studio

`/studio` 是高校 AI 课程群课堂生成入口。用户输入学习目标，选择目标课程，默认聚焦 RAG 检索增强生成、RAG 与 LangChain 应用开发、AI Agent 与多智能体系统。

生成按钮调用 `frontend/lib/classroom-api.ts` 中的 `startClassroomGeneration()`。真实后端不可用时，会自动 fallback 到 mock session，并跳转到 `/classroom?sessionId=xxx`。

## /classroom

`/classroom` 由统一 `ClassroomState` 驱动。页面顶部展示课程群、当前课程、关联课程、先修课程和知识库来源。左侧展示 AI 主讲教师、AI 助教、AI 学伴、规划师、资源设计师、评测官、质量审查员。中间舞台包含 Slides、Whiteboard、Mindmap、Quiz、Code Lab、Project。右侧通过 tabs 展示助手、资源、路径、画像和掌握度。

Stage 组件不再直接导入 mock 数据，而是由页面统一注入。Quiz 提交后会锁定答案，并显示 score、weakConcepts、updatedMasteryMap 和 replanningSuggestions。

## /demo

`/demo` 面向评委视角，突出 16 个 Markdown 课程文件覆盖、A3 赛题能力覆盖矩阵、课程选择到测评反馈的完整闭环，以及当前演示聚焦 RAG + LangChain + AI Agent。

## OpenMAIC 风格借鉴点

本轮借鉴的是沉浸式课堂交互范式：多角色协作、中央舞台、侧边学习支持、课堂生成过程可视化。项目没有复制 OpenMAIC 源码、Logo、视觉资产或文案。

## 比赛演示路线

推荐演示顺序：打开 `/demo` 介绍课程知识库与能力矩阵，进入 `/studio` 输入学习目标并选择课程，生成课堂 session，最后在 `/classroom` 展示课程归属、Agent 协作、RAG grounding、资源卡、Quiz 反馈与 mastery_map 更新。
