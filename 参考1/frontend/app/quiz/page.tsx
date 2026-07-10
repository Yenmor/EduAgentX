"use client";

import { useState } from "react";
import { LearningRequestBox } from "@/components/LearningRequestBox";
import { PageHeader } from "@/components/PageHeader";
import { QuizCard } from "@/components/QuizCard";
import { QuizRunner } from "@/components/QuizRunner";
import { quizQuestions, quizTypes } from "@/lib/learning-data";

export default function QuizPage() {
  const [activeQuiz, setActiveQuiz] = useState(quizTypes[0].id);

  return (
    <div className="space-y-8">
      <PageHeader
        eyebrow="Quiz"
        title="用测验驱动下一步学习"
        description="测验页支持诊断、章节、综合、错题重练、自定义测验和代码实操题。结果会更新薄弱点，并进入学习评分计算。"
      />

      <LearningRequestBox
        title="自定义测验"
        placeholder="例如：给我生成 10 道人工智能导论中关于搜索算法和模型评估的题，难度中等，包含 2 道应用题。"
        buttonLabel="生成测验"
        defaultValue="围绕人工智能导论的模型评估、过拟合和搜索策略生成一组诊断题。"
      />

      <section>
        <div className="mb-4 flex items-end justify-between border-b border-[var(--line)] pb-3">
          <h2 className="text-3xl font-light">测验类型</h2>
          <span className="font-mono text-xs text-[var(--muted)]">POST /api/quiz/generate</span>
        </div>
        <div className="grid gap-4 md:grid-cols-2 xl:grid-cols-3">
          {quizTypes.map((quiz) => (
            <QuizCard key={quiz.id} quiz={quiz} active={activeQuiz === quiz.id} onSelect={() => setActiveQuiz(quiz.id)} />
          ))}
        </div>
      </section>

      <div className="grid gap-6 lg:grid-cols-[1fr_320px]">
        <QuizRunner questions={quizQuestions} />
        <aside className="border border-[var(--line)] bg-white p-5">
          <div className="font-mono text-xs uppercase tracking-[0.16em] text-[var(--muted)]">结果影响</div>
          <h2 className="mt-3 text-3xl font-light">评分联动</h2>
          <div className="mt-5 space-y-4 text-sm leading-6 text-[var(--muted)]">
            <p>错题会进入薄弱知识点和错题分布。</p>
            <p>章节测验影响测验表现；错题重练影响复习质量。</p>
            <p>代码实操题会影响应用能力，并触发后续项目型资源推荐。</p>
          </div>
        </aside>
      </div>
    </div>
  );
}
