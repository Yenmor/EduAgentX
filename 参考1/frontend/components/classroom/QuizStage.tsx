"use client";

import { useState } from "react";
import type { QuizQuestion, QuizResult } from "@/lib/classroom-types";
import { submitQuizAnswers } from "@/lib/classroom-api";

type QuizStageProps = {
  sessionId: string;
  questions: QuizQuestion[];
};

export function QuizStage({ sessionId, questions }: QuizStageProps) {
  const [answers, setAnswers] = useState<Record<string, number>>({});
  const [submitted, setSubmitted] = useState(false);
  const [result, setResult] = useState<QuizResult | null>(null);
  const [loading, setLoading] = useState(false);

  async function handleSubmit() {
    setLoading(true);
    const quizResult = await submitQuizAnswers(sessionId, answers);
    setResult(quizResult);
    setSubmitted(true);
    setLoading(false);
  }

  function resetQuiz() {
    setSubmitted(false);
    setResult(null);
    setAnswers({});
  }

  const canSubmit = questions.every((question) => answers[question.id] !== undefined);

  return (
    <div className="rounded-[28px] border border-white/10 bg-white/[0.055] p-5">
      <div className="flex flex-wrap items-start justify-between gap-3">
        <div>
          <div className="text-xs uppercase tracking-[0.24em] text-emerald-300">Quiz</div>
          <h2 className="mt-2 text-2xl font-semibold text-slate-50">RAG 核心概念诊断测验</h2>
        </div>
        {result ? <div className="rounded-full border border-emerald-300/30 bg-emerald-300/10 px-4 py-2 text-sm text-emerald-100">得分 {result.score} / {questions.length}</div> : null}
      </div>
      <div className="mt-5 space-y-4">
        {questions.map((item, questionIndex) => {
          const selected = answers[item.id];
          return (
            <article key={item.id} className="rounded-2xl border border-white/10 bg-slate-950/45 p-4">
              <h3 className="font-semibold text-slate-100">{questionIndex + 1}. {item.question}</h3>
              <div className="mt-3 grid gap-2 md:grid-cols-2">
                {item.options.map((option, optionIndex) => {
                  const isCorrect = submitted && optionIndex === item.answerIndex;
                  const isWrong = submitted && selected === optionIndex && selected !== item.answerIndex;
                  return (
                    <button
                      key={option}
                      type="button"
                      disabled={submitted}
                      onClick={() => setAnswers((current) => ({ ...current, [item.id]: optionIndex }))}
                      className={[
                        "rounded-xl border px-3 py-3 text-left text-sm transition disabled:cursor-not-allowed",
                        selected === optionIndex ? "border-sky-300/60 bg-sky-300/10 text-sky-50" : "border-white/10 bg-white/[0.04] text-slate-300 hover:bg-white/[0.08]",
                        isCorrect ? "border-emerald-300/70 bg-emerald-300/15 text-emerald-50" : "",
                        isWrong ? "border-rose-300/70 bg-rose-300/15 text-rose-50" : ""
                      ].join(" ")}
                    >
                      {option}
                    </button>
                  );
                })}
              </div>
              {submitted ? (
                <div className="mt-3 rounded-xl border border-white/10 bg-white/[0.04] p-3 text-sm leading-6 text-slate-300">
                  <span className="text-slate-100">解析：</span>{item.explanation}
                  <div className="mt-2 text-emerald-200">{item.masteryImpact}</div>
                </div>
              ) : null}
            </article>
          );
        })}
      </div>
      {result ? (
        <div className="mt-5 grid gap-3 md:grid-cols-3">
          <div className="rounded-2xl border border-white/10 bg-slate-950/45 p-4">
            <div className="text-xs text-slate-400">薄弱知识点</div>
            <div className="mt-2 text-sm text-slate-100">{result.weakConcepts.length ? result.weakConcepts.join("、") : "暂无明显薄弱点"}</div>
          </div>
          <div className="rounded-2xl border border-white/10 bg-slate-950/45 p-4">
            <div className="text-xs text-slate-400">掌握度更新</div>
            <div className="mt-2 space-y-1 text-xs text-slate-300">
              {Object.entries(result.updatedMasteryMap).slice(0, 4).map(([key, value]) => <div key={key}>{key}: {value}%</div>)}
            </div>
          </div>
          <div className="rounded-2xl border border-white/10 bg-slate-950/45 p-4">
            <div className="text-xs text-slate-400">重规划建议</div>
            <div className="mt-2 space-y-1 text-xs leading-5 text-slate-300">
              {result.replanningSuggestions.map((item) => <div key={item}>{item}</div>)}
            </div>
          </div>
        </div>
      ) : null}
      <div className="mt-5 flex gap-3">
        <button type="button" onClick={handleSubmit} disabled={!canSubmit || submitted || loading} className="rounded-full bg-gradient-to-r from-emerald-400 to-sky-400 px-5 py-2 text-sm font-semibold text-slate-950 disabled:cursor-not-allowed disabled:opacity-50">
          {loading ? "提交中..." : "提交测验"}
        </button>
        {submitted ? (
          <button type="button" onClick={resetQuiz} className="rounded-full border border-white/10 px-5 py-2 text-sm text-slate-100 hover:bg-white/[0.06]">
            重新作答
          </button>
        ) : null}
      </div>
    </div>
  );
}
