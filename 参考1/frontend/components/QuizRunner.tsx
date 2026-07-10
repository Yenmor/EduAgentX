"use client";

import { useMemo, useState } from "react";
import type { QuizQuestion } from "@/lib/learning-data";

export function QuizRunner({ questions }: { questions: QuizQuestion[] }) {
  const [answers, setAnswers] = useState<Record<string, number>>({});
  const [submitted, setSubmitted] = useState(false);

  const result = useMemo(() => {
    const correct = questions.filter((question) => answers[question.id] === question.answer).length;
    const score = Math.round((correct / questions.length) * 100);
    const weakPoints = questions.filter((question) => answers[question.id] !== question.answer).map((question) => question.weakPoint);
    return { correct, score, weakPoints: Array.from(new Set(weakPoints)) };
  }, [answers, questions]);

  return (
    <section className="border border-[var(--line)] bg-white p-5">
      <div className="flex flex-wrap items-center justify-between gap-3 border-b border-[var(--line)] pb-4">
        <h2 className="text-2xl font-light">测验作答</h2>
        <span className="font-mono text-xs text-[var(--muted)]">POST /api/quiz/submit</span>
      </div>
      <div className="mt-5 space-y-5">
        {questions.map((question, questionIndex) => (
          <div key={question.id} className="border border-[var(--line)] p-4">
            <div className="font-mono text-xs text-[var(--muted)]">Q{questionIndex + 1}</div>
            <h3 className="mt-2 text-lg font-medium leading-7">{question.stem}</h3>
            <div className="mt-4 grid gap-2">
              {question.options.map((option, optionIndex) => {
                const selected = answers[question.id] === optionIndex;
                const correct = submitted && optionIndex === question.answer;
                const wrong = submitted && selected && optionIndex !== question.answer;
                return (
                  <button
                    key={option}
                    onClick={() => setAnswers((current) => ({ ...current, [question.id]: optionIndex }))}
                    className={[
                      "border px-3 py-2 text-left text-sm",
                      correct ? "border-[var(--accent)] bg-[var(--accent)] text-white" : wrong ? "border-black bg-black text-white" : selected ? "border-[var(--accent)]" : "border-[var(--line)]"
                    ].join(" ")}
                  >
                    {option}
                  </button>
                );
              })}
            </div>
            {submitted ? (
              <div className="mt-4 border-l-2 border-[var(--accent)] pl-3 text-sm leading-6 text-[var(--muted)]">
                <strong className="text-[var(--ink)]">错题分析：</strong>
                {question.analysis}
              </div>
            ) : null}
          </div>
        ))}
      </div>
      <div className="mt-5 flex flex-wrap items-center gap-3">
        <button
          onClick={() => setSubmitted(true)}
          className="bg-[var(--accent)] px-5 py-2.5 text-sm font-semibold text-white disabled:opacity-40"
          disabled={Object.keys(answers).length < questions.length}
        >
          提交测验
        </button>
        {submitted ? (
          <div className="text-sm text-[var(--muted)]">
            得分 <span className="text-xl text-[var(--accent)]">{result.score}</span>，薄弱点：
            {result.weakPoints.length ? result.weakPoints.join(" / ") : "暂无明显薄弱点"}
          </div>
        ) : null}
      </div>
    </section>
  );
}
