"use client";

import { KeyboardEvent } from "react";
import type { CourseMeta } from "@/lib/classroom-types";

type HeroGeneratePanelProps = {
  value: string;
  onChange: (value: string) => void;
  onGenerate: () => void;
  examples: string[];
  loading?: boolean;
  coursesByLevel: Record<string, CourseMeta[]>;
  selectedCourseIds: string[];
  onToggleCourse: (courseId: string) => void;
};

export function HeroGeneratePanel({
  value,
  onChange,
  onGenerate,
  examples,
  loading = false,
  coursesByLevel,
  selectedCourseIds,
  onToggleCourse
}: HeroGeneratePanelProps) {
  const disabled = loading || !value.trim();

  function handleKeyDown(event: KeyboardEvent<HTMLTextAreaElement>) {
    if (event.key === "Enter" && (event.metaKey || event.ctrlKey) && !disabled) {
      event.preventDefault();
      onGenerate();
    }
  }

  return (
    <div className="mx-auto max-w-5xl rounded-[28px] border border-white/10 bg-white/[0.07] p-4 shadow-[0_30px_120px_rgba(76,29,149,0.32)] backdrop-blur-2xl">
      <textarea
        value={value}
        onChange={(event) => onChange(event.target.value)}
        onKeyDown={handleKeyDown}
        rows={5}
        placeholder="例如：我学过深度学习，但不了解 RAG 和 LangChain，想完成一个课程知识库问答项目"
        className="min-h-40 w-full resize-none rounded-[20px] border border-white/10 bg-slate-950/60 p-5 text-lg leading-8 text-slate-50 outline-none placeholder:text-slate-500 focus:border-sky-300/50"
      />

      <div className="mt-4 grid gap-3 md:grid-cols-2">
        {Object.entries(coursesByLevel).map(([level, courses]) => (
          <section key={level} className="rounded-2xl border border-white/10 bg-slate-950/35 p-4">
            <h3 className="text-sm font-semibold text-slate-100">{level}</h3>
            <div className="mt-3 flex flex-wrap gap-2">
              {courses.map((course) => {
                const selected = selectedCourseIds.includes(course.id);
                return (
                  <button
                    key={course.id}
                    type="button"
                    onClick={() => onToggleCourse(course.id)}
                    className={[
                      "rounded-full border px-3 py-1.5 text-xs transition",
                      selected
                        ? "border-sky-300/60 bg-sky-300/15 text-sky-50"
                        : "border-white/10 bg-white/[0.04] text-slate-300 hover:border-sky-300/30 hover:text-slate-50"
                    ].join(" ")}
                  >
                    {course.name}
                  </button>
                );
              })}
            </div>
          </section>
        ))}
      </div>

      <div className="mt-4 flex flex-col gap-3 lg:flex-row lg:items-center lg:justify-between">
        <div className="flex flex-wrap gap-2">
          {examples.map((example) => (
            <button key={example} type="button" onClick={() => onChange(example)} className="rounded-full border border-white/10 bg-white/[0.06] px-3 py-1.5 text-xs text-slate-300 transition hover:border-sky-300/40 hover:text-slate-50">
              {example}
            </button>
          ))}
        </div>
        <button
          type="button"
          onClick={onGenerate}
          disabled={disabled}
          className="shrink-0 rounded-full bg-gradient-to-r from-sky-400 via-indigo-400 to-violet-500 px-6 py-3 text-sm font-semibold text-white shadow-[0_18px_42px_rgba(99,102,241,0.38)] transition hover:scale-[1.02] disabled:cursor-not-allowed disabled:opacity-50"
        >
          {loading ? "多智能体正在生成课堂..." : "生成专属高校 AI 课堂"}
        </button>
      </div>
    </div>
  );
}
