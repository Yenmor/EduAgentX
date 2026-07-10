"use client";

import { useState } from "react";
import type { CodeLabTask } from "@/lib/classroom-types";

type CodeLabStageProps = {
  task: CodeLabTask;
};

export function CodeLabStage({ task }: CodeLabStageProps) {
  const [copied, setCopied] = useState(false);
  const [copyError, setCopyError] = useState("");

  async function copyCode() {
    try {
      await navigator.clipboard.writeText(task.code);
      setCopied(true);
      setCopyError("");
      window.setTimeout(() => setCopied(false), 1400);
    } catch {
      setCopyError("复制失败，请手动选中代码。");
    }
  }

  return (
    <div className="rounded-[28px] border border-white/10 bg-white/[0.055] p-5">
      <div className="flex items-start justify-between gap-3">
        <div>
          <div className="text-xs uppercase tracking-[0.24em] text-amber-300">Code Lab</div>
          <h2 className="mt-2 text-2xl font-semibold text-slate-50">{task.title}</h2>
          <p className="mt-2 max-w-3xl text-sm leading-6 text-slate-300">{task.goal}</p>
        </div>
        <button type="button" onClick={copyCode} className="rounded-full border border-white/10 bg-white/[0.06] px-4 py-2 text-sm text-slate-100 transition hover:bg-white/[0.1]">
          {copied ? "已复制" : "复制代码"}
        </button>
      </div>
      {copyError ? <div className="mt-3 rounded-xl border border-rose-300/30 bg-rose-300/10 px-3 py-2 text-sm text-rose-100">{copyError}</div> : null}
      <div className="mt-5 grid gap-5 lg:grid-cols-[0.8fr_1.2fr]">
        <div className="space-y-4">
          <section className="rounded-2xl border border-white/10 bg-slate-950/45 p-4">
            <h3 className="text-sm font-semibold text-slate-100">先修要求</h3>
            <div className="mt-3 flex flex-wrap gap-2">
              {task.prerequisites.map((item) => (
                <span key={item} className="rounded-full border border-amber-300/20 bg-amber-300/10 px-3 py-1 text-xs text-amber-100">{item}</span>
              ))}
            </div>
          </section>
          <section className="rounded-2xl border border-white/10 bg-slate-950/45 p-4">
            <h3 className="text-sm font-semibold text-slate-100">实验步骤</h3>
            <ol className="mt-3 space-y-2">
              {task.steps.map((step, index) => (
                <li key={step} className="flex gap-3 text-sm leading-6 text-slate-300">
                  <span className="flex h-6 w-6 shrink-0 items-center justify-center rounded-full bg-sky-300/10 font-mono text-xs text-sky-200">{index + 1}</span>
                  {step}
                </li>
              ))}
            </ol>
          </section>
        </div>
        <div>
          <pre className="max-h-[460px] overflow-auto rounded-2xl border border-white/10 bg-[#050711] p-5 text-sm leading-6 text-slate-100 shadow-inner">
            <code>{task.code}</code>
          </pre>
          <div className="mt-4 rounded-2xl border border-sky-300/20 bg-sky-300/10 p-4 text-sm text-sky-100">{task.runHint}</div>
        </div>
      </div>
      <div className="mt-5 rounded-2xl border border-white/10 bg-white/[0.04] p-4">
        <h3 className="text-sm font-semibold text-slate-100">反思问题</h3>
        <div className="mt-3 grid gap-2 md:grid-cols-2">
          {task.reflection.map((item) => (
            <div key={item} className="rounded-xl border border-white/10 bg-slate-950/40 p-3 text-sm text-slate-300">{item}</div>
          ))}
        </div>
      </div>
    </div>
  );
}
