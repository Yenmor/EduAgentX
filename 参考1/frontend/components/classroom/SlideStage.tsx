"use client";

import { useState } from "react";
import type { Slide } from "@/lib/classroom-types";

type SlideStageProps = {
  slides: Slide[];
};

export function SlideStage({ slides }: SlideStageProps) {
  const [index, setIndex] = useState(0);
  const slide = slides[index];

  return (
    <div className="rounded-[28px] border border-white/10 bg-[radial-gradient(circle_at_top_left,rgba(56,189,248,0.14),transparent_34%),rgba(255,255,255,0.055)] p-6 shadow-[0_24px_80px_rgba(0,0,0,0.35)]">
      <div className="flex items-center justify-between gap-3">
        <div className="text-xs uppercase tracking-[0.24em] text-sky-300">Slides</div>
        <div className="rounded-full border border-white/10 bg-white/[0.05] px-3 py-1 font-mono text-xs text-slate-300">
          {index + 1} / {slides.length}
        </div>
      </div>
      <div className="mt-8 min-h-[430px] rounded-[24px] border border-white/10 bg-slate-950/50 p-8">
        <h2 className="max-w-3xl text-4xl font-semibold leading-tight text-slate-50">{slide.title}</h2>
        {slide.subtitle ? <p className="mt-3 max-w-2xl text-lg text-slate-300">{slide.subtitle}</p> : null}
        <div className="mt-8 grid gap-4 lg:grid-cols-[1fr_0.75fr]">
          <ul className="space-y-4">
            {slide.bullets.map((bullet) => (
              <li key={bullet} className="flex gap-3 rounded-2xl border border-white/10 bg-white/[0.045] p-4 text-slate-200">
                <span className="mt-1 h-2 w-2 shrink-0 rounded-full bg-sky-300 shadow-[0_0_14px_rgba(56,189,248,0.7)]" />
                <span>{bullet}</span>
              </li>
            ))}
          </ul>
          <div className="space-y-4">
            {slide.personalizedReason ? (
              <div className="rounded-2xl border border-emerald-300/20 bg-emerald-300/10 p-4">
                <div className="text-xs text-emerald-200">个性化原因</div>
                <p className="mt-2 text-sm leading-6 text-slate-200">{slide.personalizedReason}</p>
              </div>
            ) : null}
            {slide.judgeInfo ? (
              <div className="rounded-2xl border border-sky-300/20 bg-sky-300/10 p-4">
                <div className="flex items-center justify-between gap-3 text-xs text-sky-100">
                  <span>Judge {slide.judgeInfo.score}/100</span>
                  <span>{slide.judgeInfo.grounded ? "Grounded 通过" : "待补引用"}</span>
                </div>
                <p className="mt-2 text-sm leading-6 text-slate-200">{slide.judgeInfo.feedback}</p>
              </div>
            ) : null}
            {slide.sourceChunks?.length ? (
              <div className="rounded-2xl border border-white/10 bg-white/[0.04] p-4">
                <div className="text-xs text-slate-300">引用课程片段</div>
                <div className="mt-3 space-y-2">
                  {slide.sourceChunks.map((chunk) => (
                    <div key={chunk.id} className="rounded-xl border border-white/10 bg-slate-950/50 p-3">
                      <div className="font-mono text-[11px] text-sky-200">{chunk.sourceFile}</div>
                      <p className="mt-1 text-xs leading-5 text-slate-300">{chunk.excerpt}</p>
                    </div>
                  ))}
                </div>
              </div>
            ) : null}
            {slide.example ? (
              <div className="rounded-2xl border border-violet-300/20 bg-violet-300/10 p-4">
                <div className="text-xs text-violet-200">课堂示例</div>
                <p className="mt-2 text-sm leading-6 text-slate-200">{slide.example}</p>
              </div>
            ) : null}
          </div>
        </div>
      </div>
      <div className="mt-5 flex justify-between">
        <button type="button" onClick={() => setIndex((value) => Math.max(0, value - 1))} className="rounded-full border border-white/10 px-4 py-2 text-sm text-slate-200 disabled:opacity-40" disabled={index === 0}>
          上一页
        </button>
        <button type="button" onClick={() => setIndex((value) => Math.min(slides.length - 1, value + 1))} className="rounded-full bg-gradient-to-r from-sky-400 to-violet-500 px-5 py-2 text-sm font-semibold text-white shadow-[0_14px_36px_rgba(99,102,241,0.34)] disabled:opacity-40" disabled={index === slides.length - 1}>
          下一页
        </button>
      </div>
    </div>
  );
}
