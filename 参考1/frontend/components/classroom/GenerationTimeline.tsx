import type { AgentTraceStep } from "@/lib/classroom-types";

type GenerationTimelineProps = {
  steps: AgentTraceStep[];
  compact?: boolean;
};

export function GenerationTimeline({ steps, compact = false }: GenerationTimelineProps) {
  return (
    <div className={compact ? "space-y-3" : "grid gap-4 lg:grid-cols-7"}>
      {steps.map((step, index) => (
        <div key={step.agentId} className="relative rounded-2xl border border-white/10 bg-white/[0.055] p-4">
          {!compact && index < steps.length - 1 ? <div className="absolute left-[calc(100%-4px)] top-8 hidden h-px w-4 bg-white/20 lg:block" /> : null}
          <div className="flex items-center gap-3">
            <div
              className={[
                "flex h-8 w-8 shrink-0 items-center justify-center rounded-full border text-xs font-semibold",
                step.status === "done" ? "border-emerald-300/60 bg-emerald-300/15 text-emerald-200" : "",
                step.status === "running" ? "border-sky-300/70 bg-sky-300/15 text-sky-100 shadow-[0_0_24px_rgba(56,189,248,0.24)] animate-pulse" : "",
                step.status === "pending" ? "border-white/10 bg-white/[0.04] text-slate-400" : "",
                step.status === "error" ? "border-rose-300/70 bg-rose-300/15 text-rose-100" : ""
              ].join(" ")}
            >
              {step.status === "done" ? "✓" : index + 1}
            </div>
            <div className="min-w-0">
              <div className="text-sm font-semibold text-slate-100">{step.action}</div>
              <div className="truncate text-[11px] text-slate-400">{step.agentName}</div>
            </div>
          </div>
          <p className="mt-3 text-xs leading-5 text-slate-300">{step.outputSummary || step.role}</p>
          {step.durationMs ? <div className="mt-3 font-mono text-[11px] text-sky-200/80">{(step.durationMs / 1000).toFixed(1)}s</div> : null}
        </div>
      ))}
    </div>
  );
}
