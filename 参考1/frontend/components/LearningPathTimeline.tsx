import type { LearningStep } from "@/lib/types";

type LearningPathTimelineProps = {
  steps: LearningStep[];
};

export function LearningPathTimeline({ steps }: LearningPathTimelineProps) {
  return (
    <div className="space-y-3">
      {steps.map((step) => (
        <div key={step.order} className="flex gap-4 rounded-lg border border-circuit-line bg-circuit-panel p-4 shadow-glow">
          <div className="flex h-10 w-10 shrink-0 items-center justify-center rounded-lg border border-circuit-blue/50 font-mono text-circuit-blue">
            {step.order}
          </div>
          <div className="min-w-0">
            <div className="font-semibold text-circuit-text">{step.title}</div>
            <div className="mt-1 text-sm text-circuit-muted">{step.activity}</div>
            <div className="mt-3 flex flex-wrap gap-2 font-mono text-xs text-circuit-muted">
              <span className="rounded border border-circuit-line px-2 py-1">{step.knowledge_point}</span>
              <span className="rounded border border-circuit-line px-2 py-1">{step.estimated_minutes} min</span>
              <span className="rounded border border-circuit-green/40 px-2 py-1 text-circuit-green">{Math.round(step.mastery_target * 100)}%</span>
            </div>
          </div>
        </div>
      ))}
    </div>
  );
}
