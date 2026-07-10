import type { AgentTraceStep } from "@/lib/types";

type AgentTraceProps = {
  trace?: AgentTraceStep[];
};

export function AgentTrace({ trace = [] }: AgentTraceProps) {
  if (!trace.length) {
    return (
      <div className="rounded-lg border border-circuit-line bg-circuit-panel p-4 text-sm text-circuit-muted">
        Awaiting agent execution.
      </div>
    );
  }

  return (
    <div className="rounded-lg border border-circuit-line bg-circuit-panel p-4 shadow-glow">
      <div className="font-mono text-xs uppercase tracking-[0.18em] text-circuit-green">Agent Trace</div>
      <div className="mt-4 space-y-3">
        {trace.map((step, index) => (
          <div key={`${step.agent}-${step.action}-${index}`} className="rounded-lg border border-circuit-line bg-black/25 p-3">
            <div className="flex items-center justify-between gap-3">
              <div>
                <div className="font-mono text-sm text-circuit-text">{step.agent_name ?? step.agent}</div>
                <div className="mt-1 font-mono text-[11px] text-circuit-muted">
                  {step.model_provider ?? "local"}{step.fallback_used ? " fallback" : ""}{step.status ? ` - ${step.status}` : ""}
                </div>
              </div>
              <div className="rounded border border-circuit-blue/40 px-2 py-1 font-mono text-[11px] text-circuit-blue">{step.action}</div>
            </div>
            <pre className="mt-2 max-h-32 overflow-auto whitespace-pre-wrap text-xs text-circuit-muted">
              {JSON.stringify(step.output ?? {}, null, 2)}
            </pre>
          </div>
        ))}
      </div>
    </div>
  );
}
