import type { AgentTraceStep } from "@/lib/classroom-types";

type AgentRoleCardProps = {
  agent: AgentTraceStep;
};

const statusText: Record<AgentTraceStep["status"], string> = {
  pending: "等待中",
  running: "运行中",
  done: "已完成",
  error: "异常"
};

const statusClass: Record<AgentTraceStep["status"], string> = {
  pending: "bg-slate-400",
  running: "bg-sky-300 shadow-[0_0_18px_rgba(56,189,248,0.8)] animate-pulse",
  done: "bg-emerald-300",
  error: "bg-rose-300"
};

export function AgentRoleCard({ agent }: AgentRoleCardProps) {
  return (
    <article className={[
      "group rounded-2xl border border-white/10 bg-white/[0.055] p-4 shadow-[0_16px_50px_rgba(0,0,0,0.24)] transition duration-300 hover:-translate-y-0.5 hover:border-sky-300/30 hover:bg-white/[0.08]",
      agent.status === "running" ? "shadow-[0_0_34px_rgba(56,189,248,0.18)]" : ""
    ].join(" ")}>
      <div className="flex items-start gap-3">
        <div className={`flex h-11 w-11 shrink-0 items-center justify-center rounded-2xl bg-gradient-to-br ${agent.accent || "from-sky-400 to-violet-500"} text-base font-bold text-white shadow-[0_0_28px_rgba(99,102,241,0.28)]`}>
          {agent.status === "done" ? "✓" : agent.agentName.slice(0, 1)}
        </div>
        <div className="min-w-0 flex-1">
          <div className="flex items-center justify-between gap-2">
            <h3 className="truncate text-sm font-semibold text-slate-50">{agent.agentName}</h3>
            <span className={`h-2.5 w-2.5 rounded-full ${statusClass[agent.status]}`} />
          </div>
          <div className="mt-1 flex items-center gap-2 text-[11px] text-slate-400">
            <span>{agent.role}</span>
            <span className="h-1 w-1 rounded-full bg-slate-500" />
            <span>{statusText[agent.status]}</span>
          </div>
          <p className="mt-3 text-xs leading-5 text-slate-300">{agent.action}</p>
          {agent.outputSummary ? <p className="mt-2 text-[11px] leading-5 text-sky-100/80">{agent.outputSummary}</p> : null}
        </div>
      </div>
    </article>
  );
}
