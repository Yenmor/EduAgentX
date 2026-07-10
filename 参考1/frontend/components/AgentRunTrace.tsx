import type { AgentRunStep } from "@/lib/learning-data";

export function AgentRunTrace({ steps }: { steps: AgentRunStep[] }) {
  return (
    <section className="border border-[var(--line)] bg-white p-5">
      <div className="flex flex-wrap items-center justify-between gap-3 border-b border-[var(--line)] pb-4">
        <h2 className="text-2xl font-light">智能体协作证据</h2>
        <span className="w-full break-all font-mono text-xs text-[var(--muted)] sm:w-auto">GET /api/agent-runs/:runId</span>
      </div>
      <div className="mt-5 overflow-x-auto">
        <table className="w-full min-w-[860px] border-collapse text-left text-sm">
          <thead>
            <tr className="border-b border-[var(--line)] font-mono text-xs text-[var(--muted)]">
              <th className="py-3 pr-4 font-medium">角色</th>
              <th className="py-3 pr-4 font-medium">输入摘要</th>
              <th className="py-3 pr-4 font-medium">输出摘要</th>
              <th className="py-3 pr-4 font-medium">耗时</th>
              <th className="py-3 font-medium">状态</th>
            </tr>
          </thead>
          <tbody>
            {steps.map((step) => (
              <tr key={step.agent} className="border-b border-[var(--line)] align-top">
                <td className="py-4 pr-4 font-medium">{step.agent}</td>
                <td className="py-4 pr-4 text-[var(--muted)]">{step.input}</td>
                <td className="py-4 pr-4 text-[var(--muted)]">{step.output}</td>
                <td className="py-4 pr-4 font-mono text-xs">{step.duration}</td>
                <td className="py-4">{step.status}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </section>
  );
}
