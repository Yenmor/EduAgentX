export function GenerationProgress({ events }: { events: { label: string; percent: number; detail: string }[] }) {
  return (
    <section className="border border-[var(--line)] bg-white p-5">
      <div className="flex flex-wrap items-center justify-between gap-3 border-b border-[var(--line)] pb-4">
        <h2 className="text-2xl font-light">生成进度</h2>
        <span className="w-full break-all font-mono text-xs text-[var(--muted)] sm:w-auto">GET /api/agent-runs/:runId/events</span>
      </div>
      <div className="mt-5 space-y-4">
        {events.map((event) => (
          <div key={event.label}>
            <div className="flex items-center justify-between gap-4">
              <span className="text-sm font-medium">{event.label}</span>
              <span className="font-mono text-xs text-[var(--muted)]">{event.percent}%</span>
            </div>
            <div className="mt-2 h-2 bg-[var(--soft)]">
              <div className="h-full bg-[var(--accent)]" style={{ width: `${event.percent}%` }} />
            </div>
            <p className="mt-2 text-xs text-[var(--muted)]">{event.detail}</p>
          </div>
        ))}
      </div>
    </section>
  );
}
