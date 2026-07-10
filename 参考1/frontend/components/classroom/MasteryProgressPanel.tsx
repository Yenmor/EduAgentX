type MasteryProgressPanelProps = {
  mastery: Record<string, number>;
};

export function MasteryProgressPanel({ mastery }: MasteryProgressPanelProps) {
  return (
    <div className="rounded-2xl border border-white/10 bg-white/[0.06] p-4">
      <div className="text-sm font-semibold text-slate-100">知识掌握度</div>
      <div className="mt-4 space-y-3">
        {Object.entries(mastery).map(([name, value]) => (
          <div key={name}>
            <div className="flex justify-between text-xs">
              <span className="text-slate-300">{name}</span>
              <span className="font-mono text-slate-400">{value}%</span>
            </div>
            <div className="mt-2 h-2 overflow-hidden rounded-full bg-slate-950/70">
              <div className="h-full rounded-full bg-gradient-to-r from-sky-400 via-indigo-400 to-violet-400" style={{ width: `${value}%` }} />
            </div>
          </div>
        ))}
      </div>
    </div>
  );
}
