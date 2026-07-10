export function MetricBar({ label, value }: { label: string; value: number }) {
  return (
    <div className="border-b border-[var(--line)] py-4 last:border-b-0">
      <div className="flex items-center justify-between gap-4">
        <span className="text-sm font-medium">{label}</span>
        <span className="font-mono text-sm text-[var(--accent)]">{value}/100</span>
      </div>
      <div className="mt-3 h-2 bg-[var(--soft)]">
        <div className="h-full bg-[var(--accent)]" style={{ width: `${value}%` }} />
      </div>
    </div>
  );
}
