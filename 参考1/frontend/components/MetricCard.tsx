type MetricCardProps = {
  label: string;
  value: string;
  hint: string;
};

export function MetricCard({ label, value, hint }: MetricCardProps) {
  return (
    <div className="rounded-lg border border-circuit-line bg-circuit-panel p-5 shadow-glow">
      <div className="font-mono text-xs uppercase tracking-[0.16em] text-circuit-muted">{label}</div>
      <div className="mt-3 text-3xl font-semibold text-circuit-text">{value}</div>
      <div className="mt-2 text-sm text-circuit-muted">{hint}</div>
    </div>
  );
}
