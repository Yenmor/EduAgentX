import { MetricBar } from "@/components/MetricBar";

export function ScorePanel({ score, metrics }: { score: number; metrics: { label: string; value: number }[] }) {
  return (
    <section className="grid gap-6 border border-[var(--line)] bg-white p-6 md:grid-cols-[260px_1fr]">
      <div className="border-b border-[var(--line)] pb-5 md:border-b-0 md:border-r md:pb-0 md:pr-6">
        <div className="font-mono text-xs uppercase tracking-[0.16em] text-[var(--muted)]">总评分</div>
        <div className="mt-4 text-8xl font-extralight leading-none text-[var(--accent)]">{score}</div>
        <p className="mt-4 text-sm leading-6 text-[var(--muted)]">评分由完成度、测验表现、连续性、复习质量和应用能力共同计算。</p>
      </div>
      <div>
        {metrics.map((metric) => (
          <MetricBar key={metric.label} label={metric.label} value={metric.value} />
        ))}
      </div>
    </section>
  );
}
