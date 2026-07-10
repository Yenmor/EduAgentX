import type { ProfileDimension } from "@/lib/learning-data";

export function ProfileSummary({ dimensions }: { dimensions: ProfileDimension[] }) {
  return (
    <section className="border border-[var(--line)] bg-white p-5">
      <div className="border-b border-[var(--line)] pb-4">
        <h2 className="text-2xl font-light">动态学习画像</h2>
        <p className="mt-2 text-sm leading-6 text-[var(--muted)]">由自然语言需求、测验结果、学习行为和资源偏好综合更新。</p>
      </div>
      <div className="mt-5 grid gap-px bg-[var(--line)] md:grid-cols-3">
        {dimensions.map((dimension) => (
          <div key={dimension.label} className="bg-white p-4">
            <div className="font-mono text-xs text-[var(--muted)]">{dimension.label}</div>
            <div className="mt-2 text-sm leading-6">{dimension.value}</div>
          </div>
        ))}
      </div>
    </section>
  );
}
