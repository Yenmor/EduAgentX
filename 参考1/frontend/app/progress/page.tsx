import { PageHeader } from "@/components/PageHeader";
import { ProfileSummary } from "@/components/ProfileSummary";
import { ScorePanel } from "@/components/ScorePanel";
import { progressMetrics, profileDimensions } from "@/lib/learning-data";

export default function ProgressPage() {
  return (
    <div className="space-y-8">
      <PageHeader
        eyebrow="Progress"
        title="学习评分不是排名，是下一步导航"
        description="评分聚合完成度、测验表现、学习连续性、复习质量和应用能力，帮助学生知道该补哪里、先做什么、用什么资源。"
        aside={
          <div>
            <div className="font-mono text-xs text-[var(--muted)]">POST /api/progress/score</div>
            <div className="mt-2 text-2xl font-light">动态更新</div>
          </div>
        }
      />

      <ScorePanel score={74} metrics={progressMetrics} />

      <div className="grid gap-6 lg:grid-cols-[1fr_360px]">
        <section className="border border-[var(--line)] bg-white p-5">
          <h2 className="text-2xl font-light">学习诊断</h2>
          <div className="mt-5 grid gap-px bg-[var(--line)] md:grid-cols-2">
            {[
              ["薄弱知识点", "模型评估、过拟合、搜索策略"],
              ["最近测验表现", "诊断测验 78，章节测验 84，错题重练 91"],
              ["错题分布", "概念辨析 42%，应用推理 35%，轻量计算 23%"],
              ["资源使用情况", "讲义 4 份，题库 2 组，代码案例 1 个，交互模拟 1 个"]
            ].map(([label, value]) => (
              <div key={label} className="bg-white p-4">
                <div className="font-mono text-xs text-[var(--muted)]">{label}</div>
                <p className="mt-2 text-sm leading-6">{value}</p>
              </div>
            ))}
          </div>
        </section>
        <section className="border border-[var(--line)] bg-white p-5">
          <h2 className="text-2xl font-light">下一步建议</h2>
          <ol className="mt-5 space-y-4 text-sm leading-6 text-[var(--muted)]">
            <li>1. 先完成模型评估章节测验，目标 85 分以上。</li>
            <li>2. 用线性模型代码案例补应用能力。</li>
            <li>3. 两天后进行错题重练，观察复习质量是否提升。</li>
          </ol>
        </section>
      </div>

      <ProfileSummary dimensions={profileDimensions} />
    </div>
  );
}
