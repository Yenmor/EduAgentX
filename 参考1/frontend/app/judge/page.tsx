import { AgentRunTrace } from "@/components/AgentRunTrace";
import { GenerationProgress } from "@/components/GenerationProgress";
import { PageHeader } from "@/components/PageHeader";
import { ProfileSummary } from "@/components/ProfileSummary";
import { SafetyBadge } from "@/components/SafetyBadge";
import { agentRunSteps, apiPlaceholders, generationEvents, profileDimensions } from "@/lib/learning-data";

export default function JudgePage() {
  return (
    <div className="space-y-8">
      <PageHeader
        eyebrow="Judge Mode"
        title="隐藏评审模式"
        description="此页不进入普通导航，只用于展示后台协作证据、生成过程、资源类型、画像维度、事实校验和安全过滤状态。"
        aside={
          <div>
            <div className="font-mono text-xs text-[var(--muted)]">Run ID</div>
            <div className="mt-2 text-2xl font-light">run_mock_20260523</div>
          </div>
        }
      />

      <div className="grid gap-6 lg:grid-cols-[1fr_360px]">
        <GenerationProgress events={generationEvents} />
        <section className="border border-[var(--line)] bg-white p-5">
          <h2 className="text-2xl font-light">安全与防幻觉</h2>
          <div className="mt-5 space-y-4">
            {[
              ["事实校验", "课程概念与题目答案已通过知识库引用检查。", "通过"],
              ["内容安全", "过滤无来源绝对化表述 1 条。", "已过滤"],
              ["答案复核", "思维导图资源仍在人工可读性复核。", "复核中"]
            ].map(([label, detail, status]) => (
              <div key={label} className="border border-[var(--line)] p-4">
                <div className="flex items-center justify-between gap-3">
                  <h3 className="text-lg font-light">{label}</h3>
                  <SafetyBadge status={status as "通过" | "复核中" | "已过滤"} />
                </div>
                <p className="mt-3 text-sm leading-6 text-[var(--muted)]">{detail}</p>
              </div>
            ))}
          </div>
        </section>
      </div>

      <AgentRunTrace steps={agentRunSteps} />

      <div className="grid gap-6 lg:grid-cols-2">
        <section className="border border-[var(--line)] bg-white p-5">
          <h2 className="text-2xl font-light">生成资源类型</h2>
          <div className="mt-5 grid grid-cols-2 gap-px bg-[var(--line)] md:grid-cols-3">
            {["讲义", "PPT/速览课件", "思维导图", "题库", "拓展阅读", "代码案例", "动画/HTML 交互模拟"].map((type) => (
              <div key={type} className="bg-white p-4 text-sm">
                {type}
              </div>
            ))}
          </div>
        </section>
        <section className="border border-[var(--line)] bg-white p-5">
          <h2 className="text-2xl font-light">后端接口占位</h2>
          <div className="mt-5 grid gap-2">
            {apiPlaceholders.map((endpoint) => (
              <div key={endpoint} className="border border-[var(--line)] px-3 py-2 font-mono text-xs">
                {endpoint}
              </div>
            ))}
          </div>
        </section>
      </div>

      <ProfileSummary dimensions={profileDimensions} />
    </div>
  );
}
