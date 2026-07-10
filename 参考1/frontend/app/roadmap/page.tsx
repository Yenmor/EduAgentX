"use client";

import { useState } from "react";
import { GenerationProgress } from "@/components/GenerationProgress";
import { LearningRequestBox } from "@/components/LearningRequestBox";
import { PageHeader } from "@/components/PageHeader";
import { ProfileSummary } from "@/components/ProfileSummary";
import { RoadmapStage } from "@/components/RoadmapStage";
import { generationEvents, profileDimensions, roadmapStages as initialStages } from "@/lib/learning-data";

const pathTypes = ["课程补强", "考试复习", "项目实践", "科研入门", "就业面试", "自定义目标"];

export default function RoadmapPage() {
  const [stages, setStages] = useState(initialStages);
  const [activeType, setActiveType] = useState(pathTypes[0]);

  function move(index: number, direction: -1 | 1) {
    setStages((current) => {
      const next = [...current];
      const target = index + direction;
      if (target < 0 || target >= next.length) return current;
      [next[index], next[target]] = [next[target], next[index]];
      return next;
    });
  }

  function replaceResource(index: number) {
    setStages((current) =>
      current.map((stage, stageIndex) =>
        stageIndex === index ? { ...stage, resources: ["自适应讲义包", "错题重练题库", "HTML 交互模拟"] } : stage
      )
    );
  }

  return (
    <div className="space-y-8">
      <PageHeader
        eyebrow="Roadmap"
        title="把学习目标拆成可执行阶段"
        description="路径规划基于画像、进度和测验结果动态调整。用户只需要描述目标、基础和可投入时间，系统生成阶段、课程、资源、测验和预计时间。"
      />

      <div className="grid gap-6 lg:grid-cols-[1fr_360px]">
        <LearningRequestBox
          title="生成学习路径"
          placeholder="例如：我 4 周后要考人工智能导论，基础一般，每天能学 1 小时，想先补概念再做题。"
          buttonLabel="生成阶段式路径"
          defaultValue="目标：补齐人工智能导论并完成课程项目；基础：Python 可用，数学偏弱；时间：每天 1 小时。"
        />
        <div className="border border-[var(--line)] bg-white p-5">
          <div className="text-lg font-medium">路径类型</div>
          <div className="mt-4 grid grid-cols-2 gap-2">
            {pathTypes.map((type) => (
              <button
                key={type}
                onClick={() => setActiveType(type)}
                className={["border px-3 py-2 text-sm", activeType === type ? "border-[var(--accent)] bg-[var(--accent)] text-white" : "border-[var(--line)]"].join(" ")}
              >
                {type}
              </button>
            ))}
          </div>
          <p className="mt-4 text-sm leading-6 text-[var(--muted)]">当前路径：{activeType}。阶段顺序和资源都可调整。</p>
        </div>
      </div>

      <div className="space-y-4">
        {stages.map((stage, index) => (
          <RoadmapStage
            key={stage.id}
            stage={stage}
            index={index}
            onMoveUp={index > 0 ? () => move(index, -1) : undefined}
            onMoveDown={index < stages.length - 1 ? () => move(index, 1) : undefined}
            onReplace={() => replaceResource(index)}
          />
        ))}
      </div>

      <div className="grid gap-6 lg:grid-cols-2">
        <GenerationProgress events={generationEvents} />
        <ProfileSummary dimensions={profileDimensions} />
      </div>
    </div>
  );
}
