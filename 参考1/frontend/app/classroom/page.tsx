"use client";

import { Suspense, useEffect, useMemo, useState } from "react";
import Link from "next/link";
import { useSearchParams } from "next/navigation";
import { AgentRoleCard } from "@/components/classroom/AgentRoleCard";
import { ClassroomShell } from "@/components/classroom/ClassroomShell";
import { GenerationTimeline } from "@/components/classroom/GenerationTimeline";
import { LearningCompanionChat } from "@/components/classroom/LearningCompanionChat";
import { MasteryProgressPanel } from "@/components/classroom/MasteryProgressPanel";
import { ResourceSceneCard } from "@/components/classroom/ResourceSceneCard";
import { StageTabs } from "@/components/classroom/StageTabs";
import { LearningPathTimeline } from "@/components/LearningPathTimeline";
import { getClassroomState, replanLearningPath } from "@/lib/classroom-api";
import type { ClassroomState } from "@/lib/classroom-types";
import type { LearningStep } from "@/lib/types";

function profileChips(state: ClassroomState): string[] {
  return [
    state.profile.major,
    state.profile.foundation,
    `目标：${state.profile.goal}`,
    `节奏：${state.profile.pace}`,
    ...state.profile.preferences.slice(0, 2)
  ];
}

function toLearningSteps(state: ClassroomState): LearningStep[] {
  return state.learningPath.map((item, index) => ({
    order: index + 1,
    title: item.title,
    knowledge_point: item.focus,
    activity: item.activity,
    estimated_minutes: 45,
    mastery_target: item.status === "done" ? 0.82 : item.status === "active" ? 0.68 : 0.55
  }));
}

function ClassroomPageContent() {
  const searchParams = useSearchParams();
  const sessionId = searchParams.get("sessionId") || "demo";

  const [state, setState] = useState<ClassroomState | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [replanning, setReplanning] = useState(false);

  useEffect(() => {
    let cancelled = false;
    async function load() {
      setLoading(true);
      setError(null);
      try {
        const classroom = await getClassroomState(sessionId);
        if (!cancelled) setState(classroom);
      } catch (err) {
        if (!cancelled) setError(err instanceof Error ? err.message : "加载课堂失败");
      } finally {
        if (!cancelled) setLoading(false);
      }
    }
    void load();
    return () => {
      cancelled = true;
    };
  }, [sessionId]);

  const derivedSources = useMemo(() => {
    if (!state) return [];
    return Array.from(new Set(state.resources.flatMap((resource) => resource.sourceChunks.map((chunk) => chunk.sourceFile)).filter(Boolean)));
  }, [state]);

  const prerequisites = useMemo(() => {
    if (!state) return [];
    return Array.from(new Set(state.currentCourses.flatMap((course) => course.prerequisites))).filter(Boolean);
  }, [state]);

  async function handleReplan() {
    if (!state || replanning) return;
    setReplanning(true);
    try {
      const nextState = await replanLearningPath(state.sessionId);
      setState(nextState);
    } finally {
      setReplanning(false);
    }
  }

  if (loading) {
    return <div className="rounded-3xl border border-[var(--line)] bg-white p-8 text-sm text-[var(--muted)]">正在加载互动课堂...</div>;
  }

  if (error || !state) {
    return (
      <div className="space-y-4 rounded-3xl border border-[var(--line)] bg-white p-8">
        <div className="text-xl font-semibold text-[var(--ink)]">课堂加载失败</div>
        <div className="text-sm text-[var(--muted)]">{error || "未找到课堂数据。"}</div>
        <Link href="/studio" className="inline-flex bg-[var(--accent)] px-4 py-2.5 text-sm font-semibold text-white">
          返回 Studio 重新生成
        </Link>
      </div>
    );
  }

  return (
    <ClassroomShell
      title={state.title}
      courseGroupName={state.courseGroupName}
      currentCourses={state.currentCourses}
      relatedCourses={state.relatedCourses}
      knowledgeSources={derivedSources.length ? derivedSources : ["课程知识库"]}
      prerequisites={prerequisites.length ? prerequisites : ["Python 程序设计", "大语言模型基础"]}
      profile={profileChips(state)}
      progress={state.progress}
      onReplan={handleReplan}
      replanDisabled={replanning}
      leftSlot={
        <div className="space-y-4">
          <GenerationTimeline steps={state.agentTrace} compact />
          <div className="space-y-3">
            {state.agentTrace.map((agent) => (
              <AgentRoleCard key={agent.agentId} agent={agent} />
            ))}
          </div>
        </div>
      }
      centerSlot={
        <div className="space-y-5">
          <GenerationTimeline steps={state.agentTrace} />
          <StageTabs
            sessionId={state.sessionId}
            slides={state.slides}
            whiteboardChart={state.whiteboardChart}
            mindmapChart={state.mindmapChart}
            quiz={state.quiz}
            codeLab={state.codeLab}
            project={state.project}
          />
        </div>
      }
      rightSlot={
        <div className="space-y-4">
          <LearningCompanionChat />
          <MasteryProgressPanel mastery={state.masteryMap} />
          <section className="space-y-3">
            <div className="flex items-center justify-between">
              <h2 className="text-sm font-semibold text-slate-100">学习路径</h2>
              <span className="text-[11px] text-slate-400">{state.learningPath.length} 步</span>
            </div>
            <LearningPathTimeline steps={toLearningSteps(state)} />
          </section>
          <section className="space-y-3">
            <div className="flex items-center justify-between">
              <h2 className="text-sm font-semibold text-slate-100">课堂资源</h2>
              <span className="text-[11px] text-slate-400">{state.resources.length} 项</span>
            </div>
            {state.resources.slice(0, 4).map((resource) => (
              <ResourceSceneCard key={resource.id} resource={resource} />
            ))}
          </section>
        </div>
      }
    />
  );
}

export default function ClassroomPage() {
  return (
    <Suspense fallback={<div className="rounded-3xl border border-[var(--line)] bg-white p-8 text-sm text-[var(--muted)]">正在加载互动课堂...</div>}>
      <ClassroomPageContent />
    </Suspense>
  );
}
