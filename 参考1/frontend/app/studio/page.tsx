"use client";

import Link from "next/link";
import { Suspense, useMemo, useState } from "react";
import { useSearchParams } from "next/navigation";
import { HeroGeneratePanel } from "@/components/classroom/HeroGeneratePanel";
import { PageHeader } from "@/components/PageHeader";
import { courseLevels, defaultSelectedCourseIds, sampleCourseCatalog } from "@/lib/mock-classroom";
import { startClassroomGeneration } from "@/lib/classroom-api";

const samplePrompts = [
  "我学过深度学习，但不了解 RAG 和 LangChain，想完成一个课程知识库问答项目",
  "我想先补 Python 和机器学习基础，再逐步进入大模型应用开发",
  "我要准备 AI 课程答辩，想把 RAG、LangChain 和多智能体系统串成一个演示"
];

function StudioPageContent() {
  const searchParams = useSearchParams();
  const preselectedCourseId = searchParams.get("courseId");

  const [goal, setGoal] = useState(
    preselectedCourseId
      ? "我想围绕这门课生成一个可互动的 AI 课堂，并完成讲解、测验和项目练习。"
      : samplePrompts[0]
  );
  const [selectedCourseIds, setSelectedCourseIds] = useState<string[]>(
    preselectedCourseId ? Array.from(new Set([preselectedCourseId, ...defaultSelectedCourseIds])) : defaultSelectedCourseIds
  );
  const [loading, setLoading] = useState(false);

  const coursesByLevel = useMemo(
    () =>
      courseLevels.reduce<Record<string, typeof sampleCourseCatalog>>((acc, level) => {
        acc[level] = sampleCourseCatalog.filter((course) => course.level === level);
        return acc;
      }, {}),
    []
  );

  async function handleGenerate() {
    if (!goal.trim() || loading) return;
    setLoading(true);
    try {
      const state = await startClassroomGeneration({ goal, selectedCourseIds });
      window.location.assign(`/classroom?sessionId=${encodeURIComponent(state.sessionId)}`);
    } finally {
      setLoading(false);
    }
  }

  function toggleCourse(courseId: string) {
    setSelectedCourseIds((current) => {
      if (current.includes(courseId)) {
        return current.length === 1 ? current : current.filter((id) => id !== courseId);
      }
      return [...current, courseId];
    });
  }

  return (
    <div className="space-y-8">
      <PageHeader
        eyebrow="Studio"
        title="生成你的互动式 AI 课堂"
        description="选择课程群、描述学习目标，系统会调用后端多智能体流程生成课堂讲解、测验、代码实验与项目任务。"
        aside={
          <div>
            <div className="font-mono text-xs text-[var(--muted)]">课堂输出</div>
            <div className="mt-2 text-sm leading-6 text-[var(--muted)]">
              Slides / Whiteboard / Mindmap / Quiz / Code Lab / Project
            </div>
          </div>
        }
      />

      <HeroGeneratePanel
        value={goal}
        onChange={setGoal}
        onGenerate={handleGenerate}
        examples={samplePrompts}
        loading={loading}
        coursesByLevel={coursesByLevel}
        selectedCourseIds={selectedCourseIds}
        onToggleCourse={toggleCourse}
      />

      <div className="flex justify-center">
        <Link href="/classroom?sessionId=demo" className="inline-flex border border-[var(--accent)] bg-white px-4 py-2.5 text-sm font-semibold text-[var(--accent)]">
          直接进入 Demo 课堂
        </Link>
      </div>
    </div>
  );
}

export default function StudioPage() {
  return (
    <Suspense fallback={<div className="rounded-3xl border border-[var(--line)] bg-white p-8 text-sm text-[var(--muted)]">正在加载 Studio...</div>}>
      <StudioPageContent />
    </Suspense>
  );
}
