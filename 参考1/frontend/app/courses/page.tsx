"use client";

import Link from "next/link";
import { useMemo, useState } from "react";
import { CourseCard } from "@/components/CourseCard";
import { EmptyState } from "@/components/EmptyState";
import { FilterRail } from "@/components/FilterRail";
import { LearningRequestBox } from "@/components/LearningRequestBox";
import { PageHeader } from "@/components/PageHeader";
import { ResourceCard } from "@/components/ResourceCard";
import { courses, resources } from "@/lib/learning-data";

const unique = (values: string[]) => Array.from(new Set(values));

export default function CoursesPage() {
  const [request, setRequest] = useState("我想从人工智能导论开始，补齐机器学习基础。");
  const [filters, setFilters] = useState({
    category: "全部",
    difficulty: "全部",
    goal: "全部",
    resourceType: "全部"
  });

  const filteredCourses = useMemo(() => {
    const text = request.toLowerCase();
    return courses
      .filter((course) => filters.category === "全部" || course.category === filters.category)
      .filter((course) => filters.difficulty === "全部" || course.difficulty === filters.difficulty)
      .filter((course) => filters.goal === "全部" || course.goal === filters.goal)
      .filter((course) => filters.resourceType === "全部" || course.resourceTypes.includes(filters.resourceType as never))
      .filter((course) => {
        if (!text) return true;
        return [course.title, course.summary, course.category, course.goal, ...course.recommendedFor].join(" ").toLowerCase().includes(text) || course.id === "intro-ai";
      });
  }, [filters, request]);

  const recommendedResources = resources.filter((resource) => filteredCourses.some((course) => course.id === resource.courseId)).slice(0, 4);

  return (
    <div className="space-y-8">
      <PageHeader
        eyebrow="Courses"
        title="选择课程，生成学习入口"
        description="筛选课程，描述需求，查看资源并开始学习。示范课程默认为人工智能导论，但平台不限制在单一知识点。"
        aside={
          <div className="space-y-3">
            <div>
              <div className="font-mono text-xs text-[var(--muted)]">默认示范课程</div>
              <div className="mt-2 text-2xl font-light">人工智能导论</div>
            </div>
            <Link href="/studio" className="inline-flex bg-[var(--accent)] px-4 py-2.5 text-sm font-semibold text-white">
              直接生成 AI 课堂
            </Link>
          </div>
        }
      />

      <LearningRequestBox
        placeholder="例如：我是计算机大二学生，Python 还可以，但概率统计薄弱，想 4 周内补完人工智能导论并做一个小项目。"
        buttonLabel="推荐课程与资源"
        defaultValue={request}
        onSubmit={setRequest}
      />

      <div className="grid gap-6 lg:grid-cols-[280px_1fr]">
        <FilterRail
          categories={unique(courses.map((course) => course.category))}
          difficulties={unique(courses.map((course) => course.difficulty))}
          goals={unique(courses.map((course) => course.goal))}
          resourceTypes={unique(courses.flatMap((course) => course.resourceTypes))}
          selected={filters}
          onChange={(key, value) => setFilters((current) => ({ ...current, [key]: value }))}
        />
        <section className="space-y-5">
          {filteredCourses.length ? (
            filteredCourses.map((course) => <CourseCard key={course.id} course={course} />)
          ) : (
            <EmptyState title="没有匹配课程" description="可以放宽筛选条件，或直接输入更具体的学习目标来生成自定义计划。" />
          )}
        </section>
      </div>

      <section>
        <div className="mb-4 flex items-end justify-between border-b border-[var(--line)] pb-3">
          <h2 className="text-3xl font-light">推荐资源</h2>
          <span className="font-mono text-xs text-[var(--muted)]">POST /api/courses/recommend</span>
        </div>
        <div className="grid gap-4 md:grid-cols-2 xl:grid-cols-4">
          {recommendedResources.map((resource) => (
            <ResourceCard key={resource.id} resource={resource} />
          ))}
        </div>
      </section>
    </div>
  );
}
