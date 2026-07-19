"use client";

import { useEffect, useState } from "react";
import { AlertCircle } from "lucide-react";
import CourseCard from "@/components/course/CourseCard";
import { fetchCourses, type Course } from "@/lib/course-api";

export default function CoursesClient({
  initialCourses,
}: {
  initialCourses: Course[];
}) {
  const [courses, setCourses] = useState<Course[]>(initialCourses);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    let cancelled = false;
    fetchCourses()
      .then((items) => {
        if (!cancelled && items.length) setCourses(items);
      })
      .catch((err) => {
        if (!cancelled) {
          setError(err instanceof Error ? err.message : "加载课程失败");
        }
      });
    return () => {
      cancelled = true;
    };
  }, []);

  return (
    <div className="h-full overflow-y-auto bg-[var(--background)]">
      <div className="mx-auto flex min-h-full w-full max-w-6xl flex-col px-6 py-10">
        <header className="mb-8">
          <p className="text-sm font-medium text-[var(--primary)]">课程</p>
          <h1 className="mt-2 text-3xl font-semibold tracking-normal text-[var(--foreground)]">
            选择一门课开始学习
          </h1>
          <p className="mt-3 max-w-2xl text-sm leading-6 text-[var(--muted-foreground)]">
            每门课都内置知识库和精通之路模板，开始前会先通过几轮对话生成你的个性化路径。
          </p>
        </header>

        {error ? (
          <div className="mb-4 flex items-center gap-2 rounded-lg border border-[var(--border)] bg-[var(--card)] p-4 text-sm text-amber-700">
            <AlertCircle size={16} />
            使用内置课程目录。后台课程接口暂时不可用：{error}
          </div>
        ) : null}

        <div className="space-y-4">
          {courses.map((course) => (
            <CourseCard
              key={course.id}
              course={course}
              href={`/courses/${encodeURIComponent(course.id)}/onboarding`}
            />
          ))}
        </div>
      </div>
    </div>
  );
}
