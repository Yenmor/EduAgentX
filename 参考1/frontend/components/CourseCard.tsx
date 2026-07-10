import Link from "next/link";
import type { Course } from "@/lib/learning-data";

export function CourseCard({ course }: { course: Course }) {
  return (
    <article className="grid gap-5 border border-[var(--line)] bg-white p-5 md:grid-cols-[1fr_160px]">
      <div>
        <div className="flex flex-wrap gap-2 font-mono text-xs text-[var(--muted)]">
          <span>{course.category}</span>
          <span>/</span>
          <span>{course.difficulty}</span>
          <span>/</span>
          <span>{course.duration}</span>
        </div>
        <h2 className="mt-3 text-3xl font-light leading-tight">{course.title}</h2>
        <p className="mt-3 text-sm leading-6 text-[var(--muted)]">{course.summary}</p>
        <div className="mt-4 flex flex-wrap gap-2">
          {course.resourceTypes.map((type) => (
            <span key={type} className="border border-[var(--line)] px-2 py-1 text-xs">
              {type}
            </span>
          ))}
        </div>
        <div className="mt-5 grid gap-2 text-sm md:grid-cols-3">
          {course.outcomes.map((outcome) => (
            <div key={outcome} className="border-l-2 border-[var(--accent)] pl-3">
              {outcome}
            </div>
          ))}
        </div>
      </div>
      <div className="flex flex-col justify-between border-t border-[var(--line)] pt-4 md:border-l md:border-t-0 md:pl-5 md:pt-0">
        <div>
          <div className="font-mono text-xs uppercase tracking-[0.16em] text-[var(--muted)]">匹配度</div>
          <div className="mt-2 text-5xl font-extralight text-[var(--accent)]">{course.score}</div>
        </div>
        <div className="mt-5 grid gap-2">
          <Link href={`/learn/${course.id}`} className="inline-flex justify-center bg-[var(--accent)] px-4 py-2.5 text-sm font-semibold text-white">
            开始学习
          </Link>
          <Link href={`/studio?courseId=${course.id}`} className="inline-flex justify-center border border-[var(--accent)] bg-white px-4 py-2.5 text-sm font-semibold text-[var(--accent)]">
            进入互动课堂
          </Link>
        </div>
      </div>
    </article>
  );
}
