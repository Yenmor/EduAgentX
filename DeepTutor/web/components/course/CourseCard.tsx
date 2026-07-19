"use client";

import { ArrowRight, Clock3, GraduationCap, LibraryBig } from "lucide-react";
import Link from "next/link";
import type { Course } from "@/lib/course-api";

interface CourseCardProps {
  course: Course;
  href: string;
}

export default function CourseCard({ course, href }: CourseCardProps) {
  return (
    <article className="rounded-lg border border-[var(--border)] bg-[var(--card)] p-6 shadow-sm">
      <div className="flex flex-wrap items-center gap-2 text-xs text-[var(--muted-foreground)]">
        <span className="inline-flex items-center gap-1 rounded-md bg-[var(--accent)] px-2 py-1">
          <GraduationCap size={13} />
          {course.difficulty}
        </span>
        <span className="inline-flex items-center gap-1 rounded-md bg-[var(--accent)] px-2 py-1">
          <Clock3 size={13} />
          {course.duration}
        </span>
        <span className="inline-flex items-center gap-1 rounded-md bg-[var(--accent)] px-2 py-1">
          <LibraryBig size={13} />
          {course.kb_name}
        </span>
      </div>

      <div className="mt-5 grid gap-6 lg:grid-cols-[1fr_auto] lg:items-end">
        <div>
          <h2 className="text-2xl font-semibold tracking-normal text-[var(--foreground)]">
            {course.title}
          </h2>
          <p className="mt-3 max-w-3xl text-sm leading-6 text-[var(--muted-foreground)]">
            {course.summary}
          </p>
          <ul className="mt-5 grid gap-2 text-sm text-[var(--foreground)]/85 md:grid-cols-3">
            {course.outcomes.map((outcome) => (
              <li key={outcome} className="rounded-md bg-[var(--secondary)] px-3 py-2">
                {outcome}
              </li>
            ))}
          </ul>
        </div>

        <Link
          href={href}
          className="inline-flex h-10 items-center justify-center gap-2 rounded-md bg-[var(--primary)] px-4 text-sm font-medium text-[var(--primary-foreground)] transition-colors hover:opacity-90"
        >
          开始学习
          <ArrowRight size={16} />
        </Link>
      </div>
    </article>
  );
}
