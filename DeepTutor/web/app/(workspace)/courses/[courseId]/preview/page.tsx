import Link from "next/link";
import { ArrowLeft, ArrowRight, Sparkles } from "lucide-react";
import type { BuildPathResult } from "@/lib/course-api";
import type { MasteryMapResult } from "@/lib/learning-api";

const API_BASE_URL = process.env.DEEPTUTOR_API_BASE_URL ?? "http://127.0.0.1:8001";

async function getJson<T>(path: string): Promise<T | null> {
  const res = await fetch(`${API_BASE_URL}${path}`, { cache: "no-store" });
  if (!res.ok) return null;
  return (await res.json()) as T;
}

export default async function CoursePreviewPage({
  params,
  searchParams,
}: {
  params: Promise<{ courseId: string }>;
  searchParams: Promise<{ book_id?: string }>;
}) {
  const { courseId } = await params;
  const { book_id: bookId = "" } = await searchParams;
  const map = bookId
    ? await getJson<MasteryMapResult>(
        `/api/v1/learning/progress/${encodeURIComponent(bookId)}/map`,
      )
    : null;
  const plan = bookId
    ? await getJson<BuildPathResult>(
        `/api/v1/course/courses/${encodeURIComponent(
          courseId,
        )}/path-plan/${encodeURIComponent(bookId)}`,
      )
    : null;
  const kb = plan?.course.kb_name || "gaodengshuxue";
  const startHref = `/home/${encodeURIComponent(
    bookId,
  )}?capability=mastery_path&kb=${encodeURIComponent(kb)}&autostart=1`;

  return (
    <div className="h-full overflow-y-auto bg-[var(--background)]">
      <div className="mx-auto flex min-h-full w-full max-w-5xl flex-col px-6 py-8">
        <Link
          href={`/courses/${encodeURIComponent(courseId)}/onboarding`}
          className="mb-6 inline-flex w-fit items-center gap-2 text-sm text-[var(--muted-foreground)] transition-colors hover:text-[var(--foreground)]"
        >
          <ArrowLeft size={16} />
          重新调整
        </Link>

        <header className="mb-6 flex flex-col gap-4 md:flex-row md:items-end md:justify-between">
          <div>
            <p className="text-sm font-medium text-[var(--primary)]">路径预览</p>
            <h1 className="mt-2 text-3xl font-semibold tracking-normal text-[var(--foreground)]">
              你的高等数学精通之路
            </h1>
            <p className="mt-3 max-w-2xl text-sm leading-6 text-[var(--muted-foreground)]">
              起点、重点和模块顺序已根据学习画像调整，确认后将进入原生精通之路学习界面。
            </p>
          </div>
          {bookId ? (
            <Link
              href={startHref}
              className="inline-flex h-10 items-center justify-center gap-2 rounded-md bg-[var(--primary)] px-4 text-sm font-medium text-[var(--primary-foreground)] transition-colors hover:opacity-90"
            >
              开始学习
              <ArrowRight size={16} />
            </Link>
          ) : null}
        </header>

        {!bookId || !map ? (
          <div className="rounded-lg border border-[var(--border)] bg-[var(--card)] p-4 text-sm text-red-600">
            路径预览加载失败，请重新生成。
          </div>
        ) : (
          <div className="space-y-4">
            <section className="grid gap-3 md:grid-cols-3">
              <div className="rounded-lg border border-[var(--border)] bg-[var(--card)] p-4">
                <p className="text-xs text-[var(--muted-foreground)]">模块数</p>
                <p className="mt-2 text-2xl font-semibold">{map.map.modules.length}</p>
              </div>
              <div className="rounded-lg border border-[var(--border)] bg-[var(--card)] p-4">
                <p className="text-xs text-[var(--muted-foreground)]">知识点</p>
                <p className="mt-2 text-2xl font-semibold">{map.map.counts.total}</p>
              </div>
              <div className="rounded-lg border border-[var(--border)] bg-[var(--card)] p-4">
                <p className="text-xs text-[var(--muted-foreground)]">下一步</p>
                <p className="mt-2 text-sm font-medium">
                  {map.next.knowledge_point_name || "诊断"}
                </p>
              </div>
            </section>

            {plan ? (
              <section className="rounded-lg border border-[var(--border)] bg-[var(--card)] p-4">
                <div className="mb-3 flex items-center gap-2 text-sm font-medium">
                  <Sparkles size={16} />
                  个性化调整
                </div>
                <div className="grid gap-3 text-sm md:grid-cols-3">
                  <div>
                    <p className="text-[var(--muted-foreground)]">重点强化</p>
                    <p className="mt-1">
                      {plan.plan.focused.length
                        ? plan.plan.focused.join("、")
                        : "按默认顺序学习"}
                    </p>
                  </div>
                  <div>
                    <p className="text-[var(--muted-foreground)]">已压缩/跳过</p>
                    <p className="mt-1">
                      {plan.plan.skipped.length ? plan.plan.skipped.join("、") : "无"}
                    </p>
                  </div>
                  <div>
                    <p className="text-[var(--muted-foreground)]">模板来源</p>
                    <p className="mt-1">
                      {plan.plan.template_source === "template"
                        ? "备课模板"
                        : "内置 starter 模板"}
                    </p>
                  </div>
                </div>
              </section>
            ) : null}

            <section className="space-y-3">
              {map.map.modules.map((module) => (
                <article
                  key={module.id}
                  className="rounded-lg border border-[var(--border)] bg-[var(--card)] p-4"
                >
                  <div className="flex flex-wrap items-center justify-between gap-3">
                    <h2 className="text-base font-semibold text-[var(--foreground)]">
                      {module.order + 1}. {module.name}
                    </h2>
                    <span className="text-xs text-[var(--muted-foreground)]">
                      {module.mastered}/{module.total} 已掌握
                    </span>
                  </div>
                  <div className="mt-3 flex flex-wrap gap-2">
                    {module.knowledge_points.map((kp) => (
                      <span
                        key={kp.id}
                        className="rounded-md bg-[var(--secondary)] px-2.5 py-1.5 text-xs text-[var(--foreground)]/85"
                      >
                        {kp.name}
                      </span>
                    ))}
                  </div>
                </article>
              ))}
            </section>
          </div>
        )}
      </div>
    </div>
  );
}
