import Link from "next/link";
import { ArrowLeft } from "lucide-react";

export default async function CourseOnboardingPage({
  params,
}: {
  params: Promise<{ courseId: string }>;
}) {
  const { courseId } = await params;

  return (
    <div className="h-full overflow-y-auto bg-[var(--background)]">
      <div className="mx-auto flex min-h-full w-full max-w-4xl flex-col px-6 py-8">
        <Link
          href="/courses"
          className="mb-6 inline-flex w-fit items-center gap-2 text-sm text-[var(--muted-foreground)] transition-colors hover:text-[var(--foreground)]"
        >
          <ArrowLeft size={16} />
          返回课程
        </Link>

        <header className="mb-6">
          <p className="text-sm font-medium text-[var(--primary)]">学习画像</p>
          <h1 className="mt-2 text-3xl font-semibold tracking-normal text-[var(--foreground)]">
            生成你的高等数学精通之路
          </h1>
          <p className="mt-3 max-w-2xl text-sm leading-6 text-[var(--muted-foreground)]">
            回答四个问题后，系统会裁剪课程模板并生成路径预览。
          </p>
        </header>

        <form
          action={`/courses/${encodeURIComponent(courseId)}/onboarding/submit`}
          method="post"
          className="space-y-4 rounded-lg border border-[var(--border)] bg-[var(--card)] p-5"
        >
          {[
            ["prerequisite", "先修基础", "例如：高中函数还可以，导数有点忘"],
            ["goal", "学习目标", "例如：期末考试提分，希望系统补基础"],
            ["pace", "时间节奏", "例如：每周三次，每次一小时，一个月后考试"],
            ["weak_points", "薄弱点", "例如：极限、积分、应用题"],
          ].map(([name, label, placeholder]) => (
            <label key={name} className="block">
              <span className="text-sm font-medium text-[var(--foreground)]">
                {label}
              </span>
              <textarea
                name={name}
                required
                rows={3}
                placeholder={placeholder}
                className="mt-2 w-full rounded-md border border-[var(--border)] bg-[var(--background)] px-3 py-2 text-sm outline-none transition-colors focus:border-[var(--primary)]"
              />
            </label>
          ))}

          <button
            type="submit"
            className="inline-flex h-10 items-center justify-center rounded-md bg-[var(--primary)] px-4 text-sm font-medium text-[var(--primary-foreground)] transition-colors hover:opacity-90"
          >
            确认，生成我的精通之路
          </button>
        </form>
      </div>
    </div>
  );
}
