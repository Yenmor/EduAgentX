import Link from "next/link";
import { notFound } from "next/navigation";
import { PageHeader } from "@/components/PageHeader";
import { ResourceCard } from "@/components/ResourceCard";
import { SafetyBadge } from "@/components/SafetyBadge";
import { courses, resources } from "@/lib/learning-data";

export default function LearnPage({ params }: { params: { courseId: string } }) {
  const course = courses.find((item) => item.id === params.courseId);
  if (!course) notFound();

  const courseResources = resources.filter((resource) => resource.courseId === course.id);

  return (
    <div className="space-y-8">
      <PageHeader
        eyebrow="Learn"
        title={course.title}
        description={course.summary}
        aside={
          <div>
            <div className="font-mono text-xs text-[var(--muted)]">学习时长</div>
            <div className="mt-2 text-3xl font-light">{course.duration}</div>
          </div>
        }
      />

      <div className="grid gap-6 lg:grid-cols-[1fr_340px]">
        <section className="border border-[var(--line)] bg-white p-5">
          <div className="flex flex-wrap items-center justify-between gap-3 border-b border-[var(--line)] pb-4">
            <h2 className="text-2xl font-light">学习单元</h2>
            <SafetyBadge status="通过" />
          </div>
          <div className="mt-5 grid gap-px bg-[var(--line)]">
            {[
              ["01", "课程全局框架", "理解 AI 的问题建模、数据、模型和评估关系。"],
              ["02", "搜索与知识表示", "用状态、动作、目标和代价描述经典 AI 问题。"],
              ["03", "机器学习入门", "掌握训练、验证、泛化和过拟合的基本判断。"],
              ["04", "生成式 AI 速览", "理解大模型应用的基本链路和风险检查。"]
            ].map(([order, title, description]) => (
              <div key={order} className="grid gap-4 bg-white p-4 md:grid-cols-[80px_1fr]">
                <div className="font-mono text-3xl font-light text-[var(--accent)]">{order}</div>
                <div>
                  <h3 className="text-xl font-light">{title}</h3>
                  <p className="mt-2 text-sm leading-6 text-[var(--muted)]">{description}</p>
                </div>
              </div>
            ))}
          </div>
        </section>
        <aside className="space-y-4">
          <Link href={`/studio?courseId=${course.id}`} className="block bg-[var(--accent)] px-5 py-4 text-sm font-semibold text-white">
            进入互动课堂
          </Link>
          <Link href="/quiz" className="block bg-[var(--accent)] px-5 py-4 text-sm font-semibold text-white">
            开始章节测验
          </Link>
          <Link href="/roadmap" className="block border border-[var(--accent)] bg-white px-5 py-4 text-sm font-semibold text-[var(--accent)]">
            加入学习路径
          </Link>
          <div className="border border-[var(--line)] bg-white p-5">
            <div className="font-mono text-xs text-[var(--muted)]">学习成果</div>
            <ul className="mt-4 space-y-3 text-sm leading-6">
              {course.outcomes.map((outcome) => (
                <li key={outcome} className="border-l-2 border-[var(--accent)] pl-3">
                  {outcome}
                </li>
              ))}
            </ul>
          </div>
        </aside>
      </div>

      <section>
        <div className="mb-4 flex items-end justify-between border-b border-[var(--line)] pb-3">
          <h2 className="text-3xl font-light">课程资源</h2>
          <span className="font-mono text-xs text-[var(--muted)]">POST /api/resources/generate</span>
        </div>
        <div className="grid gap-4 md:grid-cols-2 xl:grid-cols-4">
          {courseResources.map((resource) => (
            <ResourceCard key={resource.id} resource={resource} />
          ))}
        </div>
      </section>
    </div>
  );
}
