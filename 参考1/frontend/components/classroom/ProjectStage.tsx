import type { ProjectTask } from "@/lib/classroom-types";

type ProjectStageProps = {
  task: ProjectTask;
};

export function ProjectStage({ task }: ProjectStageProps) {
  const sections = [
    ["项目目标", task.goals],
    ["里程碑", task.milestones],
    ["交付物", task.deliverables],
    ["评价标准", task.rubric]
  ] as const;

  return (
    <div className="rounded-[28px] border border-white/10 bg-[radial-gradient(circle_at_20%_0%,rgba(168,85,247,0.15),transparent_34%),rgba(255,255,255,0.055)] p-5">
      <div className="text-xs uppercase tracking-[0.24em] text-fuchsia-300">Project</div>
      <h2 className="mt-2 text-3xl font-semibold text-slate-50">{task.title}</h2>
      <p className="mt-3 max-w-4xl text-sm leading-6 text-slate-300">{task.background}</p>
      <div className="mt-6 grid gap-4 md:grid-cols-2">
        {sections.map(([title, items]) => (
          <section key={title} className="rounded-2xl border border-white/10 bg-slate-950/45 p-4">
            <h3 className="text-sm font-semibold text-slate-100">{title}</h3>
            <ul className="mt-3 space-y-2">
              {items.map((item) => (
                <li key={item} className="flex gap-3 text-sm leading-6 text-slate-300">
                  <span className="mt-2 h-1.5 w-1.5 shrink-0 rounded-full bg-violet-300" />
                  {item}
                </li>
              ))}
            </ul>
          </section>
        ))}
      </div>
    </div>
  );
}
