import type { ResourceScene } from "@/lib/classroom-types";

type ResourceSceneCardProps = {
  resource: ResourceScene;
};

const typeMap: Record<ResourceScene["type"], string> = {
  doc: "讲解文档",
  mindmap: "思维导图",
  quiz: "测验",
  reading: "拓展阅读",
  code: "代码实验",
  animation: "动画脚本",
  project: "PBL 项目"
};

export function ResourceSceneCard({ resource }: ResourceSceneCardProps) {
  const locked = resource.status === "locked";

  return (
    <article className={[
      "relative rounded-2xl border border-white/10 bg-white/[0.055] p-4 transition hover:border-sky-300/30 hover:bg-white/[0.08]",
      locked ? "opacity-55" : ""
    ].join(" ")}>
      {resource.status === "recommended" ? (
        <div className="absolute right-3 top-3 rounded-full bg-sky-300/15 px-2.5 py-1 text-[11px] text-sky-100">推荐</div>
      ) : null}
      <div className="flex items-start justify-between gap-3 pr-12">
        <div>
          <div className="flex flex-wrap gap-2">
            <span className="rounded-full bg-sky-300/10 px-2.5 py-1 text-[11px] text-sky-100">{typeMap[resource.type]}</span>
            <span className="rounded-full bg-white/[0.06] px-2.5 py-1 text-[11px] text-slate-300">{resource.difficulty}</span>
            <span className="rounded-full bg-white/[0.06] px-2.5 py-1 text-[11px] text-slate-300">{resource.estimatedMinutes} min</span>
          </div>
          <h3 className="mt-3 text-sm font-semibold text-slate-100">{resource.title}</h3>
        </div>
        {resource.status === "done" ? <span className="text-emerald-300">✓</span> : null}
      </div>
      <div className="mt-3 space-y-2 text-xs leading-5 text-slate-300">
        <div><span className="text-slate-100">课程：</span>{resource.courseName}</div>
        <div><span className="text-slate-100">知识点：</span>{resource.targetConcepts.join("、")}</div>
        <div><span className="text-slate-100">先修：</span>{resource.prerequisiteConcepts.join("、")}</div>
        <div><span className="text-slate-100">来源章节：</span>{resource.sourceChapter}</div>
        <div><span className="text-slate-100">Grounding：</span>引用 {resource.sourceChunks.length} 个课程片段</div>
        <div><span className="text-slate-100">Judge：</span>{resource.judge.score}/100，{resource.judge.grounded ? "通过 grounded 检查" : "需要补充引用"}</div>
      </div>
      <p className="mt-3 rounded-xl border border-white/10 bg-slate-950/40 p-3 text-xs leading-5 text-slate-300">
        {resource.personalizedReason}
      </p>
      <div className="mt-3 font-mono text-[11px] text-slate-500">source: {resource.source}</div>
      <button type="button" disabled={locked} className="mt-4 w-full rounded-full border border-white/10 bg-white/[0.06] px-3 py-2 text-xs text-slate-100 transition hover:bg-sky-300/10 disabled:cursor-not-allowed disabled:opacity-60">
        {locked ? "完成先修后解锁" : "加入学习任务"}
      </button>
    </article>
  );
}
