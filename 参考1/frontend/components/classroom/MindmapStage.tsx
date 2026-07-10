import { MermaidRenderer } from "@/components/MermaidRenderer";

type MindmapStageProps = {
  chart: string;
};

export function MindmapStage({ chart }: MindmapStageProps) {
  return (
    <div className="rounded-[28px] border border-white/10 bg-white/[0.055] p-5">
      <div className="text-xs uppercase tracking-[0.24em] text-violet-300">Mindmap</div>
      <h2 className="mt-2 text-2xl font-semibold text-slate-50">高校 AI 课程群知识图谱</h2>
      <p className="mt-2 text-sm text-slate-400">从基础工具、核心课程到 RAG、LangChain、AI Agent 的项目化主线。</p>
      <div className="mt-5 rounded-[24px] border border-white/10 bg-slate-950/60 p-5">
        <MermaidRenderer chart={chart} />
      </div>
    </div>
  );
}
