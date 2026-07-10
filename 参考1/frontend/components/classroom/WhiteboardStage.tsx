import { MermaidRenderer } from "@/components/MermaidRenderer";

type WhiteboardStageProps = {
  chart: string;
  annotation?: string;
};

export function WhiteboardStage({ chart, annotation = "AI 主讲教师正在把 RAG 检索流程展开到课程知识库场景" }: WhiteboardStageProps) {
  return (
    <div className="rounded-[28px] border border-white/10 bg-white/[0.055] p-5">
      <div className="mb-4 flex flex-wrap items-center justify-between gap-3">
        <div>
          <div className="text-xs uppercase tracking-[0.24em] text-sky-300">Whiteboard</div>
          <h2 className="mt-2 text-2xl font-semibold text-slate-50">RAG 知识库 Grounding 流程</h2>
        </div>
        <div className="rounded-full border border-sky-300/20 bg-sky-300/10 px-3 py-1 text-xs text-sky-100">{annotation}</div>
      </div>
      <div className="rounded-[24px] border border-white/10 bg-slate-950/60 p-5">
        <MermaidRenderer chart={chart} />
      </div>
    </div>
  );
}
