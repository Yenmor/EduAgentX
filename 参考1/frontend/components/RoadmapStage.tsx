import type { RoadmapStageData } from "@/lib/learning-data";

type RoadmapStageProps = {
  stage: RoadmapStageData;
  index: number;
  onMoveUp?: () => void;
  onMoveDown?: () => void;
  onReplace?: () => void;
};

export function RoadmapStage({ stage, index, onMoveUp, onMoveDown, onReplace }: RoadmapStageProps) {
  return (
    <article className="grid gap-5 border border-[var(--line)] bg-white p-5 md:grid-cols-[120px_1fr_180px]">
      <div>
        <div className="font-mono text-xs uppercase tracking-[0.16em] text-[var(--muted)]">{stage.phase}</div>
        <div className="mt-2 text-6xl font-extralight text-[var(--accent)]">{String(index + 1).padStart(2, "0")}</div>
      </div>
      <div>
        <h3 className="text-2xl font-light">{stage.goal}</h3>
        <div className="mt-5 grid gap-4 md:grid-cols-3">
          <div>
            <div className="font-mono text-xs text-[var(--muted)]">推荐课程</div>
            <p className="mt-2 text-sm leading-6">{stage.courses.join(" / ")}</p>
          </div>
          <div>
            <div className="font-mono text-xs text-[var(--muted)]">资源</div>
            <p className="mt-2 text-sm leading-6">{stage.resources.join(" / ")}</p>
          </div>
          <div>
            <div className="font-mono text-xs text-[var(--muted)]">测验</div>
            <p className="mt-2 text-sm leading-6">{stage.quiz}</p>
          </div>
        </div>
      </div>
      <div className="border-t border-[var(--line)] pt-4 md:border-l md:border-t-0 md:pl-5 md:pt-0">
        <div className="font-mono text-xs uppercase tracking-[0.16em] text-[var(--muted)]">预计时间</div>
        <div className="mt-2 text-3xl font-light">{stage.estimatedTime}</div>
        <div className="mt-5 flex flex-wrap gap-2">
          <button onClick={onMoveUp} className="border border-[var(--line)] px-3 py-2 text-xs disabled:opacity-30" disabled={!onMoveUp}>
            上移
          </button>
          <button onClick={onMoveDown} className="border border-[var(--line)] px-3 py-2 text-xs disabled:opacity-30" disabled={!onMoveDown}>
            下移
          </button>
          <button onClick={onReplace} className="border border-[var(--accent)] px-3 py-2 text-xs text-[var(--accent)]">
            替换资源
          </button>
        </div>
      </div>
    </article>
  );
}
