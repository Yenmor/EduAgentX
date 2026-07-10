import Link from "next/link";
import type { CourseMeta } from "@/lib/classroom-types";
import { ProfileChipBar } from "./ProfileChipBar";

type ClassroomShellProps = {
  title: string;
  courseGroupName: string;
  currentCourses: CourseMeta[];
  relatedCourses: CourseMeta[];
  knowledgeSources: string[];
  prerequisites: string[];
  profile: string[];
  progress: number;
  onReplan?: () => void;
  replanDisabled?: boolean;
  leftSlot: React.ReactNode;
  centerSlot: React.ReactNode;
  rightSlot: React.ReactNode;
};

export function ClassroomShell({
  title,
  courseGroupName,
  currentCourses,
  relatedCourses,
  knowledgeSources,
  prerequisites,
  profile,
  progress,
  onReplan,
  replanDisabled = false,
  leftSlot,
  centerSlot,
  rightSlot
}: ClassroomShellProps) {
  return (
    <section className="min-h-[calc(100vh-48px)] overflow-hidden rounded-[28px] border border-white/10 bg-slate-950/50 shadow-[0_30px_120px_rgba(0,0,0,0.45)]">
      <header className="border-b border-white/10 bg-white/[0.045] px-5 py-4 backdrop-blur-xl">
        <div className="flex flex-col gap-4 xl:flex-row xl:items-start xl:justify-between">
          <div className="min-w-0">
            <div className="text-xs uppercase tracking-[0.28em] text-sky-300">Interactive Classroom</div>
            <h1 className="mt-2 text-2xl font-semibold text-slate-50">{title}</h1>
            <div className="mt-3">
              <ProfileChipBar chips={profile} />
            </div>
            <div className="mt-4 grid gap-2 text-xs text-slate-300 lg:grid-cols-2">
              <InfoLine label="课程群" value={courseGroupName} />
              <InfoLine label="当前课程" value={currentCourses.map((course) => course.name).join("、")} />
              <InfoLine label="关联课程" value={relatedCourses.map((course) => course.name).join("、")} />
              <InfoLine label="先修课程" value={prerequisites.join("、")} />
              <InfoLine label="知识库来源" value={knowledgeSources.join("、")} wide />
            </div>
          </div>
          <div className="flex flex-wrap items-center gap-3">
            <div className="w-48">
              <div className="mb-1 flex justify-between text-[11px] text-slate-400">
                <span>学习进度</span>
                <span>{progress}%</span>
              </div>
              <div className="h-2 overflow-hidden rounded-full bg-slate-900">
                <div className="h-full rounded-full bg-gradient-to-r from-sky-400 to-violet-400" style={{ width: `${progress}%` }} />
              </div>
            </div>
            <button type="button" className="rounded-full border border-white/10 bg-white/[0.06] px-4 py-2 text-sm text-slate-100 transition hover:bg-white/[0.1]">保存课堂</button>
            <button
              type="button"
              onClick={onReplan}
              disabled={replanDisabled}
              className="rounded-full border border-sky-300/30 bg-sky-300/10 px-4 py-2 text-sm text-sky-100 transition hover:bg-sky-300/20 disabled:cursor-not-allowed disabled:opacity-60"
            >
              重规划路径
            </button>
            <Link href="/studio" className="rounded-full border border-white/10 px-4 py-2 text-sm text-slate-300 transition hover:text-white">
              返回 Studio
            </Link>
          </div>
        </div>
      </header>
      <div className="grid min-h-[760px] gap-px bg-white/10 xl:grid-cols-[290px_minmax(0,1fr)_360px]">
        <aside className="bg-slate-950/70 p-4">{leftSlot}</aside>
        <main className="bg-[#080A14]/95 p-5">{centerSlot}</main>
        <aside className="bg-slate-950/70 p-4">{rightSlot}</aside>
      </div>
    </section>
  );
}

function InfoLine({ label, value, wide = false }: { label: string; value: string; wide?: boolean }) {
  return (
    <div className={["rounded-xl border border-white/10 bg-slate-950/35 px-3 py-2", wide ? "lg:col-span-2" : ""].join(" ")}>
      <span className="text-sky-200">{label}：</span>
      <span>{value}</span>
    </div>
  );
}
