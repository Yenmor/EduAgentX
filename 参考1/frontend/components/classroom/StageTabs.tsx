"use client";

import { useState } from "react";
import type { CodeLabTask, ProjectTask, QuizQuestion, Slide } from "@/lib/classroom-types";
import { CodeLabStage } from "./CodeLabStage";
import { MindmapStage } from "./MindmapStage";
import { ProjectStage } from "./ProjectStage";
import { QuizStage } from "./QuizStage";
import { SlideStage } from "./SlideStage";
import { WhiteboardStage } from "./WhiteboardStage";

const tabs = ["Slides", "Whiteboard", "Mindmap", "Quiz", "Code Lab", "Project"] as const;
type StageTab = (typeof tabs)[number];

type StageTabsProps = {
  sessionId: string;
  slides: Slide[];
  whiteboardChart: string;
  mindmapChart: string;
  quiz: QuizQuestion[];
  codeLab: CodeLabTask;
  project: ProjectTask;
};

export function StageTabs({ sessionId, slides, whiteboardChart, mindmapChart, quiz, codeLab, project }: StageTabsProps) {
  const [active, setActive] = useState<StageTab>("Slides");

  return (
    <div>
      <div className="mb-5 flex flex-wrap gap-2 rounded-full border border-white/10 bg-white/[0.045] p-1">
        {tabs.map((tab) => (
          <button
            key={tab}
            type="button"
            onClick={() => setActive(tab)}
            className={[
              "rounded-full px-4 py-2 text-sm transition",
              active === tab
                ? "bg-gradient-to-r from-sky-400 to-violet-500 text-white shadow-[0_12px_32px_rgba(99,102,241,0.28)]"
                : "text-slate-400 hover:bg-white/[0.06] hover:text-slate-100"
            ].join(" ")}
          >
            {tab}
          </button>
        ))}
      </div>
      <div className="animate-classroom-in">
        {active === "Slides" ? <SlideStage slides={slides} /> : null}
        {active === "Whiteboard" ? <WhiteboardStage chart={whiteboardChart} /> : null}
        {active === "Mindmap" ? <MindmapStage chart={mindmapChart} /> : null}
        {active === "Quiz" ? <QuizStage sessionId={sessionId} questions={quiz} /> : null}
        {active === "Code Lab" ? <CodeLabStage task={codeLab} /> : null}
        {active === "Project" ? <ProjectStage task={project} /> : null}
      </div>
    </div>
  );
}
