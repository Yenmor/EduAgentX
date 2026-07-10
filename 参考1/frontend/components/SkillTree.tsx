"use client";

import { MermaidRenderer } from "./MermaidRenderer";

type SkillNode = {
  id: string;
  label: string;
  mastery: number;
};

const nodes: SkillNode[] = [
  { id: "prompt", label: "Prompt", mastery: 0.82 },
  { id: "embedding", label: "Embedding", mastery: 0.64 },
  { id: "rag", label: "RAG", mastery: 0.48 },
  { id: "tool", label: "Function Calling", mastery: 0.38 },
  { id: "agent", label: "Agent Workflow", mastery: 0.3 },
  { id: "judge", label: "Judge & Eval", mastery: 0.24 }
];

function nodeStyle(node: SkillNode) {
  if (node.mastery >= 0.75) return ":::mastered";
  if (node.mastery >= 0.45) return ":::learning";
  return ":::locked";
}

export function SkillTree() {
  const chart = `graph TD
  prompt[Prompt Engineering ${Math.round(nodes[0].mastery * 100)}%]${nodeStyle(nodes[0])}
  embedding[Embedding ${Math.round(nodes[1].mastery * 100)}%]${nodeStyle(nodes[1])}
  rag[RAG ${Math.round(nodes[2].mastery * 100)}%]${nodeStyle(nodes[2])}
  tool[Function Calling ${Math.round(nodes[3].mastery * 100)}%]${nodeStyle(nodes[3])}
  agent[Agent Workflow ${Math.round(nodes[4].mastery * 100)}%]${nodeStyle(nodes[4])}
  judge[Judge & Eval ${Math.round(nodes[5].mastery * 100)}%]${nodeStyle(nodes[5])}
  prompt --> embedding --> rag --> tool --> agent --> judge
  classDef mastered fill:#14532d,stroke:#22c55e,color:#e5e7eb
  classDef learning fill:#1e3a8a,stroke:#3b82f6,color:#e5e7eb
  classDef locked fill:#27272a,stroke:#6b7280,color:#9ca3af`;

  return (
    <div className="rounded-lg border border-circuit-line bg-circuit-panel p-5 shadow-glow">
      <MermaidRenderer chart={chart} />
      <div className="mt-4 grid gap-2 sm:grid-cols-3">
        {nodes.map((node) => (
          <div key={node.id} className="rounded-lg border border-circuit-line bg-black/25 p-3">
            <div className="text-sm text-circuit-text">{node.label}</div>
            <div className="mt-2 h-2 rounded bg-black/40">
              <div
                className={node.mastery >= 0.75 ? "h-2 rounded bg-circuit-green" : node.mastery >= 0.45 ? "h-2 rounded bg-circuit-blue" : "h-2 rounded bg-circuit-muted"}
                style={{ width: `${Math.round(node.mastery * 100)}%` }}
              />
            </div>
          </div>
        ))}
      </div>
    </div>
  );
}
