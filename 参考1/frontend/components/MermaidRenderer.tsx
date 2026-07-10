"use client";

import { useEffect, useId, useMemo, useRef, useState } from "react";

type MermaidRendererProps = {
  chart: string;
};

function normalizeChart(chart: string) {
  const trimmed = chart.trim();
  if (/^(graph|flowchart|mindmap|sequenceDiagram|stateDiagram|classDiagram)/.test(trimmed)) {
    return trimmed;
  }
  return `mindmap\n  root((Learning Map))\n    Concept\n      ${trimmed.slice(0, 48).replace(/[^\w\u4e00-\u9fff ]/g, " ") || "Course idea"}\n    Practice\n      Checkpoint\n    Review\n      Sources`;
}

export function MermaidRenderer({ chart }: MermaidRendererProps) {
  const rawId = useId();
  const elementId = useMemo(() => `mermaid-${rawId.replace(/[^a-zA-Z0-9_-]/g, "")}`, [rawId]);
  const ref = useRef<HTMLDivElement>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    let mounted = true;
    async function render() {
      try {
        const mermaid = (await import("mermaid")).default;
        mermaid.initialize({ startOnLoad: false, theme: "dark", securityLevel: "loose" });
        const { svg } = await mermaid.render(elementId, normalizeChart(chart));
        if (mounted && ref.current) {
          ref.current.innerHTML = svg;
          setError(null);
        }
      } catch (err) {
        if (mounted) setError(err instanceof Error ? err.message : "Mermaid render failed");
      }
    }
    render();
    return () => {
      mounted = false;
    };
  }, [chart, elementId]);

  if (error) {
    return <pre className="overflow-auto rounded-lg border border-circuit-line bg-black/30 p-3 text-xs text-circuit-muted">{normalizeChart(chart)}</pre>;
  }
  return <div ref={ref} className="overflow-hidden rounded-lg border border-circuit-line bg-black/20 p-3" />;
}
