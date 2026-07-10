"use client";

import { FormEvent, useMemo, useRef, useState } from "react";
import { API_BASE } from "@/lib/api";
import type { AgentTraceStep, ChatResponse, OrchestratorResource } from "@/lib/types";
import { AgentTrace } from "./AgentTrace";
import { ResourceCard } from "./ResourceCard";
import { MermaidRenderer } from "./MermaidRenderer";

type Message = {
  role: "student" | "agent";
  content: string;
};

type SseEvent = {
  event: string;
  data: Record<string, unknown>;
};

function isResourceArray(value: ChatResponse["result"]): value is OrchestratorResource[] {
  return Array.isArray(value);
}

function parseSseBlock(block: string): SseEvent | null {
  let event = "message";
  const dataLines: string[] = [];
  for (const line of block.split("\n")) {
    if (line.startsWith("event:")) {
      event = line.slice("event:".length).trim();
    } else if (line.startsWith("data:")) {
      dataLines.push(line.slice("data:".length).trimStart());
    }
  }
  if (!dataLines.length) return null;
  try {
    return { event, data: JSON.parse(dataLines.join("\n")) as Record<string, unknown> };
  } catch {
    return { event, data: { raw: dataLines.join("\n") } };
  }
}

function appendTrace(trace: AgentTraceStep[], step: AgentTraceStep): AgentTraceStep[] {
  const duplicateIndex = trace.findIndex((item) => item.agent === step.agent && item.action === step.action && item.status === "running");
  if (duplicateIndex >= 0) {
    return trace.map((item, index) => (index === duplicateIndex ? step : item));
  }
  return [...trace, step];
}

export function ChatPanel() {
  const [messages, setMessages] = useState<Message[]>([
    { role: "agent", content: "EduAgentX online. Knowledge index, agents, and Judge workflow are ready." }
  ]);
  const [input, setInput] = useState("我不懂 RAG，帮我生成学习资料");
  const [latest, setLatest] = useState<ChatResponse | null>(null);
  const [streamTrace, setStreamTrace] = useState<AgentTraceStep[]>([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const abortRef = useRef<AbortController | null>(null);

  const tutorResult = useMemo(() => {
    if (!latest || Array.isArray(latest.result)) return null;
    return latest.result as Record<string, unknown>;
  }, [latest]);

  function updateAgentMessage(updater: (content: string) => string) {
    setMessages((prev) => {
      const next = [...prev];
      const last = next[next.length - 1];
      if (last?.role === "agent") {
        next[next.length - 1] = { ...last, content: updater(last.content) };
        return next;
      }
      return [...next, { role: "agent", content: updater("") }];
    });
  }

  function applyTracePayload(data: Record<string, unknown>) {
    if (Array.isArray(data.agent_trace)) {
      setStreamTrace(data.agent_trace as AgentTraceStep[]);
      return;
    }
    if (data.trace_step) {
      setStreamTrace((prev) => appendTrace(prev, data.trace_step as AgentTraceStep));
    }
  }

  function handleSseEvent(event: SseEvent) {
    const { data } = event;
    applyTracePayload(data);

    if (event.event === "retrieve_start") {
      setStreamTrace((prev) =>
        appendTrace(prev, {
          agent: "RetrieverAgent",
          agent_name: "RetrieverAgent",
          action: "retrieve_start",
          status: "running",
          output: { query: data.query }
        })
      );
      return;
    }

    if (event.event === "generate_start") {
      const agent = String(data.agent ?? "GeneratorAgent");
      setStreamTrace((prev) =>
        appendTrace(prev, {
          agent,
          agent_name: agent,
          action: data.rewrite ? "rewrite_start" : "generate_start",
          status: "running",
          output: { intent: data.intent, instruction: data.instruction }
        })
      );
      return;
    }

    if (event.event === "judge_start") {
      setStreamTrace((prev) =>
        appendTrace(prev, {
          agent: "JudgeAgent",
          agent_name: "JudgeAgent",
          action: "judge_start",
          status: "running",
          output: { rewrite: data.rewrite ?? false }
        })
      );
      return;
    }

    if (event.event === "token") {
      updateAgentMessage((content) => content + String(data.content ?? ""));
      return;
    }

    if (event.event === "message") {
      if (typeof data.token === "string") {
        updateAgentMessage((content) => `${content}${content ? " " : ""}${data.token}`);
        return;
      }
      if (data.done && data.response) {
        const finalPayload = data.response as ChatResponse;
        setLatest(finalPayload);
        setStreamTrace(finalPayload.agent_trace ?? []);
        updateAgentMessage(() => finalPayload.reply);
        return;
      }
    }

    if (event.event === "final") {
      const finalPayload = (data.response ?? data) as ChatResponse;
      setLatest(finalPayload);
      setStreamTrace(finalPayload.agent_trace ?? []);
      updateAgentMessage(() => finalPayload.reply);
      return;
    }

    if (event.event === "error") {
      throw new Error(String(data.message ?? "Stream failed"));
    }
  }

  async function submit(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    const studentMessage = input.trim();
    if (!studentMessage || loading) return;

    const controller = new AbortController();
    abortRef.current = controller;
    setInput("");
    setLoading(true);
    setLatest(null);
    setStreamTrace([]);
    setError(null);
    setMessages((prev) => [...prev, { role: "student", content: studentMessage }, { role: "agent", content: "" }]);

    try {
      const response = await fetch(`${API_BASE}/api/chat/stream`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          student_id: "demo-student",
          message: studentMessage,
          mode: "auto"
        }),
        signal: controller.signal
      });
      if (!response.ok) {
        const text = await response.text();
        throw new Error(text || `Request failed: ${response.status}`);
      }
      if (!response.body) {
        throw new Error("Browser did not provide a readable stream.");
      }

      const reader = response.body.getReader();
      const decoder = new TextDecoder();
      let buffer = "";
      while (true) {
        const { value, done } = await reader.read();
        if (done) break;
        buffer += decoder.decode(value, { stream: true }).replace(/\r\n/g, "\n");
        const blocks = buffer.split("\n\n");
        buffer = blocks.pop() ?? "";
        for (const block of blocks) {
          const parsed = parseSseBlock(block);
          if (parsed) handleSseEvent(parsed);
        }
      }
      if (buffer.trim()) {
        const parsed = parseSseBlock(buffer);
        if (parsed) handleSseEvent(parsed);
      }
    } catch (err) {
      if (err instanceof DOMException && err.name === "AbortError") {
        setError("已中断本次流式生成。");
        updateAgentMessage((content) => content || "已中断本次流式生成。");
      } else {
        const message = err instanceof Error ? err.message : "Request failed";
        setError(`无法连接后端或流式接口异常：${message}`);
        updateAgentMessage((content) => content || "后端暂时不可用，请确认 FastAPI 已在 8000 端口启动。");
      }
    } finally {
      setLoading(false);
      abortRef.current = null;
    }
  }

  function stopStream() {
    abortRef.current?.abort();
  }

  const visibleTrace = latest?.agent_trace ?? streamTrace;

  return (
    <div className="grid gap-4 xl:grid-cols-[minmax(0,1fr)_360px]">
      <section className="grid min-h-[calc(100vh-8rem)] grid-rows-[1fr_auto] gap-4">
        <div className="space-y-3 overflow-y-auto rounded-lg border border-circuit-line bg-circuit-panel p-4 shadow-glow">
          {messages.map((message, index) => (
            <div
              key={`${message.role}-${index}`}
              className={
                message.role === "student"
                  ? "ml-auto max-w-2xl rounded-lg border border-circuit-blue/40 bg-circuit-blue/10 p-3"
                  : "max-w-2xl rounded-lg border border-circuit-line bg-black/30 p-3"
              }
            >
              <div className="font-mono text-xs uppercase tracking-[0.14em] text-circuit-muted">{message.role}</div>
              <div className="mt-1 whitespace-pre-wrap text-sm leading-6 text-circuit-text">
                {message.content || (loading && message.role === "agent" ? "..." : "")}
              </div>
            </div>
          ))}
          {error ? <div className="rounded-lg border border-red-500/40 bg-red-500/10 p-3 text-sm text-red-200">{error}</div> : null}
        </div>
        <form onSubmit={submit} className="flex gap-3">
          <input
            value={input}
            onChange={(event) => setInput(event.target.value)}
            className="min-w-0 flex-1 rounded-lg border border-circuit-line bg-circuit-panel px-4 py-3 text-sm text-circuit-text outline-none focus:border-circuit-blue"
            placeholder="Ask EduAgentX..."
          />
          {loading ? (
            <button
              type="button"
              onClick={stopStream}
              className="rounded-lg border border-red-400/40 px-5 py-3 text-sm font-semibold text-red-200 transition hover:bg-red-500/10"
            >
              Stop
            </button>
          ) : null}
          <button
            type="submit"
            disabled={loading}
            className="rounded-lg bg-circuit-blue px-5 py-3 text-sm font-semibold text-white transition hover:bg-blue-500 disabled:cursor-not-allowed disabled:opacity-60"
          >
            {loading ? "Streaming" : "Send"}
          </button>
        </form>
      </section>

      <aside className="space-y-4">
        <AgentTrace trace={visibleTrace} />
        {latest?.judge ? (
          <div className="rounded-lg border border-circuit-line bg-circuit-panel p-4">
            <div className="font-mono text-xs uppercase tracking-[0.18em] text-circuit-green">Judge</div>
            <div className="mt-3 text-3xl font-semibold text-circuit-text">{latest.judge.score}</div>
            <div className="mt-2 text-sm text-circuit-muted">{latest.judge.pass ? "Passed" : "Rewritten once"}</div>
          </div>
        ) : null}
      </aside>

      {latest && isResourceArray(latest.result) ? (
        <section className="grid gap-4 xl:col-span-2 md:grid-cols-2">
          {latest.result.map((resource) => (
            <ResourceCard key={`${resource.type}-${resource.title}`} resource={resource} />
          ))}
        </section>
      ) : null}

      {tutorResult?.mermaid_graph ? (
        <section className="xl:col-span-2">
          <MermaidRenderer chart={String(tutorResult.mermaid_graph)} />
        </section>
      ) : null}
    </div>
  );
}
