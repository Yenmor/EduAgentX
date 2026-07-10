"use client";

import { FormEvent, useState } from "react";

type Message = {
  id: string;
  sender: "student" | "AI 主讲教师" | "AI 助教" | "AI 学伴";
  content: string;
};

const starterMessages: Message[] = [
  { id: "m1", sender: "AI 主讲教师", content: "我们会围绕 RAG、LangChain 和 AI Agent 三门课程，把知识库问答项目拆成可完成的 14 天路径。" },
  { id: "m2", sender: "AI 助教", content: "如果你对 Embedding 或 Chunking 不熟，我会优先给出代码实验和可视化解释。" },
  { id: "m3", sender: "AI 学伴", content: "完成测验后，我会根据薄弱点提醒你复习对应课程片段。" }
];

export function LearningCompanionChat() {
  const [messages, setMessages] = useState<Message[]>(starterMessages);
  const [input, setInput] = useState("");

  function submit(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    const text = input.trim();
    if (!text) return;
    const studentMessage: Message = { id: `s-${Date.now()}`, sender: "student", content: text };
    const reply: Message = {
      id: `a-${Date.now()}`,
      sender: text.toLowerCase().includes("code") || text.includes("代码") || text.toLowerCase().includes("faiss") ? "AI 助教" : "AI 学伴",
      content: text.toLowerCase().includes("chunk")
        ? "Chunking 决定知识片段粒度。片段太小会丢上下文，太大又会带入噪声；本课堂会通过检索实验让你比较 top-k 命中效果。"
        : "我会把你的问题放回当前课程群主线中：先看课程片段依据，再给出下一步练习建议。"
    };
    setMessages((current) => [...current, studentMessage, reply]);
    setInput("");
  }

  return (
    <div className="flex h-[360px] flex-col rounded-2xl border border-white/10 bg-white/[0.055]">
      <div className="border-b border-white/10 px-4 py-3">
        <div className="text-sm font-semibold text-slate-100">课堂 AI 助手</div>
        <div className="text-xs text-slate-400">主讲教师 / 助教 / 学伴协同答疑</div>
      </div>
      <div className="flex-1 space-y-3 overflow-auto p-4">
        {messages.map((message) => (
          <div key={message.id} className={message.sender === "student" ? "ml-8 rounded-2xl bg-sky-400/15 p-3 text-sm text-sky-50" : "mr-5 rounded-2xl border border-white/10 bg-slate-950/45 p-3"}>
            <div className="mb-1 text-[11px] text-slate-400">{message.sender === "student" ? "你" : message.sender}</div>
            <div className="text-sm leading-5 text-slate-100">{message.content}</div>
          </div>
        ))}
      </div>
      <form onSubmit={submit} className="border-t border-white/10 p-3">
        <div className="flex gap-2">
          <input value={input} onChange={(event) => setInput(event.target.value)} placeholder="向课堂 AI 助手提问..." className="min-w-0 flex-1 rounded-full border border-white/10 bg-slate-950/60 px-4 py-2 text-sm text-slate-100 outline-none placeholder:text-slate-500 focus:border-sky-300/50" />
          <button type="submit" className="rounded-full bg-sky-300 px-4 py-2 text-sm font-semibold text-slate-950">发送</button>
        </div>
      </form>
    </div>
  );
}
