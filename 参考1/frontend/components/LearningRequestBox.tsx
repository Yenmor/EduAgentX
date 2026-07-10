"use client";

import { useState } from "react";

type LearningRequestBoxProps = {
  title?: string;
  placeholder: string;
  buttonLabel: string;
  defaultValue?: string;
  onSubmit?: (value: string) => void;
};

export function LearningRequestBox({
  title = "自然语言学习需求",
  placeholder,
  buttonLabel,
  defaultValue = "",
  onSubmit
}: LearningRequestBoxProps) {
  const [value, setValue] = useState(defaultValue);
  const [submitted, setSubmitted] = useState(defaultValue);

  function handleSubmit() {
    const next = value.trim();
    setSubmitted(next);
    onSubmit?.(next);
  }

  return (
    <div className="border border-[var(--line)] bg-white p-5">
      <div className="flex flex-wrap items-center justify-between gap-3 border-b border-[var(--line)] pb-3">
        <h2 className="text-lg font-medium">{title}</h2>
        <span className="w-full break-all font-mono text-xs text-[var(--muted)] sm:w-auto">POST /api/profile/chat</span>
      </div>
      <textarea
        value={value}
        onChange={(event) => setValue(event.target.value)}
        placeholder={placeholder}
        className="mt-4 min-h-28 w-full max-w-[22rem] resize-y border border-[var(--line)] bg-[var(--soft)] p-4 text-sm leading-6 outline-none focus:border-[var(--accent)] sm:max-w-none"
        style={{ overflowWrap: "anywhere" }}
      />
      <div className="mt-4 flex flex-wrap items-center gap-3">
        <button onClick={handleSubmit} className="bg-[var(--accent)] px-5 py-2.5 text-sm font-semibold text-white">
          {buttonLabel}
        </button>
        {submitted ? <span className="text-sm text-[var(--muted)]">已根据需求更新推荐结果</span> : null}
      </div>
    </div>
  );
}
