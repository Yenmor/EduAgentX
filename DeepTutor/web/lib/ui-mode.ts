"use client";

import { useEffect, useState } from "react";

export type UiMode = "student" | "author";

export const UI_MODE_STORAGE_KEY = "eduagentx.uiMode";

export function defaultUiMode(): UiMode {
  return process.env.NEXT_PUBLIC_DEFAULT_UI_MODE === "author"
    ? "author"
    : "student";
}

function normalizeUiMode(value: string | null | undefined): UiMode {
  return value === "author" || value === "student" ? value : defaultUiMode();
}

export function useUiMode() {
  const [mode, setModeState] = useState<UiMode>(defaultUiMode);

  useEffect(() => {
    if (typeof window === "undefined") return;
    const id = window.setTimeout(() => {
      setModeState(normalizeUiMode(window.localStorage.getItem(UI_MODE_STORAGE_KEY)));
    }, 0);
    return () => window.clearTimeout(id);
  }, []);

  const setMode = (next: UiMode) => {
    setModeState(next);
    if (typeof window !== "undefined") {
      window.localStorage.setItem(UI_MODE_STORAGE_KEY, next);
    }
  };

  return { mode, setMode };
}
