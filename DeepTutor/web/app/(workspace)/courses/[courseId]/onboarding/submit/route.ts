import { NextRequest, NextResponse } from "next/server";

const API_BASE_URL = process.env.DEEPTUTOR_API_BASE_URL ?? "http://127.0.0.1:8001";

function safeSessionId(courseId: string) {
  const suffix = `${Date.now()}_${Math.random().toString(16).slice(2)}`;
  return `${courseId}_${suffix.replace(/[^0-9A-Za-z_-]/g, "_")}`.slice(0, 120);
}

export async function POST(
  request: NextRequest,
  { params }: { params: Promise<{ courseId: string }> },
) {
  const { courseId } = await params;
  const form = await request.formData();
  const profile = {
    prerequisite: String(form.get("prerequisite") || ""),
    goal: String(form.get("goal") || ""),
    pace: String(form.get("pace") || ""),
    weak_points: String(form.get("weak_points") || "")
      .split(/[、,，;；\s]+/)
      .map((item) => item.trim())
      .filter(Boolean),
  };
  const sessionId = safeSessionId(courseId);

  const res = await fetch(
    `${API_BASE_URL}/api/v1/course/courses/${encodeURIComponent(courseId)}/build-path`,
    {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ session_id: sessionId, profile }),
      cache: "no-store",
    },
  );
  if (!res.ok) {
    return NextResponse.redirect(
      new URL(`/courses/${encodeURIComponent(courseId)}/onboarding`, request.url),
      303,
    );
  }
  const data = (await res.json()) as { book_id: string };
  return NextResponse.redirect(
    new URL(
      `/courses/${encodeURIComponent(courseId)}/preview?book_id=${encodeURIComponent(
        data.book_id,
      )}`,
      request.url,
    ),
    303,
  );
}
