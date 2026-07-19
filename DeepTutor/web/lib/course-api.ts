import { apiFetch, apiUrl } from "./api";

export interface Course {
  id: string;
  title: string;
  summary: string;
  difficulty: string;
  duration: string;
  outcomes: string[];
  kb_name: string;
  template_book_id: string;
}

export interface CourseMessage {
  role: "assistant" | "user";
  content: string;
}

export interface CourseProfile {
  prerequisite: string;
  goal: string;
  pace: string;
  weak_points: string[];
}

export interface OnboardingResult {
  message: string;
  done: boolean;
  profile: CourseProfile | null;
}

export interface CoursePlan {
  skipped: string[];
  focused: string[];
  start_module_id: string;
  template_source: "template" | "starter";
}

export interface BuildPathResult {
  book_id: string;
  course: Course;
  profile: CourseProfile;
  plan: CoursePlan;
}

export async function fetchCourses(): Promise<Course[]> {
  const res = await apiFetch(apiUrl("/api/v1/course/courses"));
  if (!res.ok) throw new Error(`Failed to fetch courses: ${res.status}`);
  const data = (await res.json()) as { courses: Course[] };
  return data.courses;
}

export async function runCourseOnboarding(
  courseId: string,
  sessionId: string,
  history: CourseMessage[],
  answer?: string,
): Promise<OnboardingResult> {
  const res = await apiFetch(
    apiUrl(`/api/v1/course/courses/${encodeURIComponent(courseId)}/onboarding`),
    {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ session_id: sessionId, history, answer }),
    },
  );
  if (!res.ok) throw new Error(`Failed to run onboarding: ${res.status}`);
  return res.json();
}

export async function buildCoursePath(
  courseId: string,
  sessionId: string,
  profile: CourseProfile,
): Promise<BuildPathResult> {
  const res = await apiFetch(
    apiUrl(`/api/v1/course/courses/${encodeURIComponent(courseId)}/build-path`),
    {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ session_id: sessionId, profile }),
    },
  );
  if (!res.ok) throw new Error(`Failed to build path: ${res.status}`);
  return res.json();
}

export async function fetchCoursePathPlan(
  courseId: string,
  bookId: string,
): Promise<BuildPathResult> {
  const res = await apiFetch(
    apiUrl(
      `/api/v1/course/courses/${encodeURIComponent(
        courseId,
      )}/path-plan/${encodeURIComponent(bookId)}`,
    ),
  );
  if (!res.ok) throw new Error(`Failed to fetch path plan: ${res.status}`);
  return res.json();
}
