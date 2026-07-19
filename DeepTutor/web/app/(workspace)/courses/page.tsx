import CoursesClient from "./CoursesClient";
import { COURSE_SEED } from "@/lib/course-seed";

export default function CoursesPage() {
  return <CoursesClient initialCourses={COURSE_SEED} />;
}
