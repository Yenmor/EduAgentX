import type { Course } from "./course-api";

export const COURSE_SEED: Course[] = [
  {
    id: "gaodengshuxue",
    title: "高等数学",
    summary:
      "面向大学一年级高等数学学习，从极限、连续、导数到积分与级数，按掌握程度逐步推进。",
    difficulty: "大学基础",
    duration: "6-8 周",
    kb_name: "gaodengshuxue",
    template_book_id: "template_gaodengshuxue",
    outcomes: [
      "建立极限、连续、导数、积分之间的结构化理解",
      "掌握典型计算题的步骤与常见易错点",
      "能把导数、积分和级数方法用于综合题",
    ],
  },
];
