import type { QuizType } from "@/lib/learning-data";

type QuizCardProps = {
  quiz: QuizType;
  active?: boolean;
  onSelect?: () => void;
};

export function QuizCard({ quiz, active = false, onSelect }: QuizCardProps) {
  return (
    <button
      onClick={onSelect}
      className={[
        "border p-4 text-left transition-colors",
        active ? "border-[var(--accent)] bg-[var(--accent)] text-white" : "border-[var(--line)] bg-white text-[var(--ink)] hover:border-[var(--accent)]"
      ].join(" ")}
    >
      <div className="font-mono text-xs uppercase tracking-[0.14em] opacity-70">{quiz.fit}</div>
      <h3 className="mt-3 text-2xl font-light">{quiz.title}</h3>
      <p className="mt-3 text-sm leading-6 opacity-80">{quiz.description}</p>
    </button>
  );
}
