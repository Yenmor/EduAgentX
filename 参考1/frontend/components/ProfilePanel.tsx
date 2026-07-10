import type { StudentProfile } from "@/lib/types";

type ProfilePanelProps = {
  profile: StudentProfile | null;
};

export function ProfilePanel({ profile }: ProfilePanelProps) {
  return (
    <div className="rounded-lg border border-circuit-line bg-circuit-panel p-5 shadow-glow">
      <div className="font-mono text-xs uppercase tracking-[0.18em] text-circuit-green">Student Profile JSON</div>
      <pre className="mt-4 overflow-auto rounded-lg border border-circuit-line bg-black/30 p-4 text-xs leading-5 text-circuit-text">
        {JSON.stringify(profile ?? { status: "loading" }, null, 2)}
      </pre>
    </div>
  );
}
