import { SafetyBadge } from "@/components/SafetyBadge";

type DisplayResource = {
  title: string;
  type: string;
  description?: string;
  content?: string;
  status?: "已生成" | "可生成" | "待复核";
};

export function ResourceCard({ resource }: { resource: DisplayResource }) {
  const safety = resource.status === "已生成" ? "通过" : resource.status === "待复核" ? "复核中" : "待检查";

  return (
    <article className="border border-[var(--line)] bg-white p-4">
      <div className="flex items-start justify-between gap-4">
        <span className="border border-[var(--line)] px-2 py-1 font-mono text-xs">{resource.type}</span>
        <SafetyBadge status={safety} />
      </div>
      <h3 className="mt-4 text-xl font-light leading-tight">{resource.title}</h3>
      <p className="mt-3 text-sm leading-6 text-[var(--muted)]">{resource.description ?? resource.content ?? "已生成个性化学习资源，等待后端接入真实内容。"}</p>
      <button className="mt-5 border border-[var(--accent)] px-3 py-2 text-sm font-medium text-[var(--accent)]">
        {resource.status === "可生成" ? "生成资源" : "查看资源"}
      </button>
    </article>
  );
}
