type SafetyBadgeProps = {
  status: "通过" | "复核中" | "已过滤" | "待检查";
};

export function SafetyBadge({ status }: SafetyBadgeProps) {
  const tone =
    status === "通过"
      ? "border-[var(--accent)] text-[var(--accent)]"
      : status === "已过滤"
        ? "border-black bg-black text-white"
        : "border-[var(--line)] text-[var(--muted)]";

  return <span className={`inline-flex border px-2 py-1 font-mono text-xs ${tone}`}>{status}</span>;
}
