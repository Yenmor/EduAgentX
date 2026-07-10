type EmptyStateProps = {
  title: string;
  description: string;
};

export function EmptyState({ title, description }: EmptyStateProps) {
  return (
    <div className="border border-[var(--line)] bg-white p-8">
      <div className="text-2xl font-light">{title}</div>
      <p className="mt-3 max-w-2xl text-sm leading-6 text-[var(--muted)]">{description}</p>
    </div>
  );
}
