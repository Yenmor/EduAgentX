type PageHeaderProps = {
  eyebrow?: string;
  title: string;
  description: string;
  aside?: React.ReactNode;
};

export function PageHeader({ eyebrow, title, description, aside }: PageHeaderProps) {
  return (
    <section className="grid min-w-0 gap-6 border-b border-[var(--line)] pb-8 md:grid-cols-[minmax(0,1fr)_320px] md:items-end">
      <div className="min-w-0">
        {eyebrow ? <div className="mb-3 font-mono text-xs uppercase tracking-[0.18em] text-[var(--muted)]">{eyebrow}</div> : null}
        <h1
          className="w-full max-w-[22rem] text-3xl font-extralight leading-tight tracking-normal text-[var(--ink)] sm:max-w-full sm:text-5xl md:max-w-4xl md:text-7xl md:leading-none"
          style={{ overflowWrap: "anywhere", wordBreak: "break-all" }}
        >
          {title}
        </h1>
        <p className="mt-5 max-w-[22rem] text-base leading-7 text-[var(--muted)] sm:max-w-3xl md:text-lg" style={{ overflowWrap: "anywhere", wordBreak: "break-all" }}>
          {description}
        </p>
      </div>
      {aside ? <div className="border-l-4 border-[var(--accent)] pl-5">{aside}</div> : null}
    </section>
  );
}
