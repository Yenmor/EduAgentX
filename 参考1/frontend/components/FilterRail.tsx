"use client";

type FilterRailProps = {
  categories: string[];
  difficulties: string[];
  goals: string[];
  resourceTypes: string[];
  selected: {
    category: string;
    difficulty: string;
    goal: string;
    resourceType: string;
  };
  onChange: (key: keyof FilterRailProps["selected"], value: string) => void;
};

function SelectRow({
  label,
  value,
  options,
  onChange
}: {
  label: string;
  value: string;
  options: string[];
  onChange: (value: string) => void;
}) {
  return (
    <label className="block border-b border-[var(--line)] py-4 last:border-b-0">
      <span className="mb-2 block font-mono text-xs uppercase tracking-[0.16em] text-[var(--muted)]">{label}</span>
      <select
        value={value}
        onChange={(event) => onChange(event.target.value)}
        className="w-full border border-[var(--line)] bg-white px-3 py-2 text-sm outline-none focus:border-[var(--accent)]"
      >
        {options.map((option) => (
          <option key={option} value={option}>
            {option}
          </option>
        ))}
      </select>
    </label>
  );
}

export function FilterRail({ categories, difficulties, goals, resourceTypes, selected, onChange }: FilterRailProps) {
  return (
    <aside className="border border-[var(--line)] bg-white p-5">
      <div className="border-b border-[var(--line)] pb-4">
        <div className="text-xl font-light">筛选课程</div>
        <p className="mt-2 text-sm leading-6 text-[var(--muted)]">按分类、难度、目标和资源类型缩小范围。</p>
      </div>
      <SelectRow label="分类" value={selected.category} options={["全部", ...categories]} onChange={(value) => onChange("category", value)} />
      <SelectRow label="难度" value={selected.difficulty} options={["全部", ...difficulties]} onChange={(value) => onChange("difficulty", value)} />
      <SelectRow label="目标" value={selected.goal} options={["全部", ...goals]} onChange={(value) => onChange("goal", value)} />
      <SelectRow label="资源" value={selected.resourceType} options={["全部", ...resourceTypes]} onChange={(value) => onChange("resourceType", value)} />
    </aside>
  );
}
