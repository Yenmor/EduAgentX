type ProfileChipBarProps = {
  chips: string[];
};

export function ProfileChipBar({ chips }: ProfileChipBarProps) {
  return (
    <div className="flex flex-wrap gap-2">
      {chips.map((chip) => (
        <span key={chip} className="rounded-full border border-white/10 bg-white/[0.06] px-3 py-1 text-xs text-slate-200 shadow-[inset_0_1px_0_rgba(255,255,255,0.08)]">
          {chip}
        </span>
      ))}
    </div>
  );
}
