"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";

const navItems = [
  { href: "/courses", label: "课程" },
  { href: "/studio", label: "互动课堂" },
  { href: "/roadmap", label: "路径规划" },
  { href: "/quiz", label: "测验" },
  { href: "/progress", label: "学习评分" }
];

export function TopNav() {
  const pathname = usePathname();

  return (
    <header className="sticky top-0 z-50 border-b border-[var(--line)] bg-[var(--paper)]">
      <div className="mx-auto grid max-w-7xl grid-cols-1 gap-4 px-4 py-4 md:grid-cols-[220px_1fr] md:px-6">
        <Link href="/courses" className="text-lg font-semibold tracking-tight text-[var(--ink)]">
          学习平台
        </Link>
        <nav className="flex gap-2 overflow-x-auto md:justify-end">
          {navItems.map((item) => {
            const active = pathname === item.href || (item.href !== "/courses" && pathname.startsWith(item.href));
            return (
              <Link
                key={item.href}
                href={item.href}
                className={[
                  "border border-[var(--line)] px-4 py-2 text-sm font-medium transition-colors",
                  active ? "bg-[var(--accent)] text-white" : "bg-white text-[var(--ink)] hover:border-[var(--accent)]"
                ].join(" ")}
              >
                {item.label}
              </Link>
            );
          })}
        </nav>
      </div>
    </header>
  );
}
