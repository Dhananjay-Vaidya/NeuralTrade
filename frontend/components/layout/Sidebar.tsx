"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";
import { BarChart2, Activity, Info } from "lucide-react";
import { cn } from "@/lib/utils";

const navItems = [
  { href: "/", label: "Dashboard", icon: BarChart2 },
  { href: "/signals", label: "Signals", icon: Activity },
  { href: "/about", label: "About", icon: Info },
];

export function Sidebar() {
  const pathname = usePathname();

  return (
    <aside className="hidden md:flex h-full w-56 flex-col border-r border-white/10 bg-surface px-3 py-4">
      {/* Logo */}
      <div className="mb-6 px-2">
        <span className="font-mono text-lg font-bold tracking-tight text-foreground">
          Neural<span className="text-indigo-400">Trade</span>
        </span>
        <p className="mt-0.5 text-[10px] text-muted-foreground uppercase tracking-widest">Research Tool</p>
      </div>

      {/* Nav */}
      <nav className="flex flex-col gap-0.5" aria-label="Main navigation">
        {navItems.map(({ href, label, icon: Icon }) => (
          <Link
            key={href}
            href={href}
            className={cn(
              "flex items-center gap-3 rounded-lg px-3 py-2 text-sm transition-colors",
              pathname === href
                ? "bg-indigo-600/20 text-indigo-300 font-medium"
                : "text-muted-foreground hover:bg-white/5 hover:text-foreground"
            )}
          >
            <Icon className="h-4 w-4 shrink-0" aria-hidden />
            {label}
          </Link>
        ))}
      </nav>

      {/* Disclaimer at bottom */}
      <div className="mt-auto rounded-lg border border-amber-500/20 bg-amber-500/5 p-3 text-[10px] leading-relaxed text-amber-400/80">
        ⚠️ Research &amp; backtesting only. Not financial advice.
      </div>
    </aside>
  );
}
