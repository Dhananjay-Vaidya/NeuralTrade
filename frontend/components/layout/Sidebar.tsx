"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";
import { BarChart2, Activity, Info } from "lucide-react";
import { motion, AnimatePresence } from "framer-motion";
import { cn } from "@/lib/utils";

const navItems = [
  { href: "/",        label: "Dashboard", icon: BarChart2 },
  { href: "/signals", label: "Signals",   icon: Activity  },
  { href: "/about",   label: "About",     icon: Info       },
];

export function Sidebar() {
  const pathname = usePathname();

  return (
    <aside className="hidden md:flex h-full w-56 shrink-0 flex-col border-r border-white/[0.07] bg-surface">
      {/* Logo */}
      <div className="flex h-14 items-center border-b border-white/[0.07] px-5">
        <div className="flex flex-col">
          <span className="font-mono text-[15px] font-bold tracking-tight text-foreground leading-none">
            Neural<span className="text-indigo-400">Trade</span>
          </span>
          <span className="mt-0.5 text-[9px] font-medium uppercase tracking-[0.18em] text-muted-foreground/70">
            Research Tool
          </span>
        </div>
        {/* Subtle accent dot */}
        <span
          className="ml-auto h-1.5 w-1.5 rounded-full bg-indigo-400 glow-dot"
          aria-hidden
        />
      </div>

      {/* Nav */}
      <nav className="flex flex-col gap-0.5 p-2.5 flex-1" aria-label="Main navigation">
        {navItems.map(({ href, label, icon: Icon }) => {
          const active = pathname === href;
          return (
            <Link
              key={href}
              href={href}
              className={cn(
                "group relative flex items-center gap-3 rounded-lg px-3 py-2 text-sm font-medium transition-colors duration-150",
                active
                  ? "text-indigo-300"
                  : "text-muted-foreground hover:text-foreground"
              )}
            >
              {/* Animated background pill */}
              {active && (
                <motion.span
                  layoutId="sidebar-active-bg"
                  className="absolute inset-0 rounded-lg bg-indigo-500/10 border border-indigo-500/20"
                  transition={{ duration: 0.2, ease: [0, 0, 0.2, 1] }}
                />
              )}
              {/* Hover bg (only when not active) */}
              {!active && (
                <span className="absolute inset-0 rounded-lg opacity-0 group-hover:opacity-100 bg-white/[0.04] transition-opacity duration-150" />
              )}

              <Icon
                className={cn(
                  "relative z-10 h-4 w-4 shrink-0 transition-colors duration-150",
                  active ? "text-indigo-400" : "text-muted-foreground group-hover:text-foreground"
                )}
                aria-hidden
              />
              <span className="relative z-10">{label}</span>

              {/* Active left accent bar */}
              <AnimatePresence>
                {active && (
                  <motion.span
                    key="bar"
                    className="absolute left-0 top-1/2 h-4 w-0.5 -translate-y-1/2 rounded-r-full bg-indigo-400"
                    initial={{ scaleY: 0, opacity: 0 }}
                    animate={{ scaleY: 1, opacity: 1 }}
                    exit={{ scaleY: 0, opacity: 0 }}
                    transition={{ duration: 0.2, ease: [0, 0, 0.2, 1] }}
                  />
                )}
              </AnimatePresence>
            </Link>
          );
        })}
      </nav>

      {/* Bottom disclaimer */}
      <div className="p-2.5">
        <div className="rounded-lg border border-amber-500/15 bg-amber-500/5 p-3 text-[10px] leading-relaxed text-amber-400/70">
          Research &amp; backtesting only. Not financial advice.
        </div>
      </div>
    </aside>
  );
}
