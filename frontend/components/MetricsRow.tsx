"use client";

import { TrendingUp, TrendingDown, BarChart2, Target, DollarSign } from "lucide-react";
import { motion } from "framer-motion";
import { Card, CardContent } from "@/components/ui/card";
import { type Metrics } from "@/lib/api";
import { fmt } from "@/lib/utils";
import { useAnimatedNumber } from "@/hooks/useAnimatedNumber";
import { cn } from "@/lib/utils";

// ── StatCard ──────────────────────────────────────────────────────────────────

interface StatCardProps {
  label: string;
  rawValue: number;
  format: (v: number) => string;
  positive?: boolean | null;
  icon: React.ReactNode;
  hint?: string;
  accentColor: "indigo" | "emerald" | "red" | "amber" | "orange";
}

const accentTopBorder: Record<StatCardProps["accentColor"], string> = {
  indigo:  "before:bg-indigo-500",
  emerald: "before:bg-emerald-500",
  red:     "before:bg-red-500",
  amber:   "before:bg-amber-500",
  orange:  "before:bg-orange-500",
};

const accentIconColor: Record<StatCardProps["accentColor"], string> = {
  indigo:  "text-indigo-400",
  emerald: "text-emerald-400",
  red:     "text-red-400",
  amber:   "text-amber-400",
  orange:  "text-orange-400",
};

function StatCard({ label, rawValue, format, positive, icon, hint, accentColor }: StatCardProps) {
  const valueColor =
    positive === true  ? "text-emerald-400" :
    positive === false ? "text-red-400"     :
                         "text-foreground";

  const displayValue = useAnimatedNumber(rawValue, format);

  return (
    <Card
      className={cn(
        "relative overflow-hidden",
        // Top accent bar via pseudo-element using Tailwind
        "before:absolute before:inset-x-0 before:top-0 before:h-0.5 before:rounded-t-xl",
        accentTopBorder[accentColor]
      )}
    >
      <CardContent className="p-4">
        <div className="flex items-center justify-between">
          <span className="text-[11px] font-medium uppercase tracking-widest text-muted-foreground">
            {label}
          </span>
          <span className={cn("opacity-60", accentIconColor[accentColor])}>
            {icon}
          </span>
        </div>
        <motion.p
          className={cn("mt-3 font-mono text-2xl font-semibold tabular-nums leading-none", valueColor)}
          title={hint}
        >
          {displayValue}
        </motion.p>
      </CardContent>
    </Card>
  );
}

// ── MetricsRow ────────────────────────────────────────────────────────────────

export function MetricsRow({ metrics }: { metrics: Metrics }) {
  const tr    = metrics.total_return;
  const bh    = metrics.buy_and_hold_return;
  const alpha = tr - bh;

  return (
    <div className="grid grid-cols-2 gap-3 sm:grid-cols-3 lg:grid-cols-5">
      <StatCard
        label="Total Return"
        rawValue={tr}
        format={(v) => fmt(v, "percent")}
        positive={tr >= 0}
        icon={<TrendingUp className="h-4 w-4" aria-hidden />}
        accentColor={tr >= 0 ? "emerald" : "red"}
      />
      <StatCard
        label="Sharpe Ratio"
        rawValue={metrics.sharpe_ratio}
        format={(v) => fmt(v)}
        positive={metrics.sharpe_ratio >= 1 ? true : metrics.sharpe_ratio < 0 ? false : null}
        icon={<BarChart2 className="h-4 w-4" aria-hidden />}
        hint="Annualised Sharpe (daily steps, 252 periods/yr)"
        accentColor="indigo"
      />
      <StatCard
        label="Max Drawdown"
        rawValue={metrics.max_drawdown}
        format={(v) => fmt(v, "percent")}
        positive={false}
        icon={<TrendingDown className="h-4 w-4" aria-hidden />}
        accentColor="red"
      />
      <StatCard
        label="Win Rate"
        rawValue={metrics.win_rate}
        format={(v) => fmt(v, "percent")}
        positive={metrics.win_rate >= 0.5}
        icon={<Target className="h-4 w-4" aria-hidden />}
        accentColor={metrics.win_rate >= 0.5 ? "emerald" : "amber"}
      />
      <StatCard
        label="Buy & Hold"
        rawValue={bh}
        format={(v) => fmt(v, "percent")}
        positive={bh >= 0}
        icon={<DollarSign className="h-4 w-4" aria-hidden />}
        hint={`Alpha vs B&H: ${fmt(alpha, "percent")}`}
        accentColor="orange"
      />
    </div>
  );
}
