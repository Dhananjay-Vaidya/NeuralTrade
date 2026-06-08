"use client";

import { TrendingUp, TrendingDown, BarChart, Target, DollarSign } from "lucide-react";
import { motion } from "framer-motion";
import { Card, CardContent, CardTitle } from "@/components/ui/card";
import { type Metrics } from "@/lib/api";
import { fmt } from "@/lib/utils";
import { useAnimatedNumber } from "@/hooks/useAnimatedNumber";

// ── StatCard ──────────────────────────────────────────────────────────────────

interface StatCardProps {
  label: string;
  /** Raw numeric value — animated via useAnimatedNumber internally. */
  rawValue: number;
  format: (v: number) => string;
  positive?: boolean | null;
  icon: React.ReactNode;
  hint?: string;
}

function StatCard({ label, rawValue, format, positive, icon, hint }: StatCardProps) {
  const color =
    positive === true  ? "text-emerald-400" :
    positive === false ? "text-red-400"     :
                         "text-foreground";

  const displayValue = useAnimatedNumber(rawValue, format);

  return (
    <Card>
      <CardContent className="p-4">
        <div className="flex items-center justify-between">
          <CardTitle className="text-xs">{label}</CardTitle>
          <span className="text-muted-foreground">{icon}</span>
        </div>
        <motion.p
          className={`mt-2 font-mono text-2xl font-semibold tabular-nums ${color}`}
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
      />
      <StatCard
        label="Sharpe Ratio"
        rawValue={metrics.sharpe_ratio}
        format={(v) => fmt(v)}
        positive={metrics.sharpe_ratio >= 1 ? true : metrics.sharpe_ratio < 0 ? false : null}
        icon={<BarChart className="h-4 w-4" aria-hidden />}
        hint="Annualised Sharpe (daily steps, 252 periods/yr)"
      />
      <StatCard
        label="Max Drawdown"
        rawValue={metrics.max_drawdown}
        format={(v) => fmt(v, "percent")}
        positive={false}
        icon={<TrendingDown className="h-4 w-4" aria-hidden />}
      />
      <StatCard
        label="Win Rate"
        rawValue={metrics.win_rate}
        format={(v) => fmt(v, "percent")}
        positive={metrics.win_rate >= 0.5}
        icon={<Target className="h-4 w-4" aria-hidden />}
      />
      <StatCard
        label="Buy & Hold"
        rawValue={bh}
        format={(v) => fmt(v, "percent")}
        positive={bh >= 0}
        icon={<DollarSign className="h-4 w-4" aria-hidden />}
        hint={`Alpha vs B&H: ${fmt(alpha, "percent")}`}
      />
    </div>
  );
}
