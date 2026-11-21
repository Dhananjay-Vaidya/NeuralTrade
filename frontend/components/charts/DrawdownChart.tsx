"use client";

import {
  ResponsiveContainer,
  AreaChart,
  Area,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ReferenceLine,
} from "recharts";
import { useChartStore } from "@/lib/chartStore";
import { type DrawdownPoint } from "@/lib/chartData";

interface Props {
  data: DrawdownPoint[];
  /** Override height (px). Defaults to 160. Pass 240 for side-by-side layout. */
  height?: number;
}

// ── Hoisted style constants ───────────────────────────────────────────────────

const TOOLTIP_STYLE = {
  background: "#0e0e1a",
  border: "1px solid rgba(255,255,255,0.09)",
  borderRadius: "10px",
  fontSize: 12,
  fontFamily: "var(--font-geist-mono, monospace)",
  padding: "8px 12px",
  boxShadow: "0 8px 32px rgba(0,0,0,0.5)",
} as const;

const LABEL_STYLE = { color: "#4b5563", fontSize: 10 } as const;
const MARGIN      = { top: 4, right: 4, left: 0, bottom: 0 } as const;
const TICK_STYLE  = { fontSize: 10, fill: "#4b5563" } as const;
const TICK_MONO   = { fontSize: 10, fill: "#4b5563", fontFamily: "var(--font-geist-mono, monospace)" } as const;

// ── Formatters ────────────────────────────────────────────────────────────────

function tooltipFormatter(v: unknown): [string, string] {
  return [`${((v as number) * 100).toFixed(2)}%`, "Drawdown"];
}

function yTickFormatter(v: number) {
  return `${(v * 100).toFixed(0)}%`;
}

// ── Component ─────────────────────────────────────────────────────────────────

export function DrawdownChart({ data, height = 160 }: Props) {
  const hoveredDate = useChartStore((s) => s.hoveredDate);

  if (data.length === 0) {
    return (
      <div
        style={{ height }}
        className="flex items-center justify-center text-sm text-muted-foreground"
      >
        No drawdown data for this date range.
      </div>
    );
  }

  return (
    <ResponsiveContainer width="100%" height={height}>
      <AreaChart data={data} margin={MARGIN}>
        <defs>
          <linearGradient id="ddGrad" x1="0" y1="0" x2="0" y2="1">
            <stop offset="0%"   stopColor="#ef4444" stopOpacity={0.35} />
            <stop offset="60%"  stopColor="#ef4444" stopOpacity={0.08} />
            <stop offset="100%" stopColor="#ef4444" stopOpacity={0.01} />
          </linearGradient>
        </defs>

        <CartesianGrid
          strokeDasharray="3 3"
          stroke="rgba(255,255,255,0.04)"
          vertical={false}
        />
        <XAxis
          dataKey="date"
          tick={TICK_STYLE}
          tickLine={false}
          axisLine={false}
          interval="preserveStartEnd"
          minTickGap={60}
        />
        <YAxis
          tick={TICK_MONO}
          tickLine={false}
          axisLine={false}
          tickFormatter={yTickFormatter}
          width={44}
        />
        <Tooltip
          contentStyle={TOOLTIP_STYLE}
          formatter={tooltipFormatter}
          labelStyle={LABEL_STYLE}
          active={hoveredDate ? false : undefined}
          cursor={{ stroke: "rgba(255,255,255,0.08)", strokeWidth: 1 }}
        />

        {hoveredDate && (
          <ReferenceLine
            x={hoveredDate}
            stroke="rgba(255,255,255,0.18)"
            strokeWidth={1}
            strokeDasharray="4 3"
          />
        )}

        <Area
          type="monotone"
          dataKey="drawdown"
          stroke="#ef4444"
          strokeWidth={1.5}
          fill="url(#ddGrad)"
          isAnimationActive={false}
          activeDot={{ r: 3, fill: "#ef4444", stroke: "#0e0e1a", strokeWidth: 2 }}
        />
      </AreaChart>
    </ResponsiveContainer>
  );
}
