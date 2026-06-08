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
}

const TOOLTIP_STYLE = {
  background: "#1e1e2e",
  border: "1px solid rgba(255,255,255,0.1)",
  borderRadius: "8px",
  fontSize: 12,
  fontFamily: "monospace",
} as const;

const LABEL_STYLE     = { color: "#9ca3af" } as const;
const MARGIN          = { top: 4, right: 4, left: 0, bottom: 0 } as const;
const TICK_STYLE      = { fontSize: 10, fill: "#6b7280" } as const;
const TICK_MONO       = { fontSize: 10, fill: "#6b7280", fontFamily: "monospace" } as const;
const REFLINE_STYLE   = { strokeDasharray: "3 3" };

function tooltipFormatter(v: unknown) {
  return [`${((v as number) * 100).toFixed(2)}%`, "Drawdown"] as [string, string];
}

function yTickFormatter(v: number) {
  return `${(v * 100).toFixed(1)}%`;
}

export function DrawdownChart({ data }: Props) {
  const hoveredDate = useChartStore((s) => s.hoveredDate);

  return (
    <ResponsiveContainer width="100%" height={160}>
      <AreaChart data={data} margin={MARGIN}>
        <defs>
          <linearGradient id="ddGrad" x1="0" y1="0" x2="0" y2="1">
            <stop offset="5%"  stopColor="#ef4444" stopOpacity={0.3} />
            <stop offset="95%" stopColor="#ef4444" stopOpacity={0.02} />
          </linearGradient>
        </defs>
        <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.05)" />
        <XAxis
          dataKey="date"
          tick={TICK_STYLE}
          tickLine={false}
          axisLine={false}
          interval="preserveStartEnd"
        />
        <YAxis
          tick={TICK_MONO}
          tickLine={false}
          axisLine={false}
          tickFormatter={yTickFormatter}
          width={50}
        />
        <Tooltip
          contentStyle={TOOLTIP_STYLE}
          formatter={tooltipFormatter}
          labelStyle={LABEL_STYLE}
          active={hoveredDate ? false : undefined}
        />

        {hoveredDate && (
          <ReferenceLine
            x={hoveredDate}
            stroke="rgba(255,255,255,0.25)"
            strokeWidth={1}
            strokeDasharray={REFLINE_STYLE.strokeDasharray}
          />
        )}

        <Area
          type="monotone"
          dataKey="drawdown"
          stroke="#ef4444"
          strokeWidth={1.5}
          fill="url(#ddGrad)"
          isAnimationActive={false}
        />
      </AreaChart>
    </ResponsiveContainer>
  );
}
