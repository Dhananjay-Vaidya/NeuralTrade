"use client";

import { useMemo } from "react";
import {
  ResponsiveContainer,
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
  ReferenceLine,
  type TooltipContentProps,
} from "recharts";
import { fmt } from "@/lib/utils";
import { useChartStore } from "@/lib/chartStore";
import { type EquityPoint } from "@/lib/chartData";

interface Props {
  data: EquityPoint[];
  drawdownMap: Map<string, number>;
  /** Override height (px). Defaults to 240. */
  height?: number;
}

// ── Hoisted style constants ───────────────────────────────────────────────────

const TOOLTIP_STYLE = {
  background: "#0e0e1a",
  border: "1px solid rgba(255,255,255,0.09)",
  borderRadius: "10px",
  fontSize: 12,
  fontFamily: "var(--font-geist-mono, monospace)",
  padding: "10px 13px",
  boxShadow: "0 8px 32px rgba(0,0,0,0.5)",
} as const;

const LABEL_STYLE   = { color: "#4b5563", fontSize: 10 } as const;
const LEGEND_STYLE  = { fontSize: 11, paddingTop: 4 } as const;
const MARGIN        = { top: 4, right: 4, left: 0, bottom: 0 } as const;
const TICK_STYLE    = { fontSize: 10, fill: "#4b5563" } as const;
const TICK_MONO     = { fontSize: 10, fill: "#4b5563", fontFamily: "var(--font-geist-mono, monospace)" } as const;

// ── Formatters ────────────────────────────────────────────────────────────────

function legendFormatter(v: string) {
  return v === "strategy" ? "RL Strategy" : "Buy & Hold";
}

function yTickFormatter(v: number) {
  return fmt(v, "currency", 0);
}

// ── Custom tooltip ─────────────────────────────────────────────────────────────

function makeTooltipContent(drawdownMap: Map<string, number>) {
  return function CrosshairTooltipContent(props: TooltipContentProps) {
    const { active, payload, label } = props;
    if (!active || !payload?.length) return null;

    const strategy   = payload.find((p) => p.dataKey === "strategy")?.value as number | undefined;
    const buyAndHold = payload.find((p) => p.dataKey === "buyAndHold")?.value as number | undefined;
    const drawdown   = drawdownMap.get(label as string);

    return (
      <div style={TOOLTIP_STYLE}>
        <p style={{ color: "#6b7280", fontSize: 10, marginBottom: 8, letterSpacing: "0.05em" }}>
          {label}
        </p>
        <table style={{ borderCollapse: "separate", borderSpacing: "0 3px" }}>
          <tbody>
            {strategy !== undefined && (
              <tr>
                <td style={{ paddingRight: 12, color: "#818cf8" }}>Strategy</td>
                <td style={{ textAlign: "right", color: "#e4e8f0", fontWeight: 600 }}>
                  {fmt(strategy, "currency", 2)}
                </td>
              </tr>
            )}
            {buyAndHold !== undefined && (
              <tr>
                <td style={{ paddingRight: 12, color: "#fb923c" }}>Buy &amp; Hold</td>
                <td style={{ textAlign: "right", color: "#e4e8f0", fontWeight: 600 }}>
                  {fmt(buyAndHold, "currency", 2)}
                </td>
              </tr>
            )}
            {drawdown !== undefined && (
              <tr>
                <td style={{ paddingRight: 12, color: "#f87171" }}>Drawdown</td>
                <td style={{ textAlign: "right", color: "#f87171", fontWeight: 600 }}>
                  {(drawdown * 100).toFixed(2)}%
                </td>
              </tr>
            )}
          </tbody>
        </table>
      </div>
    );
  };
}

// ── Component ─────────────────────────────────────────────────────────────────

export function EquityChart({ data, drawdownMap, height = 240 }: Props) {
  const hoveredDate = useChartStore((s) => s.hoveredDate);

  if (data.length === 0) {
    return (
      <div
        style={{ height }}
        className="flex items-center justify-center text-sm text-muted-foreground"
      >
        No equity data for this date range.
      </div>
    );
  }

  const tooltipContent = useMemo(() => makeTooltipContent(drawdownMap), [drawdownMap]);

  return (
    <ResponsiveContainer width="100%" height={height}>
      <LineChart data={data} margin={MARGIN}>
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
          width={64}
        />
        <Tooltip
          content={tooltipContent}
          labelStyle={LABEL_STYLE}
          active={hoveredDate ? false : undefined}
          cursor={{ stroke: "rgba(255,255,255,0.08)", strokeWidth: 1 }}
        />
        <Legend
          formatter={legendFormatter}
          iconType="line"
          iconSize={12}
          wrapperStyle={LEGEND_STYLE}
        />

        {hoveredDate && (
          <ReferenceLine
            x={hoveredDate}
            stroke="rgba(255,255,255,0.18)"
            strokeWidth={1}
            strokeDasharray="4 3"
          />
        )}

        <Line
          type="monotone"
          dataKey="strategy"
          stroke="#6366f1"
          strokeWidth={2}
          dot={false}
          activeDot={{ r: 4, fill: "#6366f1", stroke: "#0e0e1a", strokeWidth: 2 }}
          isAnimationActive={false}
        />
        <Line
          type="monotone"
          dataKey="buyAndHold"
          stroke="#f97316"
          strokeWidth={1.5}
          strokeDasharray="5 3"
          dot={false}
          activeDot={{ r: 4, fill: "#f97316", stroke: "#0e0e1a", strokeWidth: 2 }}
          isAnimationActive={false}
        />
      </LineChart>
    </ResponsiveContainer>
  );
}
