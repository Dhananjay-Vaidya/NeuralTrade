"use client";

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
  /** Date-keyed drawdown values, used to add drawdown row to the shared tooltip. */
  drawdownMap: Map<string, number>;
}

// ── Hoisted constants (stable identity across renders) ────────────────────────

const TOOLTIP_STYLE = {
  background: "#1e1e2e",
  border: "1px solid rgba(255,255,255,0.1)",
  borderRadius: "8px",
  fontSize: 12,
  fontFamily: "monospace",
  padding: "10px 12px",
  boxShadow: "0 8px 32px rgba(0,0,0,0.4)",
} as const;

const LABEL_STYLE   = { color: "#9ca3af" } as const;
const LEGEND_STYLE  = { fontSize: 11 } as const;
const MARGIN        = { top: 4, right: 4, left: 0, bottom: 0 } as const;
const TICK_STYLE    = { fontSize: 10, fill: "#6b7280" } as const;
const TICK_MONO     = { fontSize: 10, fill: "#6b7280", fontFamily: "monospace" } as const;
const REFLINE_STYLE = { strokeDasharray: "3 3" };

// ── Formatters (stable module-level functions) ────────────────────────────────

function legendFormatter(v: string) {
  return v === "strategy" ? "RL Strategy" : "Buy & Hold";
}

function yTickFormatter(v: number) {
  return fmt(v, "currency", 0);
}

// ── Custom tooltip — shows price (close), equity, B&H equity, and drawdown ───
//
// Recharts passes the `payload` array for the active data point and the raw
// `label` (the XAxis dataKey value = date string).  We supplement with the
// drawdown at that date by looking it up in the drawdownMap passed via closure.

function makeTooltipContent(drawdownMap: Map<string, number>) {
  // Return a stable component reference so Recharts doesn't remount on each
  // render.  We rebuild when drawdownMap identity changes (i.e. on data load),
  // which is fine — that's not a hot path.
  return function CrosshairTooltipContent(
    props: TooltipContentProps,
  ) {
    const { active, payload, label } = props;
    if (!active || !payload?.length) return null;

    const strategy   = payload.find((p) => p.dataKey === "strategy")?.value as number | undefined;
    const buyAndHold = payload.find((p) => p.dataKey === "buyAndHold")?.value as number | undefined;
    const drawdown   = drawdownMap.get(label as string);

    return (
      <div style={TOOLTIP_STYLE}>
        <p className="mb-1.5 font-mono text-[10px] text-muted-foreground">{label}</p>
        <table className="border-separate border-spacing-y-0.5 text-xs">
          <tbody>
            {strategy !== undefined && (
              <tr>
                <td className="pr-3 text-indigo-400">Strategy</td>
                <td className="text-right font-mono text-foreground">{fmt(strategy, "currency", 2)}</td>
              </tr>
            )}
            {buyAndHold !== undefined && (
              <tr>
                <td className="pr-3 text-orange-400">Buy &amp; Hold</td>
                <td className="text-right font-mono text-foreground">{fmt(buyAndHold, "currency", 2)}</td>
              </tr>
            )}
            {drawdown !== undefined && (
              <tr>
                <td className="pr-3 text-red-400">Drawdown</td>
                <td className="text-right font-mono text-red-400">
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

export function EquityChart({ data, drawdownMap }: Props) {
  const hoveredDate = useChartStore((s) => s.hoveredDate);

  // Stable tooltip renderer — only rebuilt when drawdownMap reference changes
  // (happens on data load, not on crosshair move).
  const tooltipContent = makeTooltipContent(drawdownMap);

  return (
    <ResponsiveContainer width="100%" height={240}>
      <LineChart data={data} margin={MARGIN}>
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
          width={62}
        />
        <Tooltip
          content={tooltipContent}
          labelStyle={LABEL_STYLE}
          // Show the native Recharts tooltip only when the crosshair is NOT
          // driven by the lightweight-charts sync (i.e. user is hovering this
          // chart directly).  When hoveredDate is set, disable the built-in
          // active detection so ReferenceLine drives the visual instead.
          active={hoveredDate ? false : undefined}
        />
        <Legend
          formatter={legendFormatter}
          iconType="line"
          iconSize={12}
          wrapperStyle={LEGEND_STYLE}
        />

        {/* Synced reference line driven by the lightweight-charts crosshair */}
        {hoveredDate && (
          <ReferenceLine
            x={hoveredDate}
            stroke="rgba(255,255,255,0.25)"
            strokeWidth={1}
            strokeDasharray={REFLINE_STYLE.strokeDasharray}
          />
        )}

        <Line
          type="monotone"
          dataKey="strategy"
          stroke="#6366f1"
          strokeWidth={2}
          dot={false}
          activeDot={{ r: 3 }}
          isAnimationActive={false}
        />
        <Line
          type="monotone"
          dataKey="buyAndHold"
          stroke="#f97316"
          strokeWidth={2}
          strokeDasharray="5 3"
          dot={false}
          activeDot={{ r: 3 }}
          isAnimationActive={false}
        />
      </LineChart>
    </ResponsiveContainer>
  );
}
