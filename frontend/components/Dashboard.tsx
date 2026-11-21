"use client";

import { useMemo } from "react";
import dynamic from "next/dynamic";
import { motion } from "framer-motion";
import { TrendingUp, BarChart2, TrendingDown, List } from "lucide-react";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { MetricsRow } from "@/components/MetricsRow";
import { DrawdownChart } from "@/components/charts/DrawdownChart";
import { EquityChart } from "@/components/charts/EquityChart";
import { TradeLog } from "@/components/TradeLog";
import { PanelErrorBoundary } from "@/components/PanelErrorBoundary";
import { type BacktestResponse } from "@/lib/api";
import {
  buildEquityData,
  buildDrawdownData,
  buildDrawdownMap,
} from "@/lib/chartData";
import {
  staggerContainerVariants,
  slideUpVariants,
} from "@/lib/motion";

const CandlestickChart = dynamic(
  () => import("@/components/charts/CandlestickChart").then((m) => m.CandlestickChart),
  { ssr: false, loading: () => <div className="h-95 w-full shimmer rounded-lg" /> },
);

interface Props {
  data: BacktestResponse;
}

const VIEWPORT = { once: true };

function PanelHeader({
  icon,
  title,
  sub,
}: {
  icon: React.ReactNode;
  title: string;
  sub?: string;
}) {
  return (
    <CardHeader>
      <div className="flex items-center gap-2">
        <span className="text-muted-foreground/60">{icon}</span>
        <CardTitle>{title}</CardTitle>
        {sub && (
          <span className="ml-auto font-mono text-[10px] text-muted-foreground/50">{sub}</span>
        )}
      </div>
    </CardHeader>
  );
}

export function Dashboard({ data }: Props) {
  const equityData = useMemo(
    () => buildEquityData(data.equity_curve, data.buy_and_hold_curve, data.timestamps),
    [data.equity_curve, data.buy_and_hold_curve, data.timestamps],
  );

  const drawdownData = useMemo(
    () => buildDrawdownData(data.equity_curve, data.timestamps),
    [data.equity_curve, data.timestamps],
  );

  const drawdownMap = useMemo(
    () => buildDrawdownMap(drawdownData),
    [drawdownData],
  );

  const tradeCount = data.trades.length;

  return (
    <motion.div
      className="flex flex-col gap-4"
      variants={staggerContainerVariants}
      initial="hidden"
      animate="visible"
      viewport={VIEWPORT}
    >
      {/* ── Metrics ──────────────────────────────────────────────────────── */}
      <motion.div variants={slideUpVariants}>
        <PanelErrorBoundary label="Metrics" minHeight="min-h-[92px]">
          <MetricsRow metrics={data.metrics} />
        </PanelErrorBoundary>
      </motion.div>

      {/* ── Candlestick ──────────────────────────────────────────────────── */}
      <motion.div variants={slideUpVariants}>
        <Card>
          <PanelHeader
            icon={<BarChart2 className="h-3.5 w-3.5" aria-hidden />}
            title={`${data.ticker} · Price Chart`}
            sub="▲ buy  ▼ sell"
          />
          <CardContent>
            <PanelErrorBoundary label="Price Chart" minHeight="min-h-95">
              <CandlestickChart ohlcv={data.ohlcv} trades={data.trades} />
            </PanelErrorBoundary>
          </CardContent>
        </Card>
      </motion.div>

      {/* ── Equity + Drawdown side-by-side on xl screens ─────────────────── */}
      <motion.div variants={slideUpVariants} className="grid gap-4 xl:grid-cols-2">
        <Card>
          <PanelHeader
            icon={<TrendingUp className="h-3.5 w-3.5" aria-hidden />}
            title="Equity Curve"
            sub="Strategy vs Buy &amp; Hold"
          />
          <CardContent>
            <PanelErrorBoundary label="Equity Chart" minHeight="min-h-60">
              <EquityChart data={equityData} drawdownMap={drawdownMap} />
            </PanelErrorBoundary>
          </CardContent>
        </Card>

        <Card>
          <PanelHeader
            icon={<TrendingDown className="h-3.5 w-3.5" aria-hidden />}
            title="Drawdown"
            sub="Peak-to-trough"
          />
          <CardContent>
            <PanelErrorBoundary label="Drawdown Chart" minHeight="min-h-60">
              <DrawdownChart data={drawdownData} height={240} />
            </PanelErrorBoundary>
          </CardContent>
        </Card>
      </motion.div>

      {/* ── Trade log ────────────────────────────────────────────────────── */}
      <motion.div variants={slideUpVariants}>
        <Card>
          <PanelHeader
            icon={<List className="h-3.5 w-3.5" aria-hidden />}
            title="Trade Log"
            sub={
              tradeCount > 0
                ? `${tradeCount} trade${tradeCount !== 1 ? "s" : ""}`
                : undefined
            }
          />
          <CardContent>
            <PanelErrorBoundary label="Trade Log" minHeight="min-h-[185px]">
              <TradeLog trades={data.trades} />
            </PanelErrorBoundary>
          </CardContent>
        </Card>
      </motion.div>
    </motion.div>
  );
}
