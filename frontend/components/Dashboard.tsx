"use client";

import { useMemo } from "react";
import dynamic from "next/dynamic";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { MetricsRow } from "@/components/MetricsRow";
import { DrawdownChart } from "@/components/charts/DrawdownChart";
import { EquityChart } from "@/components/charts/EquityChart";
import { TradeLog } from "@/components/TradeLog";
import { type BacktestResponse } from "@/lib/api";
import {
  buildEquityData,
  buildDrawdownData,
  buildDrawdownMap,
} from "@/lib/chartData";

// Lazy-load the heavyweight chart (no SSR — it needs the DOM)
const CandlestickChart = dynamic(
  () => import("@/components/charts/CandlestickChart").then((m) => m.CandlestickChart),
  { ssr: false, loading: () => <div className="h-80 w-full animate-pulse rounded-md bg-white/10" /> }
);

interface Props {
  data: BacktestResponse;
}

export function Dashboard({ data }: Props) {
  // Derived chart data is computed once per data-load and shared between panels.
  // Both EquityChart and DrawdownChart consume the same sampled+derived arrays,
  // so the drawdownMap lookup in the shared tooltip is always consistent.
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

  return (
    <div className="flex flex-col gap-4">
      {/* Performance metrics */}
      <MetricsRow metrics={data.metrics} />

      {/* Candlestick + volume */}
      <Card>
        <CardHeader>
          <CardTitle>
            {data.ticker} · Price Chart{" "}
            <span className="ml-2 text-[10px] font-normal text-muted-foreground">
              ▲ buy &nbsp;▼ sell
            </span>
          </CardTitle>
        </CardHeader>
        <CardContent>
          <CandlestickChart ohlcv={data.ohlcv} trades={data.trades} />
        </CardContent>
      </Card>

      {/* Equity vs B&H — owns the shared cross-panel tooltip */}
      <Card>
        <CardHeader>
          <CardTitle>Equity Curve vs Buy &amp; Hold</CardTitle>
        </CardHeader>
        <CardContent>
          <EquityChart data={equityData} drawdownMap={drawdownMap} />
        </CardContent>
      </Card>

      {/* Drawdown — reference line synced via Zustand store */}
      <Card>
        <CardHeader>
          <CardTitle>Drawdown</CardTitle>
        </CardHeader>
        <CardContent>
          <DrawdownChart data={drawdownData} />
        </CardContent>
      </Card>

      {/* Trade log */}
      <Card>
        <CardHeader>
          <CardTitle>Trade Log</CardTitle>
        </CardHeader>
        <CardContent>
          <TradeLog trades={data.trades} />
        </CardContent>
      </Card>
    </div>
  );
}
