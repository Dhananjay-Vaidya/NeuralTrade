"use client";

import { useState, useRef } from "react";
import { TopBar } from "@/components/layout/TopBar";
import { Dashboard } from "@/components/Dashboard";
import { DashboardSkeleton } from "@/components/DashboardSkeleton";
import { useBacktest, useTickers } from "@/hooks/useBacktest";
import { type BacktestResponse } from "@/lib/api";
import { AlertTriangle, ServerCrash } from "lucide-react";

function DisclaimerBanner() {
  return (
    <div
      role="alert"
      aria-live="polite"
      className="flex items-center gap-2 border-b border-amber-500/20 bg-amber-500/5 px-4 py-2 text-xs text-amber-400"
    >
      <AlertTriangle className="h-3.5 w-3.5 shrink-0" aria-hidden />
      <span>
        <strong>Research &amp; backtesting only.</strong> This tool simulates past performance
        of a trained RL policy. It is <strong>not financial advice</strong> and does not
        guarantee future results.
      </span>
    </div>
  );
}

function ErrorState({ message }: { message: string }) {
  return (
    <div className="flex flex-col items-center justify-center gap-3 py-20 text-center">
      <ServerCrash className="h-12 w-12 text-red-400/60" aria-hidden />
      <p className="text-sm text-muted-foreground">Failed to load backtest data</p>
      <p className="max-w-sm rounded-lg border border-red-500/20 bg-red-500/5 px-4 py-2 font-mono text-xs text-red-400">
        {message}
      </p>
      <p className="text-xs text-muted-foreground">
        Make sure the API server is running:{" "}
        <code className="rounded bg-white/10 px-1">uvicorn api.main:app --reload --port 8000</code>
      </p>
    </div>
  );
}

function EmptyState() {
  return (
    <div className="flex flex-col items-center justify-center gap-3 py-20 text-center">
      <div className="text-4xl">📈</div>
      <p className="text-sm font-medium text-foreground">Select a ticker and run a backtest</p>
      <p className="text-xs text-muted-foreground">
        Choose a ticker and date range above, then click{" "}
        <strong>Run Backtest</strong> to see results.
      </p>
    </div>
  );
}

export default function HomePage() {
  const { data: tickerData } = useTickers();
  const primaryTicker = tickerData?.primary ?? "AAPL";

  const [ticker, setTicker] = useState(primaryTicker);
  const [startDate, setStartDate] = useState("2022-01-01");
  const [endDate, setEndDate] = useState("2024-01-01");
  const [params, setParams] = useState<{ ticker: string; start: string; end: string } | null>(null);

  function handleRun() {
    setParams({ ticker, start: startDate, end: endDate });
  }

  const { data, isLoading, isError, error } = useBacktest(params);

  // Retain the last successfully loaded data while a new fetch is in flight so
  // the chart stays mounted and updates in-place instead of unmounting/remounting.
  const prevDataRef = useRef<BacktestResponse | null>(null);
  if (data) prevDataRef.current = data;
  const displayData = data ?? prevDataRef.current;

  const showSkeleton = isLoading && !displayData;
  const showError    = isError && !displayData;

  return (
    <>
      <DisclaimerBanner />
      <TopBar
        ticker={ticker}
        onTickerChange={setTicker}
        startDate={startDate}
        endDate={endDate}
        onStartChange={setStartDate}
        onEndChange={setEndDate}
        onRun={handleRun}
      />
      <main
        className="flex-1 overflow-y-auto p-4"
        id="main-content"
        aria-label="Dashboard"
      >
        {showSkeleton && <DashboardSkeleton />}
        {showError    && <ErrorState message={(error as Error).message} />}
        {!params && !displayData && <EmptyState />}
        {displayData && (
          // opacity signals loading state without unmounting the chart subtree
          <div className={isLoading ? "pointer-events-none opacity-50 transition-opacity" : "transition-opacity"}>
            <Dashboard data={displayData} />
          </div>
        )}
      </main>
    </>
  );
}
