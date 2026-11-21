"use client";

import { useState, useRef } from "react";
import { motion, AnimatePresence } from "framer-motion";
import { AlertTriangle, ServerCrash, BarChart2 } from "lucide-react";
import { TopBar } from "@/components/layout/TopBar";
import { Dashboard } from "@/components/Dashboard";
import { DashboardSkeleton } from "@/components/DashboardSkeleton";
import { useBacktest, useTickers } from "@/hooks/useBacktest";
import { type BacktestResponse } from "@/lib/api";
import { fadeVariants } from "@/lib/motion";

// ── Disclaimer banner ─────────────────────────────────────────────────────────

function DisclaimerBanner() {
  return (
    <div
      role="alert"
      aria-live="polite"
      className="flex items-center gap-2 border-b border-amber-500/15 bg-amber-500/4 px-4 py-2 text-xs text-amber-400/80"
    >
      <AlertTriangle className="h-3.5 w-3.5 shrink-0" aria-hidden />
      <span>
        <strong className="font-semibold text-amber-400">Research &amp; backtesting only.</strong>{" "}
        Simulates past performance of a trained RL policy. Not financial advice and does not
        guarantee future results.
      </span>
    </div>
  );
}

// ── Error state ───────────────────────────────────────────────────────────────

function ErrorState({ message }: { message: string }) {
  return (
    <motion.div
      className="flex flex-col items-center justify-center gap-4 py-24 text-center"
      variants={fadeVariants}
      initial="hidden"
      animate="visible"
    >
      <div className="flex h-14 w-14 items-center justify-center rounded-2xl border border-red-500/20 bg-red-500/8">
        <ServerCrash className="h-6 w-6 text-red-400/70" aria-hidden />
      </div>
      <div className="space-y-1">
        <p className="text-sm font-medium text-foreground">Failed to load backtest data</p>
        <p className="text-xs text-muted-foreground">Check that the API server is running</p>
      </div>
      <p className="max-w-sm rounded-lg border border-red-500/15 bg-red-500/5 px-4 py-2.5 font-mono text-xs text-red-400/90">
        {message}
      </p>
      <p className="text-xs text-muted-foreground">
        <code className="rounded bg-white/8 px-1.5 py-0.5 font-mono text-[11px]">
          uvicorn api.main:app --reload --port 8000
        </code>
      </p>
    </motion.div>
  );
}

// ── Empty state ───────────────────────────────────────────────────────────────

function EmptyState() {
  return (
    <motion.div
      className="flex flex-col items-center justify-center gap-4 py-24 text-center"
      variants={fadeVariants}
      initial="hidden"
      animate="visible"
    >
      <div className="flex h-14 w-14 items-center justify-center rounded-2xl border border-indigo-500/20 bg-indigo-500/8">
        <BarChart2 className="h-6 w-6 text-indigo-400/70" aria-hidden />
      </div>
      <div className="space-y-1">
        <p className="text-sm font-medium text-foreground">Select a ticker and run a backtest</p>
        <p className="max-w-xs text-xs text-muted-foreground">
          Choose a ticker and date range above, then click{" "}
          <strong className="text-foreground/70">Run Backtest</strong> to see results.
        </p>
      </div>
    </motion.div>
  );
}

// ── Page ──────────────────────────────────────────────────────────────────────

export default function HomePage() {
  const { data: tickerData } = useTickers();
  const primaryTicker = tickerData?.primary ?? "AAPL";

  const [ticker, setTicker]       = useState(primaryTicker);
  const [startDate, setStartDate] = useState("2022-01-01");
  const [endDate, setEndDate]     = useState("2024-01-01");
  const [params, setParams]       = useState<{ ticker: string; start: string; end: string } | null>(null);

  function handleRun() {
    setParams({ ticker, start: startDate, end: endDate });
  }

  const { data, isLoading, isError, error } = useBacktest(params);

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
        <AnimatePresence mode="wait" initial={false}>
          {showSkeleton && (
            <motion.div key="skeleton" variants={fadeVariants} initial="hidden" animate="visible" exit="exit">
              <DashboardSkeleton />
            </motion.div>
          )}
          {showError && (
            <motion.div key="error" variants={fadeVariants} initial="hidden" animate="visible" exit="exit">
              <ErrorState message={(error as Error).message} />
            </motion.div>
          )}
          {!params && !displayData && (
            <motion.div key="empty" variants={fadeVariants} initial="hidden" animate="visible" exit="exit">
              <EmptyState />
            </motion.div>
          )}
        </AnimatePresence>

        {displayData && (
          <div
            className={
              isLoading
                ? "pointer-events-none opacity-50 transition-opacity duration-200"
                : "transition-opacity duration-200"
            }
          >
            <Dashboard data={displayData} />
          </div>
        )}
      </main>
    </>
  );
}
