"use client";

import { Moon, Sun, Wifi, WifiOff } from "lucide-react";
import { useTheme } from "next-themes";
import { useEffect, useState } from "react";
import { Button } from "@/components/ui/button";
import { Select } from "@/components/ui/select";
import { useHealth, useTickers } from "@/hooks/useBacktest";

interface TopBarProps {
  ticker: string;
  onTickerChange: (t: string) => void;
  startDate: string;
  endDate: string;
  onStartChange: (d: string) => void;
  onEndChange: (d: string) => void;
  onRun: () => void;
}

export function TopBar({
  ticker,
  onTickerChange,
  startDate,
  endDate,
  onStartChange,
  onEndChange,
  onRun,
}: TopBarProps) {
  const { theme, setTheme } = useTheme();
  const [mounted, setMounted] = useState(false);
  useEffect(() => setMounted(true), []);

  const { data: health } = useHealth();
  const { data: tickerData } = useTickers();

  return (
    <header className="flex h-14 items-center gap-3 border-b border-white/10 bg-surface px-4 shrink-0">
      {/* Mobile logo */}
      <span className="font-mono font-bold text-foreground md:hidden">
        Neural<span className="text-indigo-400">Trade</span>
      </span>

      <div className="flex flex-1 flex-wrap items-end gap-3">
        {/* Ticker selector */}
        <div className="w-32">
          <Select
            label="Ticker"
            value={ticker}
            onChange={(e) => onTickerChange(e.target.value)}
            aria-label="Select ticker"
          >
            {(tickerData?.tickers ?? ["AAPL"]).map((t) => (
              <option key={t} value={t}>
                {t}
              </option>
            ))}
          </Select>
        </div>

        {/* Date range */}
        <div className="flex flex-col gap-1">
          <label className="text-xs text-muted-foreground">Start</label>
          <input
            type="date"
            value={startDate}
            onChange={(e) => onStartChange(e.target.value)}
            className="h-9 rounded-lg border border-white/10 bg-surface px-2 text-sm text-foreground focus:outline-none focus:ring-2 focus:ring-indigo-500"
            aria-label="Start date"
          />
        </div>
        <div className="flex flex-col gap-1">
          <label className="text-xs text-muted-foreground">End</label>
          <input
            type="date"
            value={endDate}
            onChange={(e) => onEndChange(e.target.value)}
            className="h-9 rounded-lg border border-white/10 bg-surface px-2 text-sm text-foreground focus:outline-none focus:ring-2 focus:ring-indigo-500"
            aria-label="End date"
          />
        </div>

        <Button onClick={onRun} size="sm" className="self-end">
          Run Backtest
        </Button>
      </div>

      {/* Right cluster */}
      <div className="ml-auto flex items-center gap-2 shrink-0">
        {/* Backend status */}
        <span
          className="hidden sm:flex items-center gap-1.5 text-xs"
          title={health ? "Backend connected" : "Backend unreachable"}
        >
          {health?.status === "ok" ? (
            <>
              <Wifi className="h-3.5 w-3.5 text-emerald-400" aria-hidden />
              <span className="text-emerald-400">Connected</span>
            </>
          ) : (
            <>
              <WifiOff className="h-3.5 w-3.5 text-red-400" aria-hidden />
              <span className="text-red-400">Offline</span>
            </>
          )}
        </span>

        {/* Theme toggle */}
        {mounted && (
          <Button
            variant="ghost"
            size="icon"
            onClick={() => setTheme(theme === "dark" ? "light" : "dark")}
            aria-label="Toggle theme"
          >
            {theme === "dark" ? (
              <Sun className="h-4 w-4" aria-hidden />
            ) : (
              <Moon className="h-4 w-4" aria-hidden />
            )}
          </Button>
        )}
      </div>
    </header>
  );
}
