"use client";

import { Moon, Sun, Wifi, WifiOff, Play } from "lucide-react";
import { useTheme } from "next-themes";
import { useEffect, useState } from "react";
import { motion, AnimatePresence } from "framer-motion";
import { Button } from "@/components/ui/button";
import { Select } from "@/components/ui/select";
import { useHealth, useTickers } from "@/hooks/useBacktest";
import { cn } from "@/lib/utils";

interface TopBarProps {
  ticker: string;
  onTickerChange: (t: string) => void;
  startDate: string;
  endDate: string;
  onStartChange: (d: string) => void;
  onEndChange: (d: string) => void;
  onRun: () => void;
}

const DATE_INPUT_CLASS = cn(
  "h-9 rounded-lg border border-white/8 bg-surface px-2.5 text-sm text-foreground",
  "focus:outline-none focus:ring-2 focus:ring-indigo-500/60 focus:border-indigo-500/40",
  "transition-colors duration-150 hover:border-white/14",
  "placeholder:text-muted-foreground"
);

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

  const isConnected = health?.status === "ok";

  return (
    <header className="flex h-14 shrink-0 items-center gap-3 border-b border-white/[0.07] bg-surface/80 px-4 backdrop-blur-sm">
      {/* Mobile logo */}
      <span className="font-mono font-bold text-foreground md:hidden">
        Neural<span className="text-indigo-400">Trade</span>
      </span>

      <div className="flex flex-1 flex-wrap items-end gap-3">
        {/* Ticker */}
        <div className="w-32">
          <Select
            label="Ticker"
            value={ticker}
            onChange={(e) => onTickerChange(e.target.value)}
            aria-label="Select ticker"
          >
            {(tickerData?.tickers ?? ["AAPL"]).map((t) => (
              <option key={t} value={t}>{t}</option>
            ))}
          </Select>
        </div>

        {/* Date range */}
        <div className="flex flex-col gap-1">
          <label className="text-[11px] font-medium text-muted-foreground">Start</label>
          <input
            type="date"
            value={startDate}
            onChange={(e) => onStartChange(e.target.value)}
            className={DATE_INPUT_CLASS}
            aria-label="Start date"
          />
        </div>
        <div className="flex flex-col gap-1">
          <label className="text-[11px] font-medium text-muted-foreground">End</label>
          <input
            type="date"
            value={endDate}
            onChange={(e) => onEndChange(e.target.value)}
            className={DATE_INPUT_CLASS}
            aria-label="End date"
          />
        </div>

        <Button
          onClick={onRun}
          size="sm"
          className="self-end gap-1.5"
        >
          <Play className="h-3 w-3 fill-current" aria-hidden />
          Run Backtest
        </Button>
      </div>

      {/* Right cluster */}
      <div className="ml-auto flex items-center gap-2 shrink-0">
        {/* Backend status */}
        <AnimatePresence mode="wait" initial={false}>
          <motion.span
            key={isConnected ? "ok" : "off"}
            className="hidden sm:flex items-center gap-1.5 text-xs"
            title={isConnected ? "Backend connected" : "Backend unreachable"}
            initial={{ opacity: 0, y: -4 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0, y: 4 }}
            transition={{ duration: 0.15, ease: [0, 0, 0.2, 1] }}
          >
            {isConnected ? (
              <>
                <span className="relative flex h-2 w-2">
                  <span className="absolute inline-flex h-full w-full animate-ping rounded-full bg-emerald-400 opacity-50" />
                  <span className="relative inline-flex h-2 w-2 rounded-full bg-emerald-400" />
                </span>
                <span className="text-emerald-400 font-medium">Connected</span>
              </>
            ) : (
              <>
                <WifiOff className="h-3.5 w-3.5 text-red-400" aria-hidden />
                <span className="text-red-400 font-medium">Offline</span>
              </>
            )}
          </motion.span>
        </AnimatePresence>

        {/* Divider */}
        <span className="hidden sm:block h-4 w-px bg-white/10" aria-hidden />

        {/* Theme toggle */}
        {mounted && (
          <button
            type="button"
            onClick={() => setTheme(theme === "dark" ? "light" : "dark")}
            aria-label="Toggle theme"
            className={cn(
              "flex h-8 w-8 items-center justify-center rounded-lg",
              "text-muted-foreground hover:text-foreground hover:bg-white/6",
              "transition-colors duration-150 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-indigo-500"
            )}
          >
            <AnimatePresence mode="wait" initial={false}>
              <motion.span
                key={theme}
                initial={{ opacity: 0, rotate: -30, scale: 0.7 }}
                animate={{ opacity: 1, rotate: 0,   scale: 1   }}
                exit={{ opacity: 0,   rotate:  30, scale: 0.7 }}
                transition={{ duration: 0.18, ease: [0, 0, 0.2, 1] }}
              >
                {theme === "dark"
                  ? <Sun  className="h-4 w-4" aria-hidden />
                  : <Moon className="h-4 w-4" aria-hidden />
                }
              </motion.span>
            </AnimatePresence>
          </button>
        )}
      </div>
    </header>
  );
}
