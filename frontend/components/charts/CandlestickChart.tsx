"use client";

import { useEffect, useRef, useState, useCallback } from "react";
import {
  createChart,
  createSeriesMarkers,
  CandlestickSeries,
  HistogramSeries,
  type IChartApi,
  type ISeriesApi,
  type ISeriesMarkersPluginApi,
  type CandlestickData,
  type HistogramData,
  type SeriesMarker,
  type Time,
  type MouseEventParams,
} from "lightweight-charts";
import { type OHLCVBar, type Trade } from "@/lib/api";
import { useChartStore } from "@/lib/chartStore";

// ── Types ──────────────────────────────────────────────────────────────────────

interface Props {
  ohlcv: OHLCVBar[];
  trades: Trade[];
  equityCurve?: number[];
}

interface TooltipState {
  visible: boolean;
  x: number;
  y: number;
  trade: Trade;
  pnl: number | null;
}

// ── Helpers ────────────────────────────────────────────────────────────────────

function tradeId(t: Trade, idx: number) {
  return `${t.side}-${t.timestamp}-${idx}`;
}

function computePnl(trade: Trade, allTrades: Trade[]): number | null {
  if (trade.side === "BUY") return null;
  const sellIdx = allTrades.indexOf(trade);
  for (let i = sellIdx - 1; i >= 0; i--) {
    if (allTrades[i].side === "BUY") {
      return (trade.price - allTrades[i].price) * trade.shares;
    }
  }
  return null;
}

// ── Component ──────────────────────────────────────────────────────────────────

export function CandlestickChart({ ohlcv, trades }: Props) {
  const setHoveredDate = useChartStore((s) => s.setHoveredDate);
  const containerRef = useRef<HTMLDivElement>(null);
  const chartRef     = useRef<IChartApi | null>(null);
  const candleRef    = useRef<ISeriesApi<"Candlestick"> | null>(null);
  const volRef       = useRef<ISeriesApi<"Histogram"> | null>(null);
  const pluginRef    = useRef<ISeriesMarkersPluginApi<Time> | null>(null);
  const tradeMapRef  = useRef<Map<string, { trade: Trade; pnl: number | null }>>(new Map());

  // Track last-seen identity to skip setData when the reference hasn't changed.
  // TanStack Query returns the same array reference when data is cached, so this
  // avoids redundant setData calls on refocus / background refetches.
  const lastOhlcvRef  = useRef<OHLCVBar[] | null>(null);
  const lastTradesRef = useRef<Trade[] | null>(null);

  // rafId used to debounce ResizeObserver — prevents the observer→applyOptions
  // →layout-change→observer feedback loop.
  const rafRef       = useRef<number>(0);
  const lastWidthRef = useRef<number>(0);

  const [tooltip, setTooltip] = useState<TooltipState | null>(null);

  // ── Chart init + event subscription (single effect, runs once) ────────────
  //
  // Combining init and subscribe in one effect is required: if they're split,
  // the subscribe effect fires synchronously after the init effect but before
  // the chart has painted, so chartRef.current is still null and subscriptions
  // are silently dropped.

  const handleClick = useCallback((param: MouseEventParams<Time>) => {
    if (!param.hoveredInfo) { setTooltip(null); return; }
    const { objectKind, objectId } = param.hoveredInfo;
    if (objectKind !== "series-marker" || !objectId) { setTooltip(null); return; }
    const entry = tradeMapRef.current.get(String(objectId));
    if (!entry || !param.point) return;
    setTooltip({ visible: true, x: param.point.x, y: param.point.y, ...entry });
  }, []);

  const handleCrosshairMove = useCallback((param: MouseEventParams<Time>) => {
    // Broadcast hovered date to the cross-panel sync store.
    // param.time is undefined when the crosshair leaves the chart area.
    setHoveredDate(param.time ? String(param.time) : null);

    if (!param.hoveredInfo || param.hoveredInfo.objectKind !== "series-marker") {
      setTooltip((prev) => (prev?.visible ? { ...prev, visible: false } : prev));
    }
  }, [setHoveredDate]);

  useEffect(() => {
    const container = containerRef.current;
    if (!container) return;

    const chart = createChart(container, {
      layout: {
        background: { color: "transparent" },
        textColor: "#9ca3af",
        fontFamily: "ui-monospace, SFMono-Regular, monospace",
      },
      grid: {
        vertLines: { color: "rgba(255,255,255,0.04)" },
        horzLines: { color: "rgba(255,255,255,0.04)" },
      },
      crosshair: { mode: 1 },
      rightPriceScale: { borderColor: "rgba(255,255,255,0.08)" },
      timeScale: {
        borderColor: "rgba(255,255,255,0.08)",
        timeVisible: true,
        secondsVisible: false,
      },
      width: container.clientWidth,
      height: 380,
    });

    const candles = chart.addSeries(CandlestickSeries, {
      upColor: "#26a69a",
      downColor: "#ef5350",
      borderVisible: false,
      wickUpColor: "#26a69a",
      wickDownColor: "#ef5350",
    });

    const vol = chart.addSeries(HistogramSeries, {
      priceFormat: { type: "volume" },
      priceScaleId: "vol",
    });
    chart.priceScale("vol").applyOptions({ scaleMargins: { top: 0.82, bottom: 0 } });

    chartRef.current  = chart;
    candleRef.current = candles;
    volRef.current    = vol;
    pluginRef.current = createSeriesMarkers(candles);
    lastWidthRef.current = container.clientWidth;

    // Subscribe here so handlers are guaranteed to be registered against a live chart.
    chart.subscribeClick(handleClick);
    chart.subscribeCrosshairMove(handleCrosshairMove);

    // ResizeObserver with RAF debounce + width-change guard to break the
    // applyOptions → layout-change → ResizeObserver notification loop.
    const ro = new ResizeObserver(() => {
      cancelAnimationFrame(rafRef.current);
      rafRef.current = requestAnimationFrame(() => {
        const w = container.clientWidth;
        if (w !== lastWidthRef.current) {
          lastWidthRef.current = w;
          chart.applyOptions({ width: w });
        }
      });
    });
    ro.observe(container);

    return () => {
      cancelAnimationFrame(rafRef.current);
      ro.disconnect();
      chart.unsubscribeClick(handleClick);
      chart.unsubscribeCrosshairMove(handleCrosshairMove);
      chart.remove();
      chartRef.current  = null;
      candleRef.current = null;
      volRef.current    = null;
      pluginRef.current = null;
      lastOhlcvRef.current  = null;
      lastTradesRef.current = null;
      // Clear cross-panel highlight so Recharts lines don't freeze on last position.
      setHoveredDate(null);
    };
  }, [handleClick, handleCrosshairMove]); // handlers are stable (useCallback [])

  // ── OHLCV data update ─────────────────────────────────────────────────────
  //
  // Guard against same-reference re-feeds: TanStack Query returns the cached
  // reference when data hasn't changed (background refetch, focus refetch, etc.).
  // Without this guard every refetch would call setData unnecessarily.

  useEffect(() => {
    if (
      !candleRef.current ||
      !volRef.current    ||
      ohlcv.length === 0 ||
      ohlcv === lastOhlcvRef.current
    ) return;
    lastOhlcvRef.current = ohlcv;

    const candleData: CandlestickData<Time>[] = ohlcv.map((b) => ({
      time:  b.timestamp.split("T")[0] as Time,
      open:  b.open,
      high:  b.high,
      low:   b.low,
      close: b.close,
    }));

    const volData: HistogramData<Time>[] = ohlcv.map((b) => ({
      time:  b.timestamp.split("T")[0] as Time,
      value: b.volume,
      color: b.close >= b.open ? "rgba(38,166,154,0.35)" : "rgba(239,83,80,0.35)",
    }));

    candleRef.current.setData(candleData);
    volRef.current.setData(volData);
    chartRef.current?.timeScale().fitContent();
  }, [ohlcv]);

  // ── Marker update ─────────────────────────────────────────────────────────

  useEffect(() => {
    const plugin = pluginRef.current;
    if (!plugin || trades === lastTradesRef.current) return;
    lastTradesRef.current = trades;

    const idMap = new Map<string, { trade: Trade; pnl: number | null }>();
    const markers: SeriesMarker<Time>[] = trades.map((t, idx) => {
      const id  = tradeId(t, idx);
      const pnl = computePnl(t, trades);
      idMap.set(id, { trade: t, pnl });
      return {
        id,
        time:     t.timestamp.split("T")[0] as Time,
        position: t.side === "BUY" ? ("belowBar" as const) : ("aboveBar" as const),
        color:    t.side === "BUY" ? "#4ade80" : "#f87171",
        shape:    t.side === "BUY" ? ("arrowUp" as const) : ("arrowDown" as const),
        text:     t.side === "BUY" ? `B $${t.price.toFixed(2)}` : `S $${t.price.toFixed(2)}`,
        size:     1.5,
      };
    });

    markers.sort((a, b) => String(a.time).localeCompare(String(b.time)));
    plugin.setMarkers(markers);
    tradeMapRef.current = idMap;
  }, [trades]);

  // ── Render ────────────────────────────────────────────────────────────────

  return (
    <div className="relative w-full" onClick={() => setTooltip(null)}>
      <div ref={containerRef} className="w-full" />
      {tooltip?.visible && (
        <MarkerTooltip tooltip={tooltip} onClose={() => setTooltip(null)} />
      )}
    </div>
  );
}

// ── Marker tooltip overlay ────────────────────────────────────────────────────

interface MarkerTooltipProps {
  tooltip: TooltipState;
  onClose: () => void;
}

function MarkerTooltip({ tooltip, onClose }: MarkerTooltipProps) {
  const { x, y, trade, pnl } = tooltip;
  const isBuy = trade.side === "BUY";

  const left = Math.max(8, x - 80);
  const top  = Math.max(8, y - (isBuy ? 120 : 16));

  const pnlColor = pnl === null ? "text-muted-foreground" : pnl >= 0 ? "text-emerald-400" : "text-red-400";
  const pnlLabel = pnl === null ? "—" : `${pnl >= 0 ? "+" : ""}$${pnl.toFixed(2)}`;

  return (
    <div
      role="tooltip"
      className="pointer-events-none absolute z-50 min-w-35 rounded-lg border border-white/10 bg-[#1a1a2e] px-3 py-2.5 text-xs shadow-xl"
      ref={(el) => {
        if (el) { el.style.left = `${left}px`; el.style.top = `${top}px`; }
      }}
      onClick={(e) => e.stopPropagation()}
    >
      <div className="mb-1.5 flex items-center gap-1.5">
        <span className={`inline-block h-2 w-2 rounded-full ${isBuy ? "bg-emerald-400" : "bg-red-400"}`} />
        <span className={`font-semibold ${isBuy ? "text-emerald-400" : "text-red-400"}`}>{trade.side}</span>
      </div>
      <table className="w-full border-separate border-spacing-y-0.5">
        <tbody className="text-muted-foreground">
          <tr>
            <td className="pr-3">Date</td>
            <td className="text-right font-mono text-foreground">{trade.timestamp.split("T")[0]}</td>
          </tr>
          <tr>
            <td className="pr-3">Price</td>
            <td className="text-right font-mono text-foreground">${trade.price.toFixed(2)}</td>
          </tr>
          <tr>
            <td className="pr-3">Shares</td>
            <td className="text-right font-mono text-foreground">{trade.shares.toFixed(4)}</td>
          </tr>
          {trade.side === "SELL" && (
            <tr>
              <td className="pr-3">Round-trip P&amp;L</td>
              <td className={`text-right font-mono font-semibold ${pnlColor}`}>{pnlLabel}</td>
            </tr>
          )}
        </tbody>
      </table>
      <button
        type="button"
        className="absolute right-2 top-2 text-muted-foreground hover:text-foreground"
        onClick={(e) => { e.stopPropagation(); onClose(); }}
        aria-label="Close"
      >
        ×
      </button>
    </div>
  );
}
