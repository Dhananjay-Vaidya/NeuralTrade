/** Shared chart-data utilities used by both EquityChart and DrawdownChart.
 *  Lifted here so Dashboard can compute once and pass consistent arrays down.
 */

export interface EquityPoint {
  date: string;
  strategy: number;
  buyAndHold: number;
}

export interface DrawdownPoint {
  date: string;
  drawdown: number;
}

export function computeDrawdown(equity: number[]): number[] {
  let peak = equity[0] ?? 0;
  return equity.map((v) => {
    if (v > peak) peak = v;
    return peak > 0 ? -(peak - v) / peak : 0;
  });
}

export function sampleArray<T>(arr: T[], max: number): T[] {
  if (arr.length <= max) return arr;
  const step = Math.ceil(arr.length / max);
  return arr.filter((_, i) => i % step === 0 || i === arr.length - 1);
}

export function buildEquityData(
  equity: number[],
  bh: number[],
  timestamps: string[],
): EquityPoint[] {
  const len = Math.min(equity.length, bh.length, timestamps.length);
  const raw = Array.from({ length: len }, (_, i) => ({
    date:       timestamps[i]?.split("T")[0] ?? String(i),
    strategy:   equity[i]!,
    buyAndHold: bh[i]!,
  }));
  return sampleArray(raw, 300);
}

export function buildDrawdownData(
  equity: number[],
  timestamps: string[],
): DrawdownPoint[] {
  const dd  = computeDrawdown(equity);
  const len = Math.min(dd.length, timestamps.length);
  const raw = Array.from({ length: len }, (_, i) => ({
    date:     timestamps[i]?.split("T")[0] ?? String(i),
    drawdown: dd[i]!,
  }));
  return sampleArray(raw, 300);
}

/** Build a date→drawdown lookup from a DrawdownPoint array. O(1) per lookup. */
export function buildDrawdownMap(points: DrawdownPoint[]): Map<string, number> {
  return new Map(points.map((p) => [p.date, p.drawdown]));
}
