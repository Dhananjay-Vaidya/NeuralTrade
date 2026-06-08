/**
 * Typed API client.
 *
 * All domain types are derived from the generated OpenAPI schema in
 * lib/api-generated.ts (run `npm run generate:api` to refresh after backend changes).
 * This file re-exports the relevant component types and adds the fetch helpers.
 */

import type { components } from "./api-generated";

export const API_BASE = process.env.NEXT_PUBLIC_API_URL ?? "http://localhost:8000";

// ── Domain types (authoritative source: OpenAPI schema) ────────────────────────

export type HealthResponse    = components["schemas"]["HealthResponse"];
export type TickersResponse   = components["schemas"]["TickersResponse"];
export type BacktestRequest   = components["schemas"]["BacktestRequest"];
export type OHLCVBar          = components["schemas"]["OHLCVBar"];
export type Trade             = components["schemas"]["Trade"];
export type Metrics           = components["schemas"]["Metrics"];
export type BacktestResponse  = components["schemas"]["BacktestResponse"];
export type PredictResponse   = components["schemas"]["PredictResponse"];

// ── Fetch helpers ──────────────────────────────────────────────────────────────

async function apiFetch<T>(path: string, init?: RequestInit): Promise<T> {
  const res = await fetch(`${API_BASE}${path}`, {
    headers: { "Content-Type": "application/json" },
    ...init,
  });
  if (!res.ok) {
    const text = await res.text().catch(() => res.statusText);
    throw new Error(`API ${res.status}: ${text}`);
  }
  return res.json() as Promise<T>;
}

export const api = {
  health: () => apiFetch<HealthResponse>("/api/health"),

  tickers: () => apiFetch<TickersResponse>("/api/tickers"),

  backtest: (body: BacktestRequest) =>
    apiFetch<BacktestResponse>("/api/backtest", {
      method: "POST",
      body: JSON.stringify(body),
    }),

  predict: (ticker: string) =>
    apiFetch<PredictResponse>(`/api/predict/${ticker}`),
};
