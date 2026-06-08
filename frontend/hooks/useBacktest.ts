"use client";

import { useQuery } from "@tanstack/react-query";
import { api, type BacktestRequest, type BacktestResponse } from "@/lib/api";

export function useBacktest(params: BacktestRequest | null) {
  return useQuery<BacktestResponse, Error>({
    queryKey: ["backtest", params],
    queryFn: () => api.backtest(params!),
    enabled: params !== null,
  });
}

export function useHealth() {
  return useQuery({
    queryKey: ["health"],
    queryFn: api.health,
    refetchInterval: 30_000,
  });
}

export function useTickers() {
  return useQuery({
    queryKey: ["tickers"],
    queryFn: api.tickers,
  });
}
