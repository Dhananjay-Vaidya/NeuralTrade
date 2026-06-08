"use client";

import { useQuery } from "@tanstack/react-query";
import { api } from "@/lib/api";
import { useTickers } from "@/hooks/useBacktest";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Skeleton } from "@/components/ui/skeleton";
import { fmt } from "@/lib/utils";
import { AlertTriangle } from "lucide-react";

function SignalCard({ ticker }: { ticker: string }) {
  const { data, isLoading, isError } = useQuery({
    queryKey: ["predict", ticker],
    queryFn: () => api.predict(ticker),
    retry: false,
  });

  const variant =
    data?.action === "BUY" ? "buy" : data?.action === "SELL" ? "sell" : "hold";

  return (
    <Card>
      <CardHeader>
        <div className="flex items-center justify-between">
          <CardTitle>{ticker}</CardTitle>
          {data && <Badge variant={variant}>{data.action}</Badge>}
        </div>
      </CardHeader>
      <CardContent>
        {isLoading && <Skeleton className="h-10 w-full" />}
        {isError && (
          <p className="text-xs text-red-400">
            Unavailable — run the pipeline first
          </p>
        )}
        {data && (
          <div className="space-y-1">
            <p className="font-mono text-2xl font-semibold">
              {fmt(data.latest_close, "currency")}
            </p>
            <p className="text-xs text-muted-foreground">{data.latest_date}</p>
          </div>
        )}
      </CardContent>
    </Card>
  );
}

export default function SignalsPage() {
  const { data: tickerData, isLoading } = useTickers();

  return (
    <>
      {/* Disclaimer */}
      <div
        role="alert"
        className="flex items-center gap-2 border-b border-amber-500/20 bg-amber-500/5 px-4 py-2 text-xs text-amber-400"
      >
        <AlertTriangle className="h-3.5 w-3.5 shrink-0" aria-hidden />
        <span>
          <strong>Research only.</strong> Signals are from a backtested policy and are{" "}
          <strong>not financial advice</strong>.
        </span>
      </div>

      <main className="flex-1 overflow-y-auto p-4">
        <h1 className="mb-4 text-lg font-semibold">Latest Signals</h1>
        <p className="mb-6 text-xs text-muted-foreground">
          The action the policy would take on the most recent bar of the test set for each ticker.
        </p>
        {isLoading ? (
          <div className="grid gap-4 sm:grid-cols-2 lg:grid-cols-3">
            {[1, 2, 3].map((i) => (
              <Card key={i}>
                <CardContent className="p-4">
                  <Skeleton className="h-16 w-full" />
                </CardContent>
              </Card>
            ))}
          </div>
        ) : (
          <div className="grid gap-4 sm:grid-cols-2 lg:grid-cols-3">
            {(tickerData?.tickers ?? []).map((t) => (
              <SignalCard key={t} ticker={t} />
            ))}
          </div>
        )}
      </main>
    </>
  );
}
