"use client";

import { useQuery } from "@tanstack/react-query";
import { motion } from "framer-motion";
import { api } from "@/lib/api";
import { useTickers } from "@/hooks/useBacktest";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Skeleton } from "@/components/ui/skeleton";
import { fmt } from "@/lib/utils";
import { AlertTriangle, TrendingUp, TrendingDown, Minus } from "lucide-react";
import { staggerContainerVariants, slideUpVariants } from "@/lib/motion";

const VIEWPORT = { once: true };

function ActionIcon({ action }: { action?: string }) {
  if (action === "BUY")  return <TrendingUp  className="h-4 w-4 text-emerald-400" aria-hidden />;
  if (action === "SELL") return <TrendingDown className="h-4 w-4 text-red-400"    aria-hidden />;
  return <Minus className="h-4 w-4 text-amber-400" aria-hidden />;
}

function SignalCard({ ticker }: { ticker: string }) {
  const { data, isLoading, isError } = useQuery({
    queryKey: ["predict", ticker],
    queryFn: () => api.predict(ticker),
    retry: false,
  });

  const variant =
    data?.action === "BUY"  ? "buy"  :
    data?.action === "SELL" ? "sell" : "hold";

  return (
    <Card className="flex flex-col">
      <CardHeader>
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-2">
            <ActionIcon action={data?.action} />
            <CardTitle className="text-sm font-bold text-foreground normal-case tracking-normal">
              {ticker}
            </CardTitle>
          </div>
          {data && <Badge variant={variant}>{data.action}</Badge>}
        </div>
      </CardHeader>
      <CardContent className="flex-1">
        {isLoading && (
          <div className="space-y-2">
            <Skeleton className="h-7 w-32" />
            <Skeleton className="h-3 w-20" />
          </div>
        )}
        {isError && (
          <p className="text-xs text-red-400/80">
            Unavailable — run the pipeline first
          </p>
        )}
        {data && (
          <div className="space-y-0.5">
            <p className="font-mono text-2xl font-semibold tabular-nums text-foreground">
              {fmt(data.latest_close, "currency")}
            </p>
            <p className="font-mono text-xs text-muted-foreground">{data.latest_date}</p>
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
      <div
        role="alert"
        className="flex items-center gap-2 border-b border-amber-500/15 bg-amber-500/4 px-4 py-2 text-xs text-amber-400/80"
      >
        <AlertTriangle className="h-3.5 w-3.5 shrink-0" aria-hidden />
        <span>
          <strong className="font-semibold text-amber-400">Research only.</strong>{" "}
          Signals are from a backtested policy and are not financial advice.
        </span>
      </div>

      <main className="flex-1 overflow-y-auto p-4">
        <div className="mb-5">
          <h1 className="text-base font-semibold text-foreground">Latest Signals</h1>
          <p className="mt-1 text-xs text-muted-foreground">
            The action the policy would take on the most recent bar of the test set for each ticker.
          </p>
        </div>

        {isLoading ? (
          <div className="grid gap-4 sm:grid-cols-2 lg:grid-cols-3">
            {[1, 2, 3].map((i) => (
              <Card key={i}>
                <CardContent className="p-4 space-y-3">
                  <Skeleton className="h-4 w-20" />
                  <Skeleton className="h-7 w-28" />
                  <Skeleton className="h-3 w-16" />
                </CardContent>
              </Card>
            ))}
          </div>
        ) : (
          <motion.div
            className="grid gap-4 sm:grid-cols-2 lg:grid-cols-3"
            variants={staggerContainerVariants}
            initial="hidden"
            animate="visible"
            viewport={VIEWPORT}
          >
            {(tickerData?.tickers ?? []).map((t) => (
              <motion.div key={t} variants={slideUpVariants}>
                <SignalCard ticker={t} />
              </motion.div>
            ))}
          </motion.div>
        )}
      </main>
    </>
  );
}
