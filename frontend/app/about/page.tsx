import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";

export default function AboutPage() {
  return (
    <main className="flex-1 overflow-y-auto p-4">
      <div className="mx-auto max-w-2xl space-y-4">

        <div className="mb-6">
          <h1 className="text-base font-semibold text-foreground">About NeuralTrade</h1>
          <p className="mt-1 text-xs text-muted-foreground">
            RL-based single-asset trading research prototype.
          </p>
        </div>

        {/* Disclaimer */}
        <div className="rounded-xl border border-amber-500/20 bg-amber-500/4 p-4 text-sm text-amber-300/90 leading-relaxed">
          <strong className="font-semibold text-amber-400">Disclaimer:</strong>{" "}
          NeuralTrade is a research and educational tool. All results are from backtesting a
          reinforcement-learning policy on historical data. Past performance does not guarantee
          future results. Nothing on this platform constitutes financial advice.
        </div>

        <Card>
          <CardHeader>
            <CardTitle>What is this?</CardTitle>
          </CardHeader>
          <CardContent className="space-y-2 text-sm text-muted-foreground leading-relaxed">
            <p>
              NeuralTrade is an RL-based single-asset trading research prototype. A PPO (or
              RecurrentPPO) agent is trained on historical OHLCV data enriched with technical
              indicators, then evaluated on a held-out test set.
            </p>
            <p>
              The dashboard lets you inspect what the policy did during the test period: which
              trades it took, how the portfolio evolved, and how it compares to a naive
              buy-and-hold baseline.
            </p>
          </CardContent>
        </Card>

        <Card>
          <CardHeader>
            <CardTitle>Pipeline</CardTitle>
          </CardHeader>
          <CardContent className="text-sm text-muted-foreground">
            <ol className="space-y-2 pl-4 list-decimal">
              {[
                "Download OHLCV from yfinance",
                "Compute 12 technical indicators (SMA, EMA, RSI, MACD, BB, ATR, ADX)",
                "Chronological 70 / 15 / 15 train / val / test split — RobustScaler fitted on train only",
                "PPO trained on training set, hyperparams logged to MLflow",
                "Evaluate on held-out test set; compare against buy-and-hold",
              ].map((step, i) => (
                <li key={i} className="leading-relaxed">{step}</li>
              ))}
            </ol>
          </CardContent>
        </Card>

        <Card>
          <CardHeader>
            <CardTitle>API</CardTitle>
          </CardHeader>
          <CardContent className="text-sm text-muted-foreground leading-relaxed">
            <p>
              The backend exposes a FastAPI server at{" "}
              <code className="rounded-md border border-white/8 bg-white/4 px-1.5 py-0.5 font-mono text-xs text-foreground/80">
                http://localhost:8000
              </code>
              . See{" "}
              <code className="rounded-md border border-white/8 bg-white/4 px-1.5 py-0.5 font-mono text-xs text-foreground/80">
                api/README.md
              </code>{" "}
              for the full contract.
            </p>
          </CardContent>
        </Card>

      </div>
    </main>
  );
}
