import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";

export default function AboutPage() {
  return (
    <main className="flex-1 overflow-y-auto p-4">
      <div className="mx-auto max-w-2xl space-y-4">
        <h1 className="text-lg font-semibold">About NeuralTrade</h1>

        <div className="rounded-xl border border-amber-500/30 bg-amber-500/5 p-4 text-sm text-amber-300">
          <strong>⚠️ Disclaimer:</strong> NeuralTrade is a research and educational tool. All
          results shown are from backtesting a reinforcement-learning policy on historical data.
          Past performance does not guarantee future results. Nothing on this platform
          constitutes financial advice.
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
              The dashboard lets you inspect what the policy did during the test period — which
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
            <ol className="list-decimal space-y-1.5 pl-4">
              <li>Download OHLCV from yfinance</li>
              <li>Compute 12 technical indicators (SMA, EMA, RSI, MACD, BB, ATR, ADX)</li>
              <li>Chronological 70 / 15 / 15 train / val / test split — RobustScaler fitted on train only</li>
              <li>PPO trained on training set, hyperparams logged to MLflow</li>
              <li>Evaluate on held-out test set; compare against buy-and-hold</li>
            </ol>
          </CardContent>
        </Card>

        <Card>
          <CardHeader>
            <CardTitle>API</CardTitle>
          </CardHeader>
          <CardContent className="text-sm text-muted-foreground">
            <p>
              The backend exposes a FastAPI server at{" "}
              <code className="rounded bg-white/10 px-1">http://localhost:8000</code>.
              See <code className="rounded bg-white/10 px-1">api/README.md</code> for the full
              contract.
            </p>
          </CardContent>
        </Card>
      </div>
    </main>
  );
}
