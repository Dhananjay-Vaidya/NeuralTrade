import { Badge } from "@/components/ui/badge";
import { type Trade } from "@/lib/api";
import { fmt } from "@/lib/utils";

export function TradeLog({ trades }: { trades: Trade[] }) {
  if (trades.length === 0) {
    return (
      <p className="py-6 text-center text-sm text-muted-foreground">
        No trades executed during this backtest.
      </p>
    );
  }

  return (
    <div className="overflow-x-auto">
      <table className="w-full text-sm" aria-label="Trade log">
        <thead>
          <tr className="border-b border-white/10 text-left text-xs text-muted-foreground">
            <th className="pb-2 pr-4 font-medium">Date</th>
            <th className="pb-2 pr-4 font-medium">Side</th>
            <th className="pb-2 pr-4 font-medium text-right">Price</th>
            <th className="pb-2 font-medium text-right">Shares</th>
          </tr>
        </thead>
        <tbody>
          {trades.map((t, i) => (
            <tr key={i} className="border-b border-white/5 last:border-0">
              <td className="py-2 pr-4 font-mono text-xs text-muted-foreground">
                {t.timestamp.split("T")[0]}
              </td>
              <td className="py-2 pr-4">
                <Badge variant={t.side === "BUY" ? "buy" : "sell"}>{t.side}</Badge>
              </td>
              <td className="py-2 pr-4 text-right font-mono">{fmt(t.price, "currency")}</td>
              <td className="py-2 text-right font-mono text-muted-foreground">
                {t.shares.toFixed(4)}
              </td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}
