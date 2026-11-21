import { Badge } from "@/components/ui/badge";
import { type Trade } from "@/lib/api";
import { fmt } from "@/lib/utils";

const TH = "pb-2.5 text-left text-[10px] font-semibold uppercase tracking-widest text-muted-foreground/60";
const TD_MONO = "py-2.5 font-mono text-xs text-muted-foreground";

export function TradeLog({ trades }: { trades: Trade[] }) {
  if (trades.length === 0) {
    return (
      <div className="flex min-h-[185px] items-center justify-center text-sm text-muted-foreground">
        No trades executed in this date range.
      </div>
    );
  }

  return (
    <div className="overflow-x-auto">
      <table className="w-full text-sm" aria-label="Trade log">
        <thead>
          <tr className="border-b border-white/5">
            <th className={TH}>Date</th>
            <th className={TH}>Side</th>
            <th className={`${TH} text-right`}>Price</th>
            <th className={`${TH} text-right`}>Shares</th>
          </tr>
        </thead>
        <tbody>
          {trades.map((t, i) => (
            <tr
              key={i}
              className="group border-b border-white/3 last:border-0 transition-colors duration-100 hover:bg-white/2"
            >
              <td className={TD_MONO}>{t.timestamp.split("T")[0]}</td>
              <td className="py-2.5">
                <Badge variant={t.side === "BUY" ? "buy" : "sell"}>{t.side}</Badge>
              </td>
              <td className="py-2.5 text-right font-mono text-xs tabular-nums text-foreground">
                {fmt(t.price, "currency")}
              </td>
              <td className="py-2.5 text-right font-mono text-xs tabular-nums text-muted-foreground">
                {t.shares.toFixed(4)}
              </td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}
