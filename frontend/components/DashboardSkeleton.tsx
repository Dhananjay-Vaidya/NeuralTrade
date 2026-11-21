import { Skeleton } from "@/components/ui/skeleton";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";

// ── Dimension constants — must stay in sync with Dashboard.tsx ────────────────
//
// CandlestickChart : height={380}  → h-95
// EquityChart      : height={240}  → h-60  (side-by-side at xl)
// DrawdownChart    : height={240}  → h-60  (side-by-side at xl)
// TradeLog         : 5 rows × ~37px ≈ 185px
// MetricCard       : p-4 + label(20) + mt-3(12) + number(24) + pb-4 = 92px interior

function SkeletonPanelHeader({ width = "w-32" }: { width?: string }) {
  return (
    <CardHeader>
      <div className="flex items-center gap-2">
        <Skeleton className="h-3.5 w-3.5 rounded" />
        <Skeleton className={`h-3 ${width}`} />
        <Skeleton className="ml-auto h-3 w-16" />
      </div>
    </CardHeader>
  );
}

export function DashboardSkeleton() {
  return (
    <div className="flex flex-col gap-4">

      {/* ── Metrics row ── */}
      <div className="grid grid-cols-2 gap-3 sm:grid-cols-3 lg:grid-cols-5">
        {Array.from({ length: 5 }).map((_, i) => (
          <Card key={i} className="overflow-hidden">
            {/* Top accent bar placeholder */}
            <Skeleton className="h-0.5 w-full rounded-none" />
            <CardContent className="p-4">
              <div className="flex items-center justify-between">
                <Skeleton className="h-3 w-20" />
                <Skeleton className="h-4 w-4 rounded" />
              </div>
              <Skeleton className="mt-3 h-6 w-24" />
            </CardContent>
          </Card>
        ))}
      </div>

      {/* ── Candlestick ── */}
      <Card>
        <SkeletonPanelHeader width="w-40" />
        <CardContent>
          <Skeleton className="h-95 w-full" />
        </CardContent>
      </Card>

      {/* ── Equity + Drawdown side-by-side (mirrors xl:grid-cols-2) ── */}
      <div className="grid gap-4 xl:grid-cols-2">
        <Card>
          <SkeletonPanelHeader width="w-28" />
          <CardContent>
            <Skeleton className="h-60 w-full" />
          </CardContent>
        </Card>
        <Card>
          <SkeletonPanelHeader width="w-24" />
          <CardContent>
            <Skeleton className="h-60 w-full" />
          </CardContent>
        </Card>
      </div>

      {/* ── Trade log ── */}
      <Card>
        <SkeletonPanelHeader width="w-20" />
        <CardContent>
          {/* Header row */}
          <div className="mb-2.5 flex gap-4 border-b border-white/5 pb-2.5">
            {["w-16", "w-10", "w-16", "w-12"].map((w, i) => (
              <Skeleton key={i} className={`h-3 ${w}`} />
            ))}
          </div>
          {/* 5 data rows */}
          {Array.from({ length: 5 }).map((_, i) => (
            <div key={i} className="flex items-center gap-4 border-b border-white/3 py-2.5 last:border-0">
              <Skeleton className="h-3 w-20" />
              <Skeleton className="h-5 w-10 rounded-md" />
              <Skeleton className="ml-auto h-3 w-16" />
              <Skeleton className="h-3 w-14" />
            </div>
          ))}
        </CardContent>
      </Card>

    </div>
  );
}
