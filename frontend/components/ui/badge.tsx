import { cn } from "@/lib/utils";

interface BadgeProps extends React.HTMLAttributes<HTMLSpanElement> {
  variant?: "default" | "buy" | "sell" | "hold" | "destructive";
}

const variantClass = {
  default: "bg-white/10 text-foreground",
  buy: "bg-emerald-500/20 text-emerald-400 border border-emerald-500/30",
  sell: "bg-red-500/20 text-red-400 border border-red-500/30",
  hold: "bg-yellow-500/20 text-yellow-400 border border-yellow-500/30",
  destructive: "bg-red-500/20 text-red-400 border border-red-500/30",
};

export function Badge({ className, variant = "default", ...props }: BadgeProps) {
  return (
    <span
      className={cn(
        "inline-flex items-center rounded-full px-2.5 py-0.5 text-xs font-semibold",
        variantClass[variant],
        className
      )}
      {...props}
    />
  );
}
