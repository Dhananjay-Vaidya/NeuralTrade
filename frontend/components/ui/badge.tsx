import { cn } from "@/lib/utils";

interface BadgeProps extends React.HTMLAttributes<HTMLSpanElement> {
  variant?: "default" | "buy" | "sell" | "hold" | "destructive";
}

const variantClass: Record<NonNullable<BadgeProps["variant"]>, string> = {
  default:     "bg-white/8 text-foreground border border-white/10",
  buy:         "bg-emerald-500/15 text-emerald-400 border border-emerald-500/25",
  sell:        "bg-red-500/15 text-red-400 border border-red-500/25",
  hold:        "bg-amber-500/15 text-amber-400 border border-amber-500/25",
  destructive: "bg-red-500/15 text-red-400 border border-red-500/25",
};

export function Badge({ className, variant = "default", ...props }: BadgeProps) {
  return (
    <span
      className={cn(
        "inline-flex items-center rounded-md px-2 py-0.5 text-[11px] font-semibold leading-none",
        variantClass[variant],
        className
      )}
      {...props}
    />
  );
}
