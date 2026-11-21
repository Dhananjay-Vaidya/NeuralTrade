import * as React from "react";
import { cn } from "@/lib/utils";

interface ButtonProps extends React.ButtonHTMLAttributes<HTMLButtonElement> {
  variant?: "default" | "ghost" | "outline";
  size?: "sm" | "md" | "icon";
}

const variantClass: Record<NonNullable<ButtonProps["variant"]>, string> = {
  default:
    "bg-indigo-600 hover:bg-indigo-500 active:bg-indigo-700 text-white shadow-[0_1px_0_rgba(255,255,255,0.12)_inset] active:scale-[0.98]",
  ghost:
    "hover:bg-white/6 text-muted-foreground hover:text-foreground active:bg-white/10 active:scale-[0.98]",
  outline:
    "border border-white/10 hover:border-white/20 hover:bg-white/5 text-foreground active:scale-[0.98]",
};

const sizeClass: Record<NonNullable<ButtonProps["size"]>, string> = {
  sm:   "h-8 px-3 text-xs rounded-lg",
  md:   "h-9 px-4 text-sm rounded-lg",
  icon: "h-8 w-8 rounded-lg",
};

export const Button = React.forwardRef<HTMLButtonElement, ButtonProps>(
  ({ className, variant = "default", size = "md", ...props }, ref) => (
    <button
      ref={ref}
      className={cn(
        "inline-flex items-center justify-center gap-1.5 font-medium",
        "transition-[background-color,border-color,transform,opacity] duration-150",
        "focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-indigo-500/70",
        "disabled:pointer-events-none disabled:opacity-40",
        variantClass[variant],
        sizeClass[size],
        className
      )}
      {...props}
    />
  )
);
Button.displayName = "Button";
