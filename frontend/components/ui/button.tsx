import * as React from "react";
import { cn } from "@/lib/utils";

interface ButtonProps extends React.ButtonHTMLAttributes<HTMLButtonElement> {
  variant?: "default" | "ghost" | "outline";
  size?: "sm" | "md" | "icon";
}

const variantClass = {
  default: "bg-indigo-600 hover:bg-indigo-500 text-white",
  ghost: "hover:bg-white/10 text-muted-foreground hover:text-foreground",
  outline: "border border-white/10 hover:bg-white/10 text-foreground",
};

const sizeClass = {
  sm: "h-7 px-3 text-xs",
  md: "h-9 px-4 text-sm",
  icon: "h-9 w-9",
};

export const Button = React.forwardRef<HTMLButtonElement, ButtonProps>(
  ({ className, variant = "default", size = "md", ...props }, ref) => (
    <button
      ref={ref}
      className={cn(
        "inline-flex items-center justify-center gap-1.5 rounded-lg font-medium transition-colors focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-indigo-500 disabled:pointer-events-none disabled:opacity-50",
        variantClass[variant],
        sizeClass[size],
        className
      )}
      {...props}
    />
  )
);
Button.displayName = "Button";
