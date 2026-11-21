"use client";

import * as React from "react";
import { ChevronDown } from "lucide-react";
import { cn } from "@/lib/utils";

interface SelectProps extends React.SelectHTMLAttributes<HTMLSelectElement> {
  label?: string;
}

export function Select({ className, label, id, children, ...props }: SelectProps) {
  const uid = id ?? React.useId();
  return (
    <div className="flex flex-col gap-1">
      {label && (
        <label htmlFor={uid} className="text-[11px] font-medium text-muted-foreground">
          {label}
        </label>
      )}
      <div className="relative">
        <select
          id={uid}
          className={cn(
            "h-9 w-full appearance-none rounded-lg border border-white/8 bg-surface",
            "px-3 pr-8 text-sm text-foreground",
            "focus:outline-none focus:ring-2 focus:ring-indigo-500/60 focus:border-indigo-500/40",
            "hover:border-white/14 transition-colors duration-150",
            className
          )}
          {...props}
        >
          {children}
        </select>
        <ChevronDown
          className="pointer-events-none absolute right-2.5 top-1/2 h-3.5 w-3.5 -translate-y-1/2 text-muted-foreground"
          aria-hidden
        />
      </div>
    </div>
  );
}
