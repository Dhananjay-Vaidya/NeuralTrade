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
        <label htmlFor={uid} className="text-xs text-muted-foreground">
          {label}
        </label>
      )}
      <div className="relative">
        <select
          id={uid}
          className={cn(
            "h-9 w-full appearance-none rounded-lg border border-white/10 bg-surface px-3 pr-8 text-sm text-foreground focus:outline-none focus:ring-2 focus:ring-indigo-500",
            className
          )}
          {...props}
        >
          {children}
        </select>
        <ChevronDown
          className="pointer-events-none absolute right-2.5 top-1/2 h-4 w-4 -translate-y-1/2 text-muted-foreground"
          aria-hidden
        />
      </div>
    </div>
  );
}
