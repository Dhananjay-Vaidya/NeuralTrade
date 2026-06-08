import { clsx, type ClassValue } from "clsx";
import { twMerge } from "tailwind-merge";

export function cn(...inputs: ClassValue[]) {
  return twMerge(clsx(inputs));
}

export function fmt(value: number, style: "percent" | "decimal" | "currency" = "decimal", digits = 2): string {
  if (!isFinite(value)) return "—";
  if (style === "percent") {
    const sign = value >= 0 ? "+" : "";
    return `${sign}${(value * 100).toFixed(digits)}%`;
  }
  if (style === "currency") {
    return new Intl.NumberFormat("en-US", { style: "currency", currency: "USD", maximumFractionDigits: digits }).format(value);
  }
  return value.toFixed(digits);
}
