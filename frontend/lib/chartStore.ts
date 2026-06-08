import { create } from "zustand";

interface ChartState {
  /** ISO date string "YYYY-MM-DD" of the bar under the lightweight-charts crosshair,
   *  or null when the crosshair is outside the chart area. */
  hoveredDate: string | null;
  setHoveredDate: (date: string | null) => void;
}

export const useChartStore = create<ChartState>((set) => ({
  hoveredDate: null,
  setHoveredDate: (date) => set({ hoveredDate: date }),
}));
