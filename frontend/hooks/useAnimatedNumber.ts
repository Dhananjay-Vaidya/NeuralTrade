"use client";

import { useEffect, useRef } from "react";
import { useMotionValue, animate, useTransform, type MotionValue } from "framer-motion";
import { duration, ease } from "@/lib/motion";

/** Returns a MotionValue<string> that tweens from the previous value to `target`
 *  whenever `target` changes.  Snaps immediately when prefers-reduced-motion is set.
 *
 *  @param target   The raw number to animate toward.
 *  @param format   Formatter applied to the interpolated value on every frame.
 */
export function useAnimatedNumber(
  target: number,
  format: (v: number) => string,
): MotionValue<string> {
  const motionVal = useMotionValue(target);

  // Read prefers-reduced-motion once on mount; no need to resubscribe since
  // the OS setting rarely changes mid-session and the tween is cosmetic.
  const reducedRef = useRef(false);
  useEffect(() => {
    reducedRef.current =
      typeof window !== "undefined" &&
      window.matchMedia("(prefers-reduced-motion: reduce)").matches;
  }, []);

  useEffect(() => {
    if (reducedRef.current) {
      motionVal.set(target);
      return;
    }
    const ctrl = animate(motionVal, target, {
      duration: duration.slow,
      ease: ease.out,
    });
    return () => ctrl.stop();
  }, [target, motionVal]);

  // useTransform is stable — it derives a new MotionValue from motionVal
  // without creating a new subscription on every render.
  return useTransform(motionVal, format);
}
