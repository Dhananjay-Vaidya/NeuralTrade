/**
 * Motion system tokens and reusable Framer Motion variants.
 *
 * Rules:
 *  - All animated components must import durations/easing from here.
 *  - No ad-hoc `transition` objects with inline numbers anywhere in the codebase.
 *  - Use the variant objects directly; override only when semantically justified.
 */

import type { Variants } from "framer-motion";

// ── Easing ─────────────────────────────────────────────────────────────────────

/** Single project-wide easing curve: decelerate into final position. */
export const ease = {
  out: [0.0, 0.0, 0.2, 1.0] as [number, number, number, number],
} as const;

// ── Duration scale (ms → seconds for Framer Motion) ───────────────────────────

export const duration = {
  fast: 0.12,
  base: 0.2,
  slow: 0.32,
} as const;

// ── Shared transition presets ──────────────────────────────────────────────────

export const transition = {
  fast: { duration: duration.fast, ease: ease.out },
  base: { duration: duration.base, ease: ease.out },
  slow: { duration: duration.slow, ease: ease.out },
} as const;

// ── Reusable Framer Motion variants ───────────────────────────────────────────

/** Fade in from transparent, fade out to transparent. */
export const fadeVariants: Variants = {
  hidden: { opacity: 0 },
  visible: { opacity: 1, transition: transition.base },
  exit:    { opacity: 0, transition: transition.fast },
};

/** Slide up and fade in; slide down and fade out. */
export const slideUpVariants: Variants = {
  hidden:  { opacity: 0, y: 8 },
  visible: { opacity: 1, y: 0, transition: transition.base },
  exit:    { opacity: 0, y: 4, transition: transition.fast },
};

/** Scale up from 95% and fade in. Useful for cards and modals. */
export const scaleInVariants: Variants = {
  hidden:  { opacity: 0, scale: 0.95 },
  visible: { opacity: 1, scale: 1, transition: transition.slow },
  exit:    { opacity: 0, scale: 0.97, transition: transition.fast },
};

/** Stagger container — wraps a list; children animate sequentially. */
export const staggerContainerVariants: Variants = {
  hidden:  {},
  visible: { transition: { staggerChildren: 0.06, delayChildren: 0 } },
};

/** List item — pair with staggerContainerVariants on the parent. */
export const listItemVariants: Variants = {
  hidden:  { opacity: 0, y: 6 },
  visible: { opacity: 1, y: 0, transition: transition.base },
};
