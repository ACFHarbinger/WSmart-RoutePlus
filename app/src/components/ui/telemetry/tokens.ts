/**
 * Shared NOC & SetProduct Orion Design Tokens (Slice V1).
 *
 * Provides paired Dark and Light mode surface palettes, WCAG AA verified
 * text tokens, luminous chart stroke tokens, and contrast math utilities.
 */

export const NOC_COLORS = {
  ok: "#10b981",
  warn: "#eab308",
  crit: "#ef4444",
  idle: "#94a3b8",
  cyan: "#38bdf8",
  teal: "#06b6d4",
  indigo: "#6366f1",
  indigoText: "#818cf8",
  purple: "#a855f7",
  purpleText: "#c084fc",
  pink: "#ec4899",
  amber: "#f59e0b",
  orange: "#f97316",
  emerald: "#22c55e",
} as const;

export const NOC_LIGHT_TEXT_COLORS = {
  ok: "#047857",
  warn: "#a16207",
  crit: "#b91c1c",
  idle: "#475569",
  cyan: "#0369a1",
  teal: "#0e7490",
  indigo: "#4338ca",
  purple: "#6b21a8",
  pink: "#be185d",
  amber: "#b45309",
  orange: "#c2410c",
  emerald: "#15803d",
} as const;

export const NOC_SURFACES = {
  bgDeep: "#090e1a",
  bgPanel: "#0f172a",
  bgCard: "#1e293b",
  textPrimary: "#f8fafc",
  textSecondary: "#e2e8f0",
  textMuted: "#94a3b8",
  border: "rgba(148, 163, 184, 0.14)",
  borderStrong: "rgba(148, 163, 184, 0.28)",
  gridline: "rgba(148, 163, 184, 0.14)",
  focusRing: "rgba(56, 189, 248, 0.55)",
} as const;

export const NOC_LIGHT_SURFACES = {
  bgDeep: "#f8fafc",
  bgPanel: "#ffffff",
  bgCard: "#f1f5f9",
  textPrimary: "#0f172a",
  textSecondary: "#1e293b",
  textMuted: "#475569",
  border: "rgba(15, 23, 42, 0.12)",
  borderStrong: "rgba(15, 23, 42, 0.24)",
  gridline: "rgba(15, 23, 42, 0.10)",
  focusRing: "rgba(67, 56, 202, 0.45)",
} as const;

export type NocTone = "ok" | "warn" | "crit" | "idle" | "cyan" | "indigo" | "purple" | "amber";

/** Compute WCAG 2.1 relative luminance for a `#rrggbb` hex string. */
export function relativeLuminance(hex: string): number {
  const clean = hex.replace("#", "").trim();
  const r = parseInt(clean.slice(0, 2), 16) / 255;
  const g = parseInt(clean.slice(2, 4), 16) / 255;
  const b = parseInt(clean.slice(4, 6), 16) / 255;
  const toLinear = (c: number) => (c <= 0.03928 ? c / 12.92 : Math.pow((c + 0.055) / 1.055, 2.4));
  return 0.2126 * toLinear(r) + 0.7152 * toLinear(g) + 0.0722 * toLinear(b);
}

/** Compute WCAG 2.1 contrast ratio between two `#rrggbb` hex colors. */
export function contrastRatio(hexA: string, hexB: string): number {
  const l1 = relativeLuminance(hexA);
  const l2 = relativeLuminance(hexB);
  const lighter = Math.max(l1, l2);
  const darker = Math.min(l1, l2);
  return (lighter + 0.05) / (darker + 0.05);
}
