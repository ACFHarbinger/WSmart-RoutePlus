/**
 * CVD-safe (Deuteranopia, Protanopia, Tritanopia) scientific palettes for
 * ECharts and Deck.gl (Slice V1).
 *
 * Keyed strictly by real WSmart-Route simulation policy IDs from
 * `logic/src/policies/routing/_manifest.py` and `logic/configs/tasks/test_sim.yaml`.
 */

export interface PolicyPaletteEntry {
  id: string;
  label: string;
  family: "exact" | "matheuristic" | "metaheuristic" | "neural" | "heuristic";
  hex: string;
  rgb: [number, number, number];
}

export const NOC_POLICY_PALETTE: readonly PolicyPaletteEntry[] = [
  { id: "bpc",      label: "BPC",      family: "exact",         hex: "#38bdf8", rgb: [56, 189, 248] },
  { id: "hgs",      label: "HGS",      family: "metaheuristic", hex: "#f59e0b", rgb: [245, 158, 11] },
  { id: "alns",     label: "ALNS",     family: "metaheuristic", hex: "#818cf8", rgb: [129, 140, 248] },
  { id: "swc_tcf",  label: "SWC-TCF",  family: "exact",         hex: "#2dd4bf", rgb: [45, 212, 191] },
  { id: "pg_clns",  label: "PG-CLNS",  family: "metaheuristic", hex: "#fb923c", rgb: [251, 146, 60] },
  { id: "sans",     label: "SANS",     family: "metaheuristic", hex: "#f472b6", rgb: [244, 114, 182] },
  { id: "psoma",    label: "PSOMA",    family: "metaheuristic", hex: "#a3e635", rgb: [163, 230, 53] },
  { id: "aco_hh",   label: "ACO_HH",   family: "metaheuristic", hex: "#c084fc", rgb: [192, 132, 252] },
  { id: "na",       label: "NA",       family: "neural",        hex: "#4ade80", rgb: [74, 222, 128] },
  { id: "hna",      label: "HNA",      family: "neural",        hex: "#fb7185", rgb: [251, 113, 133] },
  { id: "lbbd",     label: "LBBD",     family: "exact",         hex: "#60a5fa", rgb: [96, 165, 250] },
  { id: "baseline", label: "Baseline", family: "heuristic",     hex: "#facc15", rgb: [250, 204, 21] },
] as const;

export const POLICY_HEX_PALETTE: string[] = NOC_POLICY_PALETTE.map((p) => p.hex);

const POLICY_MAP: Record<string, PolicyPaletteEntry> = Object.fromEntries(
  NOC_POLICY_PALETTE.map((p) => [p.id.toLowerCase(), p])
);

function paletteByToken(policy: string): PolicyPaletteEntry | undefined {
  const tokens = policy.toLowerCase().trim().split(/[^a-z0-9]+/).filter(Boolean);
  const byLength = [...NOC_POLICY_PALETTE].sort((a, b) => b.id.length - a.id.length);
  return byLength.find((entry) => tokens.includes(entry.id));
}

/** Resolve a policy identifier or compound variant label to its CVD-safe hex color. */
export function getPolicyColor(policy: string, fallbackIdx = 0): string {
  const norm = policy.toLowerCase().trim();
  return (
    POLICY_MAP[norm]?.hex ??
    paletteByToken(policy)?.hex ??
    NOC_POLICY_PALETTE[fallbackIdx % NOC_POLICY_PALETTE.length]!.hex
  );
}

/** Resolve a policy identifier to its Deck.gl `[r, g, b]` tuple. */
export function getPolicyRgb(policy: string, fallbackIdx = 0): [number, number, number] {
  const norm = policy.toLowerCase().trim();
  return (
    POLICY_MAP[norm]?.rgb ??
    paletteByToken(policy)?.rgb ??
    NOC_POLICY_PALETTE[fallbackIdx % NOC_POLICY_PALETTE.length]!.rgb
  );
}

/** ECharts visualMap piecewise configuration for bin fill levels (0 to >=100% overflow). */
export const ECHARTS_FILL_VISUAL_MAP = {
  type: "piecewise" as const,
  pieces: [
    { min: 0, max: 0.5, label: "0–50% (Nominal)", color: "#10b981" },
    { min: 0.5, max: 0.8, label: "50–80% (Elevated)", color: "#0284c7" },
    { min: 0.8, max: 0.999, label: "80–99% (Urgent)", color: "#f59e0b" },
    { min: 1.0, label: "≥100% (Overflow)", color: "#ef4444" },
  ],
};

/** Deck.gl RGBA tuple for bin fill fraction in `[0, 1.2+]`. */
export function getBinFillColor(fillRatio: number): [number, number, number, number] {
  if (fillRatio >= 1.0) return [239, 68, 68, 240];
  if (fillRatio >= 0.8) return [245, 158, 11, 220];
  if (fillRatio >= 0.5) return [2, 132, 199, 200];
  return [16, 185, 129, 180];
}

/** Diverging scale for profit / cost deltas (Cyan-Emerald positive vs Amber-Crimson negative). */
export const NOC_DIVERGING_DELTA = {
  positiveStrong: "#10b981",
  positiveMild: "#38bdf8",
  neutral: "#64748b",
  negativeMild: "#f59e0b",
  negativeStrong: "#ef4444",
} as const;

export function getDeltaColor(delta: number, higherIsBetter = true): string {
  if (Math.abs(delta) < 1e-6) return NOC_DIVERGING_DELTA.neutral;
  const isGood = higherIsBetter ? delta > 0 : delta < 0;
  return isGood ? NOC_DIVERGING_DELTA.positiveStrong : NOC_DIVERGING_DELTA.negativeStrong;
}
