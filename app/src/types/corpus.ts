/** Simulation Summary corpus (§G.21, #175). Shared by the scanner, the strip, and the scenes. */

export const CORPUS_METRICS = [
  "profit",
  "kg",
  "km",
  "cost",
  "overflows",
  "kg_per_km",
] as const;

export type CorpusMetric = (typeof CORPUS_METRICS)[number];

export const CORPUS_METRIC_LABEL: Record<CorpusMetric, string> = {
  profit: "Profit",
  kg: "kg",
  km: "km",
  cost: "Cost",
  overflows: "Overflows",
  kg_per_km: "kg/km",
};

/** Lower is better for distance, cost, and overflows. */
export const CORPUS_LOWER_IS_BETTER: Record<CorpusMetric, boolean> = {
  profit: false,
  kg: false,
  km: true,
  cost: true,
  overflows: true,
  kg_per_km: false,
};

export interface CorpusFailure {
  path: string;
  reason: string;
}

export interface CorpusBin {
  binId: string;
  /** Absent when the log has no fill for this bin. */
  fill: number | null;
}

/** One day row inside a simulation JSON or JSONL file. Missing metrics stay absent. */
export interface CorpusDay {
  day: number | null;
  policy: string | null;
  city: string | null;
  metrics: Partial<Record<CorpusMetric, number>>;
  /** Waste left uncollected. Not one of the six chart metrics. */
  kgLost: number | null;
  lat: number | null;
  lon: number | null;
  bins: CorpusBin[];
}

export interface CorpusFileRecord {
  path: string;
  /** True when the user opened this file on its own. */
  partial: boolean;
  days: CorpusDay[];
}

export interface CorpusScanResult {
  roots: string[];
  files: CorpusFileRecord[];
  kept: number;
  skipped: number;
  failed: CorpusFailure[];
}

export interface CorpusRun {
  id: string;
  label: string;
  scenarioId: string;
  scenarioLabel: string;
  /** Parent directory of a full run. Empty for a partial file. */
  directory: string;
  partial: boolean;
  files: string[];
  hasCoordinates: boolean;
}

export interface CorpusHighlight {
  runId?: string;
  policy?: string;
  city?: string;
}

export interface CorpusFilters {
  /** Empty means every run that is not hidden. */
  selectedRunIds: string[];
  hiddenRunIds: string[];
  metric: CorpusMetric;
  dayMin: number | null;
  dayMax: number | null;
  /** Empty means every policy. */
  policies: string[];
  /** Empty means every city. */
  cities: string[];
  baselineMode: "run" | "value";
  pinnedRunId: string | null;
  typedBaseline: number | null;
}

export interface CorpusSeriesPoint {
  runId: string;
  runLabel: string;
  scenarioId: string;
  scenarioLabel: string;
  partial: boolean;
  day: number | null;
  policy: string | null;
  city: string | null;
  /** Selected metric. Null is a gap, not zero. */
  value: number | null;
  metrics: Partial<Record<CorpusMetric, number>>;
  kgLost: number | null;
  lat: number | null;
  lon: number | null;
  bins: CorpusBin[];
  hasCoordinates: boolean;
}

export interface CorpusView {
  runs: CorpusRun[];
  points: CorpusSeriesPoint[];
  baseline: number | null;
  baselineLabel: string;
  kept: number;
  skipped: number;
  failed: CorpusFailure[];
  /** True when no point carries bin rows. Scenes draw one dot per day and say so. */
  dayLevelOnly: boolean;
}

export interface GeneratedLayoutNode {
  id: string;
  label: string;
  x: number;
  y: number;
}
