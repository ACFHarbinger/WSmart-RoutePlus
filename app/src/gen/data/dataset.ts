/**
 * Dataset statistics engine (§H.1) — native port of the data layer of
 * `logic/gen/gen_dataset_analysis.py`.
 *
 * NPZ/TD statistics CSV loading, raw NPZ waste matrices (via the Rust
 * `load_npz_flat` reader), extended statistics (median, variance, quartiles,
 * IQR, fences, binned mode), Gaussian KDE + histogram binning, and the three
 * markdown table builders.
 */
import { DATASET_CFG } from "../config";
import { joinPath, listFilesRecursive, loadCsv, loadNpzFlat } from "../io";

export interface NpzStatRow {
  city: string;
  N: number;
  dist: string;
  horizon: number;
  mean_kg: number;
  std_kg: number;
  max_kg: number;
  overflow_pct: number;
  skewness: number;
}

export interface TdStatRow {
  N: number;
  dist: string;
  instances: number;
  waste_mean: number;
  waste_std: number;
  waste_skew: number;
}

export interface ExtendedStats {
  median: number;
  variance: number;
  q1: number;
  q3: number;
  iqr: number;
  min: number;
  lower_fence: number;
  upper_fence: number;
  mode: number;
}

export interface ExtendedRow extends ExtendedStats {
  city: string;
  N: number;
  dist: string;
}

export type RawWaste = Map<string, { city: string; N: number; dist: string; values: number[] }>;

const num = (v: unknown) => {
  const n = typeof v === "number" ? v : parseFloat(String(v ?? ""));
  return Number.isFinite(n) ? n : NaN;
};

export async function loadNpzStats(projectRoot: string, csvRel: string): Promise<NpzStatRow[]> {
  const file = await loadCsv(joinPath(projectRoot, csvRel));
  return file.rows.map((r) => ({
    city: String(r.city ?? ""),
    N: num(r.N),
    dist: String(r.dist ?? ""),
    horizon: num(r.horizon),
    mean_kg: num(r.mean_kg),
    std_kg: num(r.std_kg),
    max_kg: num(r.max_kg),
    overflow_pct: num(r.overflow_pct),
    skewness: num(r.skewness),
  }));
}

export async function loadTdStats(projectRoot: string, csvRel: string): Promise<TdStatRow[]> {
  const file = await loadCsv(joinPath(projectRoot, csvRel));
  return file.rows.map((r) => ({
    N: num(r.N),
    dist: String(r.dist ?? ""),
    instances: num(r.instances),
    waste_mean: num(r.waste_mean),
    waste_std: num(r.waste_std),
    waste_skew: num(r.waste_skew),
  }));
}

// ── Extended statistics (ports extended_stats) ───────────────────────────────

export function percentile(sorted: number[], p: number): number {
  if (!sorted.length) return NaN;
  const idx = (p / 100) * (sorted.length - 1);
  const lo = Math.floor(idx);
  const hi = Math.ceil(idx);
  if (lo === hi) return sorted[lo];
  return sorted[lo] + (sorted[hi] - sorted[lo]) * (idx - lo);
}

export function extendedStats(values: number[]): ExtendedStats {
  const sorted = [...values].sort((a, b) => a - b);
  const q1 = percentile(sorted, 25);
  const med = percentile(sorted, 50);
  const q3 = percentile(sorted, 75);
  const iqr = q3 - q1;
  const m = sorted.reduce((a, b) => a + b, 0) / sorted.length;
  const variance = sorted.reduce((a, b) => a + (b - m) ** 2, 0) / sorted.length;
  const { counts, edges } = histogram(sorted, 50);
  let maxBin = 0;
  for (let i = 1; i < counts.length; i++) if (counts[i] > counts[maxBin]) maxBin = i;
  return {
    median: med,
    variance,
    q1,
    q3,
    iqr,
    min: sorted[0],
    lower_fence: q1 - 1.5 * iqr,
    upper_fence: q3 + 1.5 * iqr,
    mode: (edges[maxBin] + edges[maxBin + 1]) / 2,
  };
}

// ── Histogram + Gaussian KDE (for violin / hist+KDE charts) ──────────────────

export function histogram(values: number[], bins: number): { counts: number[]; edges: number[] } {
  const min = Math.min(...values);
  const max = Math.max(...values);
  const width = (max - min) / bins || 1;
  const counts = new Array<number>(bins).fill(0);
  const edges = [...Array(bins + 1).keys()].map((i) => min + i * width);
  for (const v of values) {
    let b = Math.floor((v - min) / width);
    if (b >= bins) b = bins - 1;
    if (b < 0) b = 0;
    counts[b]++;
  }
  return { counts, edges };
}

/** Gaussian KDE evaluated on a uniform grid (Scott's rule bandwidth). */
export function gaussianKde(
  values: number[],
  gridSize = 120,
  lo?: number,
  hi?: number
): { x: number[]; y: number[] } {
  const n = values.length;
  if (!n) return { x: [], y: [] };
  const m = values.reduce((a, b) => a + b, 0) / n;
  const sd = Math.sqrt(values.reduce((a, b) => a + (b - m) ** 2, 0) / n) || 1;
  const bw = 1.06 * sd * Math.pow(n, -1 / 5) || 1;
  const min = lo ?? Math.min(...values) - 2 * bw;
  const max = hi ?? Math.max(...values) + 2 * bw;
  const x: number[] = [];
  const y: number[] = [];
  const norm = 1 / (n * bw * Math.sqrt(2 * Math.PI));
  // Subsample very large arrays for tractable KDE evaluation
  const sample = n > 20000 ? values.filter((_, i) => i % Math.ceil(n / 20000) === 0) : values;
  const sNorm = 1 / (sample.length * bw * Math.sqrt(2 * Math.PI));
  void norm;
  for (let i = 0; i < gridSize; i++) {
    const xi = min + ((max - min) * i) / (gridSize - 1);
    let acc = 0;
    for (const v of sample) {
      const z = (xi - v) / bw;
      acc += Math.exp(-0.5 * z * z);
    }
    x.push(xi);
    y.push(acc * sNorm);
  }
  return { x, y };
}

// ── Raw NPZ loading (ports load_raw_waste) ───────────────────────────────────

export async function loadRawWaste(
  projectRoot: string,
  npzDirRel: string,
  npz: NpzStatRow[],
  horizon = 30
): Promise<RawWaste> {
  const raw: RawWaste = new Map();
  const npzDir = joinPath(projectRoot, npzDirRel);
  const allNpz = await listFilesRecursive(npzDir, { suffix: ".npz" });
  for (const row of npz.filter((r) => r.horizon === horizon)) {
    const stemPrefix = `${row.city}${row.N}_${row.dist}_wsr${horizon}_`;
    const match = allNpz.find((p) => p.split(/[\\/]/).pop()!.startsWith(stemPrefix));
    if (!match) continue;
    try {
      const values = await loadNpzFlat(match, "waste");
      raw.set(`${row.city}|${row.N}|${row.dist}`, { city: row.city, N: row.N, dist: row.dist, values });
    } catch {
      // defensive: skip unreadable archives (matches the Python warning path)
    }
  }
  return raw;
}

export function buildExtendedRows(raw: RawWaste): ExtendedRow[] {
  const rows: ExtendedRow[] = [];
  for (const { city, N, dist, values } of [...raw.values()].sort(
    (a, b) => a.city.localeCompare(b.city) || a.N - b.N || a.dist.localeCompare(b.dist)
  )) {
    rows.push({ city, N, dist, ...extendedStats(values) });
  }
  return rows;
}

// ── Markdown table builders (port build_npz_table / extended / td) ───────────

const CITY_LABELS = DATASET_CFG.city_labels;
const DIST_LABELS = DATASET_CFG.dist_labels;

export function buildNpzTable(npz: NpzStatRow[], ext: ExtendedRow[], horizon = 30): string {
  const extKey = new Map(ext.map((e) => [`${e.city}|${e.N}|${e.dist}`, e]));
  const rows = [
    "| City | N | Distribution | Mean kg | Median kg | Std kg | Max kg | IQR kg | Skewness |",
    "|------|---|-------------|---------|-----------|--------|--------|--------|---------|",
  ];
  const sub = npz
    .filter((r) => r.horizon === horizon)
    .sort((a, b) => a.city.localeCompare(b.city) || a.N - b.N || a.dist.localeCompare(b.dist));
  for (const r of sub) {
    const e = extKey.get(`${r.city}|${r.N}|${r.dist}`);
    const med = e ? e.median.toFixed(2) : "—";
    const iqr = e ? e.iqr.toFixed(2) : "—";
    rows.push(
      `| ${CITY_LABELS[r.city] ?? r.city} | ${r.N} | ${DIST_LABELS[r.dist] ?? r.dist} | ` +
        `${r.mean_kg.toFixed(2)} | ${med} | ${r.std_kg.toFixed(2)} | ${r.max_kg.toFixed(1)} | ` +
        `${iqr} | ${r.skewness.toFixed(3)} |`
    );
  }
  return rows.join("\n");
}

export function buildExtendedTable(ext: ExtendedRow[]): string {
  if (!ext.length) return "_Raw NPZ data unavailable — extended statistics could not be computed._";
  const rows = [
    "| City | N | Distribution | Median | Variance | Q1 | Q3 | IQR | Min | Fences (lo/hi) | Mode |",
    "|------|---|-------------|--------|----------|----|----|-----|-----|----------------|------|",
  ];
  for (const r of [...ext].sort(
    (a, b) => a.city.localeCompare(b.city) || a.N - b.N || a.dist.localeCompare(b.dist)
  )) {
    rows.push(
      `| ${CITY_LABELS[r.city] ?? r.city} | ${r.N} | ${DIST_LABELS[r.dist] ?? r.dist} | ` +
        `${r.median.toFixed(2)} | ${r.variance.toFixed(2)} | ${r.q1.toFixed(2)} | ${r.q3.toFixed(2)} | ` +
        `${r.iqr.toFixed(2)} | ${r.min.toFixed(2)} | ${Math.max(r.lower_fence, 0).toFixed(2)} / ` +
        `${r.upper_fence.toFixed(2)} | ${r.mode.toFixed(2)} |`
    );
  }
  return rows.join("\n");
}

export function buildTdTable(td: TdStatRow[]): string {
  const rows = [
    "| N | Distribution | Instances | Mean Waste | Std Waste | Skewness |",
    "|---|-------------|-----------|------------|-----------|---------|",
  ];
  for (const r of [...td].sort((a, b) => a.N - b.N || a.dist.localeCompare(b.dist))) {
    rows.push(
      `| ${r.N} | ${DIST_LABELS[r.dist] ?? r.dist} | ${Math.round(r.instances).toLocaleString("en-US")} | ` +
        `${r.waste_mean.toFixed(4)} | ${r.waste_std.toFixed(4)} | ${r.waste_skew.toFixed(3)} |`
    );
  }
  return rows.join("\n");
}
