/**
 * Pheromone accumulation and normalization for ACO topology routing (§G.4).
 */
import type { DayLogEntry } from "../../types";
import { pheromoneWeightDisplay } from "../charts/chartLogScale";
import { pheromoneEdgeKey } from "./graphTopologyLayout";

export function guiIdToMatrixIndex(guiId: number, nodeIds: number[]): number | null {
  if (guiId === -1) {
    const depotIdx = nodeIds.indexOf(0);
    return depotIdx >= 0 ? depotIdx : 0;
  }
  const binId = guiId;
  const idx = nodeIds.indexOf(binId);
  if (idx >= 0) return idx;
  return guiId >= 0 && guiId < nodeIds.length ? guiId : null;
}

/** Extract consecutive tour segments as undirected matrix-index pairs. */
export function tourIndicesToMatrixPairs(
  tourIndices: number[] | undefined,
  nodeIds: number[]
): Array<[number, number]> {
  if (!tourIndices?.length) return [];
  const indices = tourIndices
    .map((guiId) => guiIdToMatrixIndex(guiId, nodeIds))
    .filter((idx): idx is number => idx != null);
  const pairs: Array<[number, number]> = [];
  for (let i = 0; i < indices.length - 1; i++) {
    const a = indices[i];
    const b = indices[i + 1];
    if (a !== b) pairs.push([a, b]);
  }
  return pairs;
}

/** Count consecutive tour edge pairs for the given day entry. */
export function countTourEdgeSteps(
  entry: DayLogEntry | undefined,
  nodeIds: number[]
): number {
  if (!entry) return 0;
  return tourIndicesToMatrixPairs(entry.data.tour_indices, nodeIds).length;
}

/**
 * Deposit synthetic ACO τ along tour edges up to `upToStep` (per-iteration stepping).
 * Each ant step adds τ on the next consecutive tour edge; later steps deposit slightly more.
 */
export function accumulateTourPheromoneByStep(
  entry: DayLogEntry,
  nodeIds: number[],
  upToStep: number,
  opts: { depositScale?: number } = {}
): Map<string, number> {
  const pairs = tourIndicesToMatrixPairs(entry.data.tour_indices, nodeIds);
  const weights = new Map<string, number>();
  if (!pairs.length || upToStep <= 0) return weights;

  const scale =
    opts.depositScale ?? 1 + Math.max(0, (entry.data.profit ?? 0) / 500);
  const limit = Math.min(upToStep, pairs.length);

  for (let i = 0; i < limit; i++) {
    const [a, b] = pairs[i];
    const key = pheromoneEdgeKey(a, b);
    const stepFactor = 1 + (i / Math.max(pairs.length, 1)) * 0.5;
    weights.set(key, (weights.get(key) ?? 0) + scale * stepFactor);
  }

  return weights;
}

/**
 * Accumulate synthetic ACO pheromone from tour edges up to `upToDay`.
 * Older days decay exponentially; deposit scales with route profit when present.
 */
export function accumulateTourPheromone(
  entries: DayLogEntry[],
  nodeIds: number[],
  upToDay: number,
  opts: { decay?: number; minDay?: number } = {}
): Map<string, number> {
  const { decay = 0.88, minDay = 1 } = opts;
  const weights = new Map<string, number>();
  const relevant = entries
    .filter((e) => e.day >= minDay && e.day <= upToDay)
    .sort((a, b) => a.day - b.day);

  if (!relevant.length) return weights;

  const maxDay = Math.max(...relevant.map((e) => e.day));
  for (const entry of relevant) {
    const age = maxDay - entry.day;
    const depositScale = 1 + Math.max(0, (entry.data.profit ?? 0) / 500);
    const dayFactor = Math.pow(decay, age) * depositScale;
    for (const [a, b] of tourIndicesToMatrixPairs(entry.data.tour_indices, nodeIds)) {
      const key = pheromoneEdgeKey(a, b);
      weights.set(key, (weights.get(key) ?? 0) + dayFactor);
    }
  }

  return weights;
}

export function normalizePheromone(
  weights: Map<string, number>,
  logScale = false
): Map<string, number> {
  if (!weights.size) return weights;
  let max = 0;
  const scaled = new Map<string, number>();
  for (const [k, v] of weights) {
    const t = logScale ? pheromoneWeightDisplay(v, true) : v;
    scaled.set(k, t);
    max = Math.max(max, t);
  }
  if (max <= 0) return weights;
  const out = new Map<string, number>();
  for (const [k, v] of scaled) out.set(k, v / max);
  return out;
}

/** Normalized τ intensity in [0, 1] for edge styling across renderers. */
export function pheromoneIntensity(
  weights: Map<string, number>,
  key: string,
  logScale = false
): number {
  return normalizePheromone(weights, logScale).get(key) ?? 0;
}
