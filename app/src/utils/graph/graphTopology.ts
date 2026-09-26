/**
 * Topological graph analytics — distance matrix → edge list + force layout (§G.4).
 */
import { invoke } from "@tauri-apps/api/core";
import type { DayLogEntry, SimDayData } from "../../types";
import { pheromoneWeightDisplay } from "../charts/chartLogScale";
import { parseLogPath } from "../sim/simMetadata";

export type TopologyLayoutMode = "force" | "radial";

/** Auto-select radial layout for dense graphs (Cosmograph-style fallback). */
export const DENSE_GRAPH_NODE_THRESHOLD = 200;

export interface DistanceMatrixData {
  nodeIds: number[];
  distances: number[][];
}

export interface GraphEdge {
  source: number;
  target: number;
  distance: number;
}

export interface TopologyNodeMeta {
  matrixIndex: number;
  nodeId: number;
  fillPct: number | null;
  onTour: boolean;
  mandatory: boolean;
  collected: number;
}

export interface TopologyBuildOptions {
  kNeighbors?: number;
  maxEdges?: number;
  seedPositions?: Map<number, [number, number]>;
  layoutIterations?: number;
  relayoutOnFilter?: boolean;
  fillRange?: [number, number] | null;
  tourIndices?: number[];
  binIdByMatrixIndex?: Map<number, number>;
  layoutMode?: TopologyLayoutMode | "auto";
  pheromoneWeights?: Map<string, number>;
  showPheromone?: boolean;
  logScale?: boolean;
  theme?: "dark" | "light";
}

function joinPath(root: string, rel: string): string {
  const base = root.replace(/[/\\]+$/, "");
  return [base, ...rel.split("/")].join("/");
}

/** Resolve distance-matrix CSV beside the log or under project data/. */
export function resolveDistanceMatrixCandidates(
  logPath: string,
  projectRoot?: string | null
): string[] {
  const candidates: string[] = [];
  const dir = logPath.replace(/[/\\][^/\\]+$/, "");
  candidates.push(`${dir}/gmaps_distmat.csv`);

  if (projectRoot) {
    const meta = parseLogPath(logPath);
    const areaKey =
      meta.city === "Figueira da Foz"
        ? "figdafoz"
        : meta.city === "Rio Maior"
          ? "riomaior"
          : null;
    if (areaKey) {
      candidates.push(
        joinPath(
          projectRoot,
          `data/simulator/distance_matrix/gmaps_distmat_plastic[${areaKey}].csv`
        )
      );
    }
  }
  return [...new Set(candidates)];
}

/** Parse gmaps distance-matrix CSV: header row = node IDs, following rows = distances. */
export function parseDistanceMatrixCsv(text: string): DistanceMatrixData {
  const lines = text
    .split(/\r?\n/)
    .map((l) => l.trim())
    .filter(Boolean);
  if (lines.length < 2) throw new Error("Distance matrix CSV is empty");

  const nodeIds = lines[0].split(",").map((v) => Number(v.trim()));
  if (nodeIds.some((id) => Number.isNaN(id))) {
    throw new Error("Invalid node IDs in distance matrix header");
  }

  const n = nodeIds.length;
  const distances: number[][] = [];
  for (let i = 1; i < lines.length && distances.length < n; i++) {
    const vals = lines[i].split(",").map((v) => Number(v.trim()));
    if (vals.length < n) continue;
    distances.push(vals.slice(0, n));
  }
  if (distances.length !== n) {
    throw new Error(`Expected ${n} distance rows, got ${distances.length}`);
  }
  return { nodeIds, distances };
}

export async function loadDistanceMatrix(path: string): Promise<DistanceMatrixData> {
  const text = await invoke<string>("read_text_file", { path });
  return parseDistanceMatrixCsv(text);
}

export async function loadDistanceMatrixForLog(
  logPath: string,
  projectRoot?: string | null
): Promise<{ data: DistanceMatrixData; path: string }> {
  const candidates = resolveDistanceMatrixCandidates(logPath, projectRoot);
  let lastErr: unknown = null;
  for (const path of candidates) {
    try {
      const data = await loadDistanceMatrix(path);
      return { data, path };
    } catch (err) {
      lastErr = err;
    }
  }
  throw lastErr ?? new Error("No distance matrix found for log");
}

/** k-nearest-neighbor edge list (undirected, deduplicated). */
export function buildKnnEdgeList(
  distances: number[][],
  k = 3,
  maxEdges = 1200
): GraphEdge[] {
  const n = distances.length;
  const edgeMap = new Map<string, GraphEdge>();

  for (let i = 0; i < n; i++) {
    const neighbors = distances[i]
      .map((d, j) => ({ j, d }))
      .filter(({ j, d }) => j !== i && Number.isFinite(d) && d > 0)
      .sort((a, b) => a.d - b.d)
      .slice(0, k);

    for (const { j, d } of neighbors) {
      const a = Math.min(i, j);
      const b = Math.max(i, j);
      const key = `${a}-${b}`;
      const existing = edgeMap.get(key);
      if (!existing || d < existing.distance) {
        edgeMap.set(key, { source: a, target: b, distance: d });
      }
      if (edgeMap.size >= maxEdges) break;
    }
    if (edgeMap.size >= maxEdges) break;
  }

  return [...edgeMap.values()];
}

/** Simple force-directed layout (Fruchterman-Reingold style). */
export function forceDirectedLayout(
  nodeCount: number,
  edges: GraphEdge[],
  seedPositions?: Map<number, [number, number]>,
  iterations = 80
): [number, number][] {
  const area = Math.max(400, nodeCount * 12);
  const pos: [number, number][] = Array.from({ length: nodeCount }, (_, i) => {
    const seed = seedPositions?.get(i);
    if (seed) return [...seed] as [number, number];
    const angle = (2 * Math.PI * i) / Math.max(nodeCount, 1);
    return [Math.cos(angle) * area * 0.35, Math.sin(angle) * area * 0.35];
  });

  const k = Math.sqrt((area * area) / Math.max(nodeCount, 1));

  for (let iter = 0; iter < iterations; iter++) {
    const disp: [number, number][] = pos.map(() => [0, 0]);
    const temp = area * (1 - iter / iterations);

    for (let i = 0; i < nodeCount; i++) {
      for (let j = i + 1; j < nodeCount; j++) {
        let dx = pos[i][0] - pos[j][0];
        let dy = pos[i][1] - pos[j][1];
        let dist = Math.hypot(dx, dy);
        if (dist < 1e-6) {
          dx = (Math.random() - 0.5) * 0.01;
          dy = (Math.random() - 0.5) * 0.01;
          dist = Math.hypot(dx, dy);
        }
        const repulse = (k * k) / dist;
        const fx = (dx / dist) * repulse;
        const fy = (dy / dist) * repulse;
        disp[i][0] += fx;
        disp[i][1] += fy;
        disp[j][0] -= fx;
        disp[j][1] -= fy;
      }
    }

    for (const { source, target, distance } of edges) {
      let dx = pos[source][0] - pos[target][0];
      let dy = pos[source][1] - pos[target][1];
      let dist = Math.hypot(dx, dy);
      if (dist < 1e-6) dist = 1e-6;
      const attract = (dist * dist) / Math.max(distance, 1e-3);
      const fx = (dx / dist) * attract;
      const fy = (dy / dist) * attract;
      disp[source][0] -= fx;
      disp[source][1] -= fy;
      disp[target][0] += fx;
      disp[target][1] += fy;
    }

    for (let i = 0; i < nodeCount; i++) {
      let dx = disp[i][0];
      let dy = disp[i][1];
      const dist = Math.hypot(dx, dy);
      if (dist > 0) {
        const scale = Math.min(dist, temp) / dist;
        dx *= scale;
        dy *= scale;
      }
      pos[i][0] += dx;
      pos[i][1] += dy;
    }
  }

  return pos;
}

/** Radial dense layout — depot at centre, bins on concentric rings (§G.4 Cosmograph fallback). */
export function radialDenseLayout(
  nodeCount: number,
  nodeIds: number[]
): [number, number][] {
  const area = Math.max(320, nodeCount * 10);
  const pos: [number, number][] = Array.from({ length: nodeCount }, () => [0, 0]);

  const depotIdx = nodeIds.indexOf(0);
  if (depotIdx >= 0) pos[depotIdx] = [0, 0];

  const ringNodes = nodeIds
    .map((id, idx) => ({ id, idx }))
    .filter(({ idx }) => idx !== depotIdx);
  const rings = nodeCount >= DENSE_GRAPH_NODE_THRESHOLD ? 3 : 2;
  const perRing = Math.ceil(ringNodes.length / rings);

  ringNodes.forEach(({ idx }, order) => {
    const ring = Math.floor(order / Math.max(perRing, 1));
    const slot = order % Math.max(perRing, 1);
    const ringCount = Math.min(perRing, ringNodes.length - ring * perRing);
    const radius = area * (0.28 + (ring / Math.max(rings - 1, 1)) * 0.42);
    const angle = (2 * Math.PI * slot) / Math.max(ringCount, 1);
    pos[idx] = [Math.cos(angle) * radius, Math.sin(angle) * radius];
  });

  return pos;
}

export function resolveLayoutMode(
  nodeCount: number,
  mode: TopologyLayoutMode | "auto" = "auto"
): TopologyLayoutMode {
  if (mode === "auto") {
    return nodeCount >= DENSE_GRAPH_NODE_THRESHOLD ? "radial" : "force";
  }
  return mode;
}

export function pheromoneEdgeKey(a: number, b: number): string {
  const lo = Math.min(a, b);
  const hi = Math.max(a, b);
  return `${lo}-${hi}`;
}

function guiIdToMatrixIndex(guiId: number, nodeIds: number[]): number | null {
  if (guiId === -1) {
    const depotIdx = nodeIds.indexOf(0);
    return depotIdx >= 0 ? depotIdx : 0;
  }
  const binId = guiId;
  const idx = nodeIds.indexOf(binId);
  if (idx >= 0) return idx;
  return guiId >= 0 && guiId < nodeIds.length ? guiId : null;
}

/** Map simulation day data onto matrix node indices. */
export function buildNodeMetaFromSim(
  data: SimDayData | null,
  nodeIds: number[]
): TopologyNodeMeta[] {
  const tourSet = new Set(data?.tour_indices ?? []);
  const mandatorySet = new Set(data?.mandatory ?? []);
  const collected = data?.bin_state_collected ?? [];
  const fills = data?.bin_state_c ?? [];
  const bins = data?.all_bin_coords ?? [];

  const fillByGuiId = new Map<number, number>();
  bins.forEach((bin, idx) => {
    if (fills[idx] != null) fillByGuiId.set(bin.id, fills[idx]);
  });

  return nodeIds.map((nodeId, matrixIndex) => {
    let guiId = nodeId;
    if (matrixIndex === 0 && nodeId === 0) guiId = -1;
    else if (matrixIndex > 0) {
      const match = bins.find((b) => b.dataset_id === nodeId || b.id === nodeId - 1);
      if (match) guiId = match.id;
    }

    const onTour = [...tourSet].some((tid) => guiIdToMatrixIndex(tid, nodeIds) === matrixIndex);
    const fillPct = fillByGuiId.get(guiId) ?? null;

    return {
      matrixIndex,
      nodeId,
      fillPct,
      onTour,
      mandatory: mandatorySet.has(guiId),
      collected: collected[bins.findIndex((b) => b.id === guiId)] ?? false,
    };
  });
}

function edgeWidth(distance: number, minD: number, maxD: number): number {
  if (!Number.isFinite(distance) || distance <= 0) return 0.5;
  const t = (distance - minD) / Math.max(maxD - minD, 1e-6);
  return 0.5 + (1 - t) * 3.5;
}

function nodeSize(meta: TopologyNodeMeta): number {
  const fill = meta.fillPct ?? (meta.onTour ? 55 : 25);
  return 6 + (fill / 100) * 14;
}

function nodeColor(
  meta: TopologyNodeMeta,
  fillRange: [number, number] | null,
  dimmed: boolean
): string {
  if (dimmed) return "#4b5563";
  if (meta.onTour) return "#34d399";
  if (meta.mandatory) return "#f87171";
  if (meta.fillPct != null && meta.fillPct >= 100) return "#fbbf24";
  if (fillRange && meta.fillPct != null) {
    const [lo, hi] = fillRange;
    if (meta.fillPct >= lo && meta.fillPct <= hi) return "#6366f1";
  }
  return "#94a3b8";
}

/** Shared node styling for ECharts and Sigma.js topology views (§G.4). */
export function topologyNodeStyle(
  meta: TopologyNodeMeta,
  fillRange: [number, number] | null
): { size: number; color: string; dimmed: boolean } {
  const hasFilter = fillRange != null && (fillRange[0] > 0 || fillRange[1] < 100);
  const highlighted =
    !hasFilter ||
    meta.fillPct == null ||
    (meta.fillPct >= fillRange![0] && meta.fillPct <= fillRange![1]);
  return {
    size: nodeSize(meta),
    color: nodeColor(meta, fillRange, !highlighted),
    dimmed: !highlighted,
  };
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

function pheromoneEdgeStyle(
  weight: number,
  baseWidth: number,
  dimmed: boolean
): { width: number; opacity: number; color: string } {
  if (dimmed) {
    return { width: baseWidth * 0.5, opacity: 0.06, color: "#374151" };
  }
  const t = Math.min(1, Math.max(0, weight));
  const opacity = 0.12 + t * 0.78;
  const r = Math.round(55 + t * 200);
  const g = Math.round(65 + t * 120);
  const b = Math.round(81 - t * 50);
  return {
    width: baseWidth + t * 2.5,
    opacity,
    color: `rgb(${r},${g},${b})`,
  };
}

export function buildTopologyGraphOption(
  nodeMeta: TopologyNodeMeta[],
  edges: GraphEdge[],
  positions: [number, number][],
  opts: {
    fillRange?: [number, number] | null;
    title?: string;
    theme?: "dark" | "light";
    pheromoneWeights?: Map<string, number>;
    showPheromone?: boolean;
    logScale?: boolean;
  } = {}
): Record<string, unknown> {
  const {
    fillRange = null,
    title,
    theme = "dark",
    pheromoneWeights,
    showPheromone = false,
    logScale = false,
  } = opts;
  const hasFilter = fillRange != null && (fillRange[0] > 0 || fillRange[1] < 100);
  const pheromone = showPheromone
    ? normalizePheromone(pheromoneWeights ?? new Map(), logScale)
    : null;

  const dists = edges.map((e) => e.distance);
  const minD = dists.length ? Math.min(...dists) : 0;
  const maxD = dists.length ? Math.max(...dists) : 1;

  const nodes = nodeMeta.map((meta) => {
    const [x, y] = positions[meta.matrixIndex] ?? [0, 0];
    const highlighted =
      !hasFilter ||
      meta.fillPct == null ||
      (meta.fillPct >= fillRange![0] && meta.fillPct <= fillRange![1]);
    return {
      id: String(meta.matrixIndex),
      name: meta.nodeId === 0 ? "Depot" : `#${meta.nodeId}`,
      x,
      y,
      symbolSize: nodeSize(meta),
      itemStyle: {
        color: nodeColor(meta, fillRange, !highlighted),
        opacity: highlighted ? 1 : 0.25,
        borderColor: meta.onTour ? "#a7f3d0" : undefined,
        borderWidth: meta.onTour ? 2 : 0,
      },
      label: { show: meta.matrixIndex === 0, fontSize: 9, color: theme === "dark" ? "#e5e7eb" : "#374151" },
    };
  });

  const links = edges.map((e) => {
    const srcMeta = nodeMeta[e.source];
    const tgtMeta = nodeMeta[e.target];
    const srcHi =
      !hasFilter ||
      srcMeta.fillPct == null ||
      (srcMeta.fillPct >= fillRange![0] && srcMeta.fillPct <= fillRange![1]);
    const tgtHi =
      !hasFilter ||
      tgtMeta.fillPct == null ||
      (tgtMeta.fillPct >= fillRange![0] && tgtMeta.fillPct <= fillRange![1]);
    const dimmed = hasFilter && !(srcHi && tgtHi);
    const baseWidth = edgeWidth(e.distance, minD, maxD);
    const phKey = pheromoneEdgeKey(e.source, e.target);
    const ph = pheromone?.get(phKey) ?? 0;
    const phStyle =
      pheromone && ph > 0
        ? pheromoneEdgeStyle(ph, baseWidth, dimmed)
        : {
            width: baseWidth,
            opacity: dimmed ? 0.08 : 0.35,
            color: dimmed ? "#374151" : "#64748b",
          };
    return {
      source: String(e.source),
      target: String(e.target),
      pheromone: ph,
      lineStyle: phStyle,
    };
  });

  return {
    backgroundColor: "transparent",
    title: title
      ? { text: title, left: "center", top: 4, textStyle: { fontSize: 11, color: theme === "dark" ? "#9ca3af" : "#6b7280" } }
      : undefined,
    tooltip: {
      formatter: (p: { dataType?: string; data?: { name?: string }; value?: unknown }) => {
        if (p.dataType === "node" && p.data?.name) {
          const idx = Number((p.data as { id?: string }).id);
          const meta = nodeMeta[idx];
          if (!meta) return p.data.name;
          const bits = [
            meta.fillPct != null ? `Fill ${meta.fillPct.toFixed(0)}%` : null,
            meta.onTour ? "On tour" : null,
            meta.mandatory ? "Mandatory" : null,
          ].filter(Boolean);
          return `${p.data.name}${bits.length ? `<br/>${bits.join(" · ")}` : ""}`;
        }
        if (p.dataType === "edge") {
          const ph = (p.data as { pheromone?: number })?.pheromone ?? 0;
          if (ph > 0) return `Pheromone τ = ${ph.toFixed(2)}`;
        }
        return "";
      },
    },
    series: [
      {
        type: "graph",
        layout: "none",
        data: nodes,
        links,
        roam: true,
        draggable: true,
        emphasis: { focus: "adjacency", lineStyle: { width: 6 } },
        lineStyle: { curveness: 0.12 },
      },
    ],
  };
}

/** Build full topology graph payload from matrix + simulation day. */
export function buildTopologyFromMatrix(
  matrix: DistanceMatrixData,
  simData: SimDayData | null,
  options: TopologyBuildOptions = {}
): {
  nodeMeta: TopologyNodeMeta[];
  edges: GraphEdge[];
  positions: [number, number][];
  option: Record<string, unknown>;
} {
  const {
    kNeighbors = 3,
    maxEdges = 1200,
    seedPositions,
    layoutIterations = 80,
    relayoutOnFilter = false,
    fillRange = null,
    layoutMode = "auto",
    pheromoneWeights,
    showPheromone = false,
    logScale = false,
    theme = "dark",
  } = options;

  const nodeCount = matrix.nodeIds.length;
  const resolvedLayout = resolveLayoutMode(nodeCount, layoutMode);

  const nodeMeta = buildNodeMetaFromSim(simData, matrix.nodeIds);
  let activeIndices = nodeMeta.map((_, i) => i);

  if (relayoutOnFilter && fillRange) {
    const [lo, hi] = fillRange;
    activeIndices = nodeMeta
      .filter((m) => m.fillPct == null || (m.fillPct >= lo && m.fillPct <= hi) || m.onTour)
      .map((m) => m.matrixIndex);
  }

  const indexSet = new Set(activeIndices);
  let edges = buildKnnEdgeList(matrix.distances, kNeighbors, maxEdges).filter(
    (e) => indexSet.has(e.source) && indexSet.has(e.target)
  );

  if (showPheromone && pheromoneWeights?.size) {
    const edgeKeys = new Set(edges.map((e) => pheromoneEdgeKey(e.source, e.target)));
    for (const key of pheromoneWeights.keys()) {
      if (edgeKeys.has(key)) continue;
      const [a, b] = key.split("-").map(Number);
      if (!indexSet.has(a) || !indexSet.has(b)) continue;
      const distance = matrix.distances[a]?.[b];
      if (!Number.isFinite(distance) || distance <= 0) continue;
      edges.push({ source: a, target: b, distance });
      edgeKeys.add(key);
      if (edges.length >= maxEdges + 200) break;
    }
  }

  const layoutEdges =
    edges.length > 0 ? edges : buildKnnEdgeList(matrix.distances, kNeighbors, maxEdges);

  const positions =
    resolvedLayout === "radial"
      ? radialDenseLayout(nodeCount, matrix.nodeIds)
      : forceDirectedLayout(nodeCount, layoutEdges, seedPositions, layoutIterations);

  const option = buildTopologyGraphOption(nodeMeta, edges, positions, {
    fillRange,
    theme,
    pheromoneWeights,
    showPheromone,
    logScale,
  });

  return { nodeMeta, edges, positions, option };
}
