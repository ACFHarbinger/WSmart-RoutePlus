/**
 * Topological graph analytics — distance matrix → edge list + force layout (§G.4).
 */
import { invoke } from "@tauri-apps/api/core";
import type { SimDayData } from "../../types";
import { parseLogPath } from "../sim/simMetadata";
import {
  TopologyLayoutMode,
  DENSE_GRAPH_NODE_THRESHOLD,
  GraphEdge,
  pheromoneEdgeKey,
  resolveLayoutMode,
  buildKnnEdgeList,
  forceDirectedLayout,
  radialDenseLayout,
} from "./graphTopologyLayout";
import {
  guiIdToMatrixIndex,
  tourIndicesToMatrixPairs,
  countTourEdgeSteps,
  accumulateTourPheromoneByStep,
  accumulateTourPheromone,
  normalizePheromone,
  pheromoneIntensity,
} from "./graphTopologyPheromone";

export type { TopologyLayoutMode, GraphEdge };
export {
  DENSE_GRAPH_NODE_THRESHOLD,
  pheromoneEdgeKey,
  resolveLayoutMode,
  buildKnnEdgeList,
  forceDirectedLayout,
  radialDenseLayout,
  tourIndicesToMatrixPairs,
  countTourEdgeSteps,
  accumulateTourPheromoneByStep,
  accumulateTourPheromone,
  normalizePheromone,
  pheromoneIntensity,
};

export interface DistanceMatrixData {
  nodeIds: number[];
  distances: number[][];
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
