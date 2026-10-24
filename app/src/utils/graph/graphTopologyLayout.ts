/**
 * Graph topology layout algorithms: KNN edge construction, force-directed, and radial layout.
 */

export type TopologyLayoutMode = "force" | "radial";

/** Auto-select radial layout for dense graphs (Cosmograph-style fallback). */
export const DENSE_GRAPH_NODE_THRESHOLD = 200;

export interface GraphEdge {
  source: number;
  target: number;
  distance: number;
}

export function pheromoneEdgeKey(a: number, b: number): string {
  const lo = Math.min(a, b);
  const hi = Math.max(a, b);
  return `${lo}-${hi}`;
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
