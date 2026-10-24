import { resolveBinPositions } from "./mapPositions";
import {
  computeTourDiff,
  failureBinIdSets,
} from "./routeFailureOverlay";
import { splitVehicleTourIndices, VEHICLE_COLORS_RGB } from "./vehicleTours";
import type { SimDayData, SimFailureSummary } from "../../types";

export const TRIP_SEGMENT_MS = 120;

export interface MapRoute {
  id: string;
  label: string;
  data: SimDayData;
  color: [number, number, number];
}

export interface VehicleGeometry {
  vehicleId: number;
  color: [number, number, number];
  path: [number, number][];
  tripPath: [number, number, number][];
  tripLength: number;
}

export interface VehicleTourStop {
  position: [number, number];
  fill: number;
  collected: number;
  vehicleId: number;
  color: [number, number, number];
}

export interface FailureHighlightPoint {
  position: [number, number];
  kind: "overflow" | "skipped";
}

export interface TourDiffPoint {
  position: [number, number];
  side: "onlyFirst" | "onlySecond";
}

export interface HexDensityCell {
  id: string;
  center: [number, number];
  polygon: [number, number][];
  binCount: number;
  meanFill: number;
  maxFill: number;
  overflowCount: number;
}

export function fillRgb(pct: number): [number, number, number] {
  if (pct >= 100) return [248, 113, 113];
  if (pct >= 80) return [251, 191, 36];
  return [52, 211, 153];
}

export function stopRadius(fill: number, collected: number, multiPolicy: boolean): number {
  const base = multiPolicy ? 40 : 50;
  const fillScale = 0.5 + (Math.min(100, fill) / 100) * 0.5;
  const collectedScale = collected > 0 ? 1 + Math.min(collected, 50) / 50 : 1;
  return base * fillScale * collectedScale;
}

function buildVehicleSegment(
  segment: number[],
  posById: Map<number, [number, number]>,
  vehicleId: number,
  timeOffset: number
): VehicleGeometry {
  const pathCoords: [number, number][] = [];
  const tripCoords: [number, number, number][] = [];
  let t = timeOffset;

  const pushStop = (id: number) => {
    const p = posById.get(id);
    if (!p) return;
    pathCoords.push(p);
    tripCoords.push([p[0], p[1], t]);
    t += TRIP_SEGMENT_MS;
  };

  const depot = posById.get(-1);
  if (depot) pushStop(-1);
  for (const id of segment) pushStop(id);
  if (depot && pathCoords.length > 1) {
    pathCoords.push(depot);
    tripCoords.push([depot[0], depot[1], t]);
  }

  return {
    vehicleId,
    color: VEHICLE_COLORS_RGB[vehicleId % VEHICLE_COLORS_RGB.length],
    path: pathCoords,
    tripPath: tripCoords,
    tripLength: t,
  };
}

export function buildRouteGeometry(data: SimDayData) {
  const { all_bin_coords, tour_indices, bin_state_c, bin_state_collected } = data;
  if (!all_bin_coords?.length) {
    return {
      vehicles: [] as VehicleGeometry[],
      path: [] as [number, number][],
      tripPath: [] as [number, number, number][],
      tripLength: 0,
      tourPoints: [] as Array<{ position: [number, number]; fill: number; collected: number }>,
      vehicleTourStops: [] as VehicleTourStop[],
      idlePoints: [] as Array<{ position: [number, number] }>,
      depotPoint: null as { position: [number, number] } | null,
      hasGeo: false,
      hasBins: false,
      center: { longitude: 0, latitude: 0 },
    };
  }

  const { posById, hasGeo } = resolveBinPositions(all_bin_coords);
  const segments = splitVehicleTourIndices(data);
  const allTourIds = new Set(segments.flat());
  const tourSet = new Set(tour_indices ?? allTourIds);

  let timeOffset = 0;
  const vehicles = segments.map((segment, i) => {
    const geom = buildVehicleSegment(segment, posById, i, timeOffset);
    timeOffset = geom.tripLength + TRIP_SEGMENT_MS;
    return geom;
  });

  const primary = vehicles[0];
  const pathCoords = primary?.path ?? [];
  const tripCoords = primary?.tripPath ?? [];
  const tripLength = vehicles.length
    ? Math.max(...vehicles.map((v) => v.tripLength), 0)
    : 0;

  const collectedAt = (id: number) => {
    const v = bin_state_collected?.[id];
    return typeof v === "number" && v > 0 ? v : 0;
  };

  const tourPts = [...tourSet]
    .map((id) => {
      const position = posById.get(id);
      if (!position) return null;
      return {
        position,
        fill: (bin_state_c?.[id] ?? 0) * 100,
        collected: collectedAt(id),
      };
    })
    .filter(Boolean) as Array<{ position: [number, number]; fill: number; collected: number }>;

  const vehicleTourStops: VehicleTourStop[] = segments.flatMap((segment, vi) =>
    segment
      .map((id) => {
        const position = posById.get(id);
        if (!position) return null;
        return {
          position,
          fill: (bin_state_c?.[id] ?? 0) * 100,
          collected: collectedAt(id),
          vehicleId: vi,
          color: VEHICLE_COLORS_RGB[vi % VEHICLE_COLORS_RGB.length],
        };
      })
      .filter(Boolean) as VehicleTourStop[]
  );

  const idlePts = all_bin_coords
    .filter((b) => b.id >= 0 && !tourSet.has(b.id))
    .map((b) => {
      const position = posById.get(b.id);
      return position ? { position } : null;
    })
    .filter(Boolean) as Array<{ position: [number, number] }>;

  const depot = posById.get(-1);
  const lngs = [...posById.values()].map((p) => p[0]);
  const lats = [...posById.values()].map((p) => p[1]);

  return {
    vehicles,
    path: pathCoords,
    tripPath: tripCoords,
    tripLength,
    tourPoints: tourPts,
    vehicleTourStops,
    idlePoints: idlePts,
    depotPoint: depot ? { position: depot } : null,
    hasGeo,
    hasBins: posById.size > 0,
    center: {
      longitude: lngs.length ? lngs.reduce((a, b) => a + b, 0) / lngs.length : 0,
      latitude: lats.length ? lats.reduce((a, b) => a + b, 0) / lats.length : 0,
    },
  };
}

export function buildFailureHighlights(
  posById: Map<number, [number, number]>,
  summary: SimFailureSummary | null | undefined
): FailureHighlightPoint[] {
  const { overflowIds, skippedIds } = failureBinIdSets(summary);
  const points: FailureHighlightPoint[] = [];

  for (const id of overflowIds) {
    const position = posById.get(id);
    if (position) points.push({ position, kind: "overflow" });
  }
  for (const id of skippedIds) {
    if (overflowIds.has(id)) continue;
    const position = posById.get(id);
    if (position) points.push({ position, kind: "skipped" });
  }

  return points;
}

export function buildTourDiffHighlights(
  first: SimDayData,
  second: SimDayData,
  posById: Map<number, [number, number]>
): TourDiffPoint[] {
  const diff = computeTourDiff(first, second);
  const points: TourDiffPoint[] = [];

  for (const id of diff.onlyFirst) {
    const position = posById.get(id);
    if (position) points.push({ position, side: "onlyFirst" });
  }
  for (const id of diff.onlySecond) {
    const position = posById.get(id);
    if (position) points.push({ position, side: "onlySecond" });
  }

  return points;
}

/**
 * Compute hexagonal spatial density clusters for a `SimDayData` snapshot (Owner Ruling #3).
 *
 * Groups bin coordinates into axial hex cells and aggregates bin count, mean fill %,
 * peak fill %, and overflow count without replacing ground-truth GPS routes.
 */
export function buildHexDensityCells(
  data: SimDayData | undefined,
  gridDivisions = 10
): HexDensityCell[] {
  if (!data?.all_bin_coords?.length) return [];
  const { posById } = resolveBinPositions(data.all_bin_coords);
  const entries: Array<{ id: number; pos: [number, number]; fill: number }> = [];
  for (const b of data.all_bin_coords) {
    if (b.id < 0) continue;
    const pos = posById.get(b.id);
    if (!pos) continue;
    const fill = (data.bin_state_c?.[b.id] ?? 0) * 100;
    entries.push({ id: b.id, pos, fill });
  }
  if (entries.length === 0) return [];

  const xs = entries.map((e) => e.pos[0]);
  const ys = entries.map((e) => e.pos[1]);
  const minX = Math.min(...xs);
  const maxX = Math.max(...xs);
  const minY = Math.min(...ys);
  const maxY = Math.max(...ys);
  const span = Math.max(maxX - minX, maxY - minY, 1e-4);
  const radius = span / Math.max(4, gridDivisions);

  const hexW = Math.sqrt(3) * radius;
  const vertStep = 1.5 * radius;

  const buckets = new Map<
    string,
    { row: number; col: number; cx: number; cy: number; fills: number[] }
  >();

  for (const e of entries) {
    const row = Math.round((e.pos[1] - minY) / vertStep);
    const xOffset = row % 2 !== 0 ? hexW / 2 : 0;
    const col = Math.round((e.pos[0] - minX - xOffset) / hexW);
    const key = `${row}:${col}`;
    let bucket = buckets.get(key);
    if (!bucket) {
      const cx = minX + col * hexW + xOffset;
      const cy = minY + row * vertStep;
      bucket = { row, col, cx, cy, fills: [] };
      buckets.set(key, bucket);
    }
    bucket.fills.push(e.fill);
  }

  const cells: HexDensityCell[] = [];
  for (const [id, b] of buckets.entries()) {
    const polygon: [number, number][] = [];
    for (let i = 0; i < 6; i++) {
      const angle = (Math.PI / 180) * (60 * i - 30);
      polygon.push([
        b.cx + radius * 0.92 * Math.cos(angle),
        b.cy + radius * 0.92 * Math.sin(angle),
      ]);
    }
    const sum = b.fills.reduce((a, v) => a + v, 0);
    const maxFill = Math.max(...b.fills);
    const overflowCount = b.fills.filter((v) => v >= 100).length;
    cells.push({
      id,
      center: [b.cx, b.cy],
      polygon,
      binCount: b.fills.length,
      meanFill: sum / b.fills.length,
      maxFill,
      overflowCount,
    });
  }

  return cells;
}
