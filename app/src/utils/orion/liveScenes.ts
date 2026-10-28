/**
 * Live Orion scene props from simulation rows.
 * Missing metrics stay gaps. A layout without coordinates is marked generated.
 */
import type { HoneycombZone, ZoneTrajectory } from "../../components/ui/orion/OrionHoneycombMap";
import type { OrbitalBubble } from "../../components/ui/orion/OrionOrbitalBubbleChart";

export interface LiveHoneycomb {
  zones: HoneycombZone[];
  trajectories: ZoneTrajectory[];
  /** True when no input row had coordinates. Positions are a square grid. */
  generated: boolean;
  /** Rows left out because other rows had coordinates and these did not. */
  omitted: number;
}

export interface HoneycombBinInput {
  id: string;
  lat: number | null;
  lon: number | null;
  /** 0..1+ fill. Null is a gap. */
  fill: number | null;
  depot?: boolean;
}

export interface CityHoneycombInput {
  city: string;
  lat: number | null;
  lon: number | null;
  fill: number | null;
  /** Present overflow count. Null is a gap and is not drawn as zero fill. */
  overflows: number | null;
}

export interface PolicyMetricRow {
  policy: string;
  metrics: Record<string, number | null | undefined>;
}

const LOWER_IS_BETTER = new Set(["km", "cost", "overflows", "kg_lost"]);

function finite(value: number | null | undefined): value is number {
  return typeof value === "number" && Number.isFinite(value);
}

function squareSlot(index: number, count: number): { row: number; col: number } {
  const columns = Math.max(1, Math.ceil(Math.sqrt(count)));
  return { row: Math.floor(index / columns), col: index % columns };
}

function fillStatus(fill: number | null): Pick<HoneycombZone, "status" | "overflowRisk" | "fillRatio"> & {
  gap: boolean;
} {
  if (!finite(fill)) {
    return { status: "idle", overflowRisk: false, fillRatio: 0, gap: true };
  }
  if (fill >= 1) return { status: "crit", overflowRisk: true, fillRatio: fill, gap: false };
  if (fill >= 0.8) return { status: "warn", overflowRisk: false, fillRatio: fill, gap: false };
  return { status: "ok", overflowRisk: false, fillRatio: fill, gap: false };
}

function zoneFromFill(
  id: string,
  label: string,
  index: number,
  count: number,
  fill: number | null,
  extra: Partial<HoneycombZone> & { overflows?: number | null }
): HoneycombZone {
  const measured = fillStatus(fill);
  const metrics: Record<string, string | number> = {};
  if (measured.gap) metrics.fill = "gap";
  else metrics.fill = measured.fillRatio;
  let status = measured.status;
  let overflowRisk = measured.overflowRisk;
  if (finite(extra.overflows)) {
    metrics.overflows = extra.overflows;
    if (extra.overflows > 0) {
      status = "crit";
      overflowRisk = true;
    } else if (measured.gap) {
      status = "ok";
    }
  }
  return {
    id,
    label,
    row: squareSlot(index, count).row,
    col: squareSlot(index, count).col,
    fillRatio: measured.fillRatio,
    binCount: extra.binCount ?? 1,
    overflowRisk,
    status,
    depot: extra.depot,
    metrics,
  };
}

function byNorthThenEast<T extends { lat: number | null; lon: number | null }>(rows: T[]): T[] {
  return [...rows].sort((a, b) => {
    const latDelta = (b.lat as number) - (a.lat as number);
    if (latDelta !== 0) return latDelta;
    return (a.lon as number) - (b.lon as number);
  });
}

/** Place bins on a honeycomb. Trajectories stay empty: this view does not invent tours. */
export function honeycombFromBins(bins: readonly HoneycombBinInput[]): LiveHoneycomb | null {
  if (bins.length === 0) return null;
  const located = bins.filter((bin) => finite(bin.lat) && finite(bin.lon));
  const generated = located.length === 0;
  const source = generated ? [...bins] : byNorthThenEast(located);
  const zones = source.map((bin, index) =>
    zoneFromFill(bin.id, bin.id, index, source.length, bin.fill, {
      depot: bin.depot === true || bin.id === "0" || bin.id === "-1",
    })
  );
  return {
    zones,
    trajectories: [],
    generated,
    omitted: generated ? 0 : bins.length - located.length,
  };
}

/** One cell per city. Cities without coordinates stay off a coordinate grid. */
export function honeycombFromCities(cities: readonly CityHoneycombInput[]): LiveHoneycomb | null {
  const named = cities.filter((city) => city.city.trim().length > 0);
  if (named.length === 0) return null;
  const located = named.filter((city) => finite(city.lat) && finite(city.lon));
  const generated = located.length === 0;
  const source = generated
    ? [...named].sort((a, b) => a.city.localeCompare(b.city))
    : byNorthThenEast(located);
  const zones = source.map((city, index) =>
    zoneFromFill(city.city, city.city, index, source.length, city.fill, {
      overflows: city.overflows,
    })
  );
  return {
    zones,
    trajectories: [],
    generated,
    omitted: generated ? 0 : named.length - located.length,
  };
}

export function honeycombSubtitle(scene: LiveHoneycomb): string {
  if (scene.generated) {
    return "Generated layout. These positions are not geographic coordinates.";
  }
  const omitted =
    scene.omitted > 0
      ? ` ${scene.omitted} ${scene.omitted === 1 ? "row" : "rows"} without coordinates are not drawn.`
      : "";
  return `Grid placed from recorded coordinates.${omitted}`;
}

/**
 * One bubble per policy that has a finite value for `metric`.
 * A missing value is omitted. It is not drawn as zero.
 */
export function orbitalBubblesFromPolicies(
  rows: readonly PolicyMetricRow[],
  metric: string
): OrbitalBubble[] {
  const present = rows.flatMap((row, index) => {
    const value = row.metrics[metric];
    if (!finite(value) || row.policy.trim().length === 0) return [];
    return [{ policy: row.policy, value, index }];
  });
  if (present.length === 0) return [];
  const values = present.map((row) => row.value);
  const min = Math.min(...values);
  const max = Math.max(...values);
  const span = max - min;
  const lowerIsBetter = LOWER_IS_BETTER.has(metric);
  return present.map((row, index) => {
    const magnitude = span === 0 ? 0.5 : (row.value - min) / span;
    const quality = span === 0 ? 0.5 : lowerIsBetter ? (max - row.value) / span : magnitude;
    const orbitIdx = quality >= 2 / 3 ? 0 : quality >= 1 / 3 ? 1 : 2;
    return {
      id: `${row.policy}:${row.index}`,
      label: row.policy,
      orbitIdx,
      angleDeg: (360 * index) / present.length,
      radius: 12 + 16 * magnitude,
      value: row.value,
      unit: metric,
      tone: orbitIdx === 0 ? "ok" : orbitIdx === 1 ? "warn" : "crit",
      metrics: { [metric]: row.value },
    };
  });
}
