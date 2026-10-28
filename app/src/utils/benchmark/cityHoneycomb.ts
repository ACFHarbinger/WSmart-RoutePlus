/**
 * One honeycomb row per city group. Missing fill, overflows, or coordinates stay null.
 */
import type { DayLogEntry } from "../../types";
import type { CityHoneycombInput } from "../orion/liveScenes";
import type { CityRunSlice } from "./cityComparison";

function finite(value: number | null | undefined): value is number {
  return typeof value === "number" && Number.isFinite(value);
}

function meanFinite(values: readonly (number | null | undefined)[]): number | null {
  let sum = 0;
  let count = 0;
  for (const value of values) {
    if (!finite(value)) continue;
    sum += value;
    count += 1;
  }
  return count === 0 ? null : sum / count;
}

/** City label is the groupRunsByCity key. Fill is bin state only, never profit, kg, or overflows. */
export function cityHoneycombInputs(
  cityGroups: ReadonlyArray<readonly [string, readonly CityRunSlice[]]>
): CityHoneycombInput[] {
  return cityGroups.map(([city, runs]) => {
    const entries: DayLogEntry[] = runs.flatMap((run) => run.entries);
    const lats: number[] = [];
    const lons: number[] = [];
    const fills: number[] = [];
    for (const entry of entries) {
      for (const bin of entry.data.all_bin_coords ?? []) {
        if (finite(bin.lat)) lats.push(bin.lat);
        if (finite(bin.lng)) lons.push(bin.lng);
      }
      for (const fill of entry.data.bin_state_c ?? []) {
        if (finite(fill)) fills.push(fill);
      }
    }
    return {
      city,
      lat: meanFinite(lats),
      lon: meanFinite(lons),
      fill: meanFinite(fills),
      overflows: meanFinite(entries.map((entry) => entry.data.overflows)),
    };
  });
}
