/**
 * Bin rows for the summary honeycomb.
 * Missing coordinates and fills stay null. This module does not invent positions.
 */
import type { CorpusBin, CorpusRouteBin } from "../../../types/corpus";
import type { DayLogEntry } from "../../../types";
import type { HoneycombBinInput } from "../../../utils/orion/liveScenes";

export interface SummaryHoneycombPoint {
  bins: readonly CorpusBin[];
  routeBins: readonly CorpusRouteBin[];
}

function finite(value: number | null | undefined): value is number {
  return typeof value === "number" && Number.isFinite(value);
}

function depotId(id: string): boolean {
  return id === "0" || id === "-1";
}

function remember(
  bins: Map<string, HoneycombBinInput>,
  id: string,
  lat: number | null,
  lon: number | null,
  fill: number | null
): void {
  const previous = bins.get(id);
  bins.set(id, {
    id,
    lat: lat ?? previous?.lat ?? null,
    lon: lon ?? previous?.lon ?? null,
    fill: fill ?? previous?.fill ?? null,
    depot: depotId(id),
  });
}

/** Match each corpus fill to route coordinates by id. Unmatched bins stay in the list. */
export function binsFromCorpusPoints(
  points: readonly SummaryHoneycombPoint[]
): HoneycombBinInput[] {
  const bins = new Map<string, HoneycombBinInput>();
  for (const point of points) {
    const coords = new Map<string, CorpusRouteBin>();
    for (const routeBin of point.routeBins) coords.set(String(routeBin.id), routeBin);
    const seen = new Set<string>();
    for (const bin of point.bins) {
      seen.add(bin.binId);
      const routeBin = coords.get(bin.binId);
      remember(
        bins,
        bin.binId,
        routeBin != null && finite(routeBin.lat) ? routeBin.lat : null,
        routeBin != null && finite(routeBin.lon) ? routeBin.lon : null,
        finite(bin.fill) ? bin.fill : null
      );
    }
    for (const routeBin of point.routeBins) {
      const id = String(routeBin.id);
      if (seen.has(id)) continue;
      const previous = bins.get(id);
      remember(
        bins,
        id,
        finite(routeBin.lat) ? routeBin.lat : null,
        finite(routeBin.lon) ? routeBin.lon : null,
        previous?.fill ?? null
      );
    }
  }
  return [...bins.values()];
}

/**
 * One cell per bin id. `bin_state_c` aligns by index only when it is the same length
 * as `all_bin_coords`. A later row replaces the same id.
 */
export function binsFromLogEntries(entries: readonly DayLogEntry[]): HoneycombBinInput[] {
  const bins = new Map<string, HoneycombBinInput>();
  for (const entry of entries) {
    const coords = entry.data.all_bin_coords;
    if (!coords || coords.length === 0) continue;
    const fills = entry.data.bin_state_c;
    const aligned = fills != null && fills.length === coords.length ? fills : null;
    coords.forEach((bin, index) => {
      const raw = aligned != null ? aligned[index] : null;
      remember(
        bins,
        String(bin.id),
        finite(bin.lat) ? bin.lat : null,
        finite(bin.lng) ? bin.lng : null,
        finite(raw) ? raw : null
      );
    });
  }
  return [...bins.values()];
}
