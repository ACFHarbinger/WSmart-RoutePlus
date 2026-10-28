import type { CorpusSeriesPoint, CorpusView } from "../../types/corpus";
import type { MapRoute } from "../map/deckRouteGeometry";
import type { SimDayData } from "../../types";

export interface CorpusMapFrame {
  /** Empty when the selected day has no route geometry. The caller keeps the current map. */
  routes: MapRoute[];
  reason: string | null;
}

const COLORS: [number, number, number][] = [
  [56, 189, 248],
  [251, 191, 36],
  [52, 211, 153],
  [244, 114, 182],
  [167, 139, 250],
];

function finite(value: number | null): value is number {
  return typeof value === "number" && Number.isFinite(value);
}

function hasRouteGeometry(point: CorpusSeriesPoint): boolean {
  const tour = point.tour.length > 0 || point.tourIndices.length > 0;
  const coords = point.routeBins.some((bin) => finite(bin.lat) && finite(bin.lon));
  return tour && coords;
}

function simDay(point: CorpusSeriesPoint): SimDayData {
  return {
    tour: point.tour,
    tour_indices: point.tourIndices,
    all_bin_coords: point.routeBins.map((bin) => ({
      id: bin.id,
      lat: bin.lat ?? undefined,
      lng: bin.lon ?? undefined,
    })),
  };
}

/**
 * Routes for one corpus day. A day with no tour, or no coordinates, returns no routes
 * and a reason so the existing map stays where it is.
 */
export function corpusMapFrame(view: CorpusView, day: number | null): CorpusMapFrame {
  const points =
    day == null ? view.points : view.points.filter((point) => point.day === day);
  if (points.length === 0) {
    return { routes: [], reason: "No rows for this day." };
  }
  const drawable = points.filter(hasRouteGeometry);
  if (drawable.length === 0) {
    const partial = points.some((point) => point.partial);
    const reason = partial
      ? "This partial run has no route geometry."
      : "This day has no route geometry.";
    return { routes: [], reason };
  }
  const routes = drawable.map((point, index) => ({
    id: `${point.runId}:${point.policy ?? "run"}:${point.day ?? "day"}:${index}`,
    label: point.policy ?? point.runLabel,
    data: simDay(point),
    color: COLORS[index % COLORS.length]!,
  }));
  return { routes, reason: null };
}
