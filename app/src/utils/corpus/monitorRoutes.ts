import type { MapRoute } from "../map/deckRouteGeometry";

/**
 * Swap the monitor's current routes for one corpus day.
 * A day with no geometry leaves the current routes in place and surfaces `reason`.
 */
export function applyCorpusMapFrame(
  current: MapRoute[],
  frame: { routes: MapRoute[]; reason: string | null } | null,
  playbackDay: number | null
): { routes: MapRoute[]; reason: string | null; driven: boolean } {
  if (playbackDay == null || frame == null) {
    return { routes: current, reason: null, driven: false };
  }
  if (frame.routes.length > 0) {
    return { routes: frame.routes, reason: null, driven: true };
  }
  return { routes: current, reason: frame.reason, driven: false };
}
