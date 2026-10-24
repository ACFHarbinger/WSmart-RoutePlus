import { ChevronLeft, ChevronRight } from "lucide-react";
import { RouteViz } from "../routes/RouteViz";
import { parsePolicyLabel } from "../../../utils/sim/simMetadata";
import type { DayLogEntry } from "../../../types";

export function SummaryRouteSection({
  routeVizDays,
  routeVizDay,
  setRouteVizDay,
  routeVizHasFailureData,
  showFailureOverlay,
  setShowFailureOverlay,
  routeVizEntries,
  showRouteDiff,
  setShowRouteDiff,
}: {
  routeVizDays: number[];
  routeVizDay: number;
  setRouteVizDay: (day: number) => void;
  routeVizHasFailureData: boolean;
  showFailureOverlay: boolean;
  setShowFailureOverlay: (fn: (v: boolean) => boolean) => void;
  routeVizEntries: DayLogEntry[];
  showRouteDiff: boolean;
  setShowRouteDiff: (fn: (v: boolean) => boolean) => void;
}) {
  if (routeVizDays.length === 0) return null;

  return (
    <div className="space-y-2">
      <div className="flex items-center justify-between gap-2 flex-wrap">
        <div>
          <p className="text-xs font-semibold text-gray-300">
            Route Solution (§A.1 / §A.6)
          </p>
          <p className="text-[10px] text-canvas-muted">
            ECharts spatial overlay — depot, demand-sized nodes, per-vehicle
            edges; failure + route-diff toggles when comparing two policies
            (§A.6)
          </p>
        </div>
        <div className="flex items-center gap-2 flex-wrap justify-end">
          {routeVizHasFailureData && (
            <button
              className={`btn-ghost text-xs ${
                showFailureOverlay ? "text-accent-danger" : ""
              }`}
              onClick={() => setShowFailureOverlay((v) => !v)}
            >
              {showFailureOverlay ? "Hide" : "Show"} failure overlay
            </button>
          )}
          {routeVizEntries.length === 2 && (
            <button
              className={`btn-ghost text-xs ${
                showRouteDiff ? "text-accent-secondary" : ""
              }`}
              onClick={() => setShowRouteDiff((v) => !v)}
            >
              {showRouteDiff ? "Hide" : "Show"} route diff
            </button>
          )}
          <button
            onClick={() => {
              const idx = routeVizDays.indexOf(routeVizDay);
              if (idx > 0) setRouteVizDay(routeVizDays[idx - 1]);
            }}
            disabled={routeVizDays.indexOf(routeVizDay) <= 0}
            className="btn-ghost p-1"
            title="Previous day"
          >
            <ChevronLeft size={14} />
          </button>
          <span className="text-xs font-mono text-gray-300 min-w-[4rem] text-center tabular-nums">
            Day {routeVizDay}
          </span>
          <button
            onClick={() => {
              const idx = routeVizDays.indexOf(routeVizDay);
              if (idx >= 0 && idx < routeVizDays.length - 1) {
                setRouteVizDay(routeVizDays[idx + 1]);
              }
            }}
            disabled={routeVizDays.indexOf(routeVizDay) >= routeVizDays.length - 1}
            className="btn-ghost p-1"
            title="Next day"
          >
            <ChevronRight size={14} />
          </button>
          <input
            type="range"
            min={routeVizDays[0]}
            max={routeVizDays[routeVizDays.length - 1]}
            value={routeVizDay}
            onChange={(e) => setRouteVizDay(Number(e.target.value))}
            className="w-28 accent-accent-primary"
          />
        </div>
      </div>
      {routeVizEntries.length > 0 ? (
        showRouteDiff && routeVizEntries.length === 2 ? (
          <RouteViz
            data={routeVizEntries[0].data}
            compareData={routeVizEntries[1].data}
            primaryLabel={routeVizEntries[0].policy}
            compareLabel={routeVizEntries[1].policy}
            title="Route Solution"
            subtitle={`Day ${routeVizDay} · overlay compare`}
            filenameStem={`route-viz-overlay-day${routeVizDay}`}
            showFailureOverlay={showFailureOverlay}
            showTourDiff
          />
        ) : (
          <div
            className={`grid gap-3 ${
              routeVizEntries.length > 1
                ? "grid-cols-1 lg:grid-cols-2"
                : "grid-cols-1"
            }`}
          >
            {routeVizEntries.map((entry) => (
              <RouteViz
                key={`${entry.policy}-${entry.day}`}
                data={entry.data}
                title="Route Solution"
                subtitle={`${
                  parsePolicyLabel(entry.policy).selectionStrategy
                } · Day ${entry.day}`}
                filenameStem={`route-viz-day${entry.day}-${entry.policy}`}
                showFailureOverlay={showFailureOverlay}
              />
            ))}
          </div>
        )
      ) : (
        <p className="text-xs text-canvas-muted py-4 text-center card">
          No bin coordinates for day {routeVizDay} — run simulation with
          coordinate logging enabled.
        </p>
      )}
    </div>
  );
}
