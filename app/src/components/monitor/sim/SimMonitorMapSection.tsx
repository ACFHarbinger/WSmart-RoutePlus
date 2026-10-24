import { lazy, Suspense } from "react";
import { RouteViz } from "../../analysis/routes/RouteViz";
import { GRAPH_PRESETS, guessGraphPreset } from "../../../utils/graph/graphCoords";
import { POLICY_COLORS } from "./SimMonitorPanels";
import type { DayLogEntry } from "../../../types";
import type { MapRoute } from "../../maps/DeckRouteMap";

const DeckRouteMap = lazy(() => import("../../maps/DeckRouteMap"));

export interface SimMonitorMapSectionProps {
  showBinFill: boolean;
  setShowBinFill: (fn: (v: boolean) => boolean) => void;
  showTourTable: boolean;
  setShowTourTable: (fn: (v: boolean) => boolean) => void;
  showRouteMap: boolean;
  setShowRouteMap: (fn: (v: boolean) => boolean) => void;
  policies: string[];
  activeMapPolicies: string[];
  toggleMapPolicy: (policy: string) => void;
  mapRoutes: MapRoute[];
  mapLayout: "overlay" | "split";
  setMapLayout: (layout: "overlay" | "split") => void;
  hasGeoCoords: boolean;
  hasBinCoords: boolean;
  entries: DayLogEntry[];
  graphPreset: string;
  setGraphPreset: (id: string) => void;
  activeLogPath: string | null;
  applyGraphCoords: () => Promise<void>;
  loadingGraphCoords: boolean;
  mapHasFailureData: boolean;
  showFailureOverlay: boolean;
  setShowFailureOverlay: (fn: (v: boolean) => boolean) => void;
  showRouteDiff: boolean;
  setShowRouteDiff: (fn: (v: boolean) => boolean) => void;
  showHexDensity: boolean;
  setShowHexDensity: (fn: (v: boolean) => boolean) => void;
  routeMapMode: "echarts" | "deckgl";
  setRouteMapMode: (mode: "echarts" | "deckgl") => void;
  isPlaying: boolean;
  playbackSpeed: 1 | 2 | 4;
  displayDay: number;
  displayEntry: DayLogEntry | null;
}

export function SimMonitorMapSection({
  showBinFill,
  setShowBinFill,
  showTourTable,
  setShowTourTable,
  showRouteMap,
  setShowRouteMap,
  policies,
  activeMapPolicies,
  toggleMapPolicy,
  mapRoutes,
  mapLayout,
  setMapLayout,
  hasGeoCoords,
  hasBinCoords,
  entries,
  graphPreset,
  setGraphPreset,
  activeLogPath,
  applyGraphCoords,
  loadingGraphCoords,
  mapHasFailureData,
  showFailureOverlay,
  setShowFailureOverlay,
  showRouteDiff,
  setShowRouteDiff,
  showHexDensity,
  setShowHexDensity,
  routeMapMode,
  setRouteMapMode,
  isPlaying,
  playbackSpeed,
  displayDay,
  displayEntry,
}: SimMonitorMapSectionProps) {
  return (
    <>
      <div className="flex items-center gap-2 flex-wrap">
        <button
          type="button"
          className="btn-ghost text-xs"
          onClick={() => setShowBinFill((v) => !v)}
        >
          {showBinFill ? "Hide" : "Show"} bin fill chart
        </button>
        <button
          type="button"
          className="btn-ghost text-xs"
          onClick={() => setShowTourTable((v) => !v)}
        >
          {showTourTable ? "Hide" : "Show"} tour table
        </button>
        <button
          type="button"
          className="btn-ghost text-xs"
          onClick={() => setShowRouteMap((v) => !v)}
        >
          {showRouteMap ? "Hide" : "Show"} route map
        </button>
        {showRouteMap && policies.length > 1 && (
          <div className="flex items-center gap-1 flex-wrap ml-1">
            <span className="text-[10px] text-canvas-muted">Map:</span>
            {policies.map((p, i) => {
              const color = POLICY_COLORS[i % POLICY_COLORS.length];
              const active = activeMapPolicies.includes(p);
              return (
                <button
                  key={p}
                  type="button"
                  onClick={() => toggleMapPolicy(p)}
                  className={`text-[10px] px-1.5 py-0.5 rounded-full border ${
                    active ? "opacity-100" : "opacity-35"
                  }`}
                  style={{ borderColor: color, color: active ? color : undefined }}
                >
                  {p}
                </button>
              );
            })}
          </div>
        )}
        {showRouteMap && mapRoutes.length === 2 && (
          <div className="flex items-center gap-1 bg-canvas-elevated rounded-lg p-0.5 ml-1">
            {(["overlay", "split"] as const).map((layout) => (
              <button
                key={layout}
                type="button"
                onClick={() => setMapLayout(layout)}
                className={`text-xs px-2 py-0.5 rounded-md capitalize ${
                  mapLayout === layout
                    ? "bg-accent-primary text-white"
                    : "text-canvas-muted hover:text-gray-200"
                }`}
              >
                {layout}
              </button>
            ))}
          </div>
        )}
        {showRouteMap && !hasGeoCoords && entries.length > 0 && (
          <div className="flex items-center gap-2 ml-1 flex-wrap">
            <select
              className="select-base text-xs py-0.5"
              value={graphPreset}
              onChange={(e) => setGraphPreset(e.target.value)}
            >
              {GRAPH_PRESETS.map((p) => (
                <option key={p.id} value={p.id}>
                  {p.label}
                </option>
              ))}
            </select>
            {activeLogPath &&
              guessGraphPreset(activeLogPath, entries) === graphPreset && (
                <span className="text-[10px] text-accent-secondary">
                  auto-detected
                </span>
              )}
            <button
              type="button"
              onClick={() => void applyGraphCoords()}
              disabled={loadingGraphCoords}
              className="btn-ghost text-xs"
            >
              {loadingGraphCoords ? "Loading…" : "Load graph coords"}
            </button>
          </div>
        )}
        {showRouteMap && mapHasFailureData && (
          <button
            type="button"
            className={`btn-ghost text-xs ml-1 ${
              showFailureOverlay ? "text-accent-danger" : ""
            }`}
            onClick={() => setShowFailureOverlay((v) => !v)}
          >
            {showFailureOverlay ? "Hide" : "Show"} failure overlay
          </button>
        )}
        {showRouteMap && mapRoutes.length === 2 && mapLayout === "overlay" && (
          <button
            type="button"
            className={`btn-ghost text-xs ${
              showRouteDiff ? "text-accent-secondary" : ""
            }`}
            onClick={() => setShowRouteDiff((v) => !v)}
          >
            {showRouteDiff ? "Hide" : "Show"} route diff
          </button>
        )}
        {showRouteMap && routeMapMode === "deckgl" && hasBinCoords && (
          <button
            type="button"
            className={`btn-ghost text-xs ${
              showHexDensity ? "text-accent-secondary" : ""
            }`}
            onClick={() => setShowHexDensity((v) => !v)}
          >
            {showHexDensity ? "Hide" : "Show"} hex density
          </button>
        )}
        {showRouteMap && hasBinCoords && (
          <div className="flex items-center gap-1 bg-canvas-elevated rounded-lg p-0.5 ml-1">
            {(["echarts", "deckgl"] as const).map((m) => (
              <button
                key={m}
                type="button"
                onClick={() => setRouteMapMode(m)}
                className={`text-xs px-2 py-0.5 rounded-md ${
                  routeMapMode === m
                    ? "bg-accent-primary text-white"
                    : "text-canvas-muted hover:text-gray-200"
                }`}
              >
                {m === "echarts" ? "ECharts" : hasGeoCoords ? "Mercator" : "OrbitView"}
              </button>
            ))}
          </div>
        )}
      </div>

      {showRouteMap && mapRoutes.length > 0 ? (
        routeMapMode === "deckgl" && hasBinCoords ? (
          <Suspense
            fallback={<p className="text-xs text-canvas-muted">Loading tile map…</p>}
          >
            {mapLayout === "split" && mapRoutes.length === 2 ? (
              <div className="grid grid-cols-1 lg:grid-cols-2 gap-3">
                {mapRoutes.map((route) => (
                  <div key={route.id} className="space-y-1">
                    <p className="text-xs font-mono text-canvas-muted px-1">
                      {route.label}
                    </p>
                    <DeckRouteMap
                      routes={[route]}
                      animate={isPlaying}
                      playbackSpeed={playbackSpeed}
                      showFailureOverlay={showFailureOverlay}
                      showHexDensity={showHexDensity}
                    />
                  </div>
                ))}
              </div>
            ) : (
              <DeckRouteMap
                routes={mapRoutes}
                animate={isPlaying}
                playbackSpeed={playbackSpeed}
                showFailureOverlay={showFailureOverlay}
                showRouteDiff={showRouteDiff}
                showHexDensity={showHexDensity}
              />
            )}
          </Suspense>
        ) : mapLayout === "split" && mapRoutes.length === 2 ? (
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-3">
            {mapRoutes.map((route) => (
              <div key={route.id} className="space-y-1">
                <p className="text-xs font-mono text-canvas-muted px-1">
                  {route.label}
                </p>
                <RouteViz
                  data={route.data}
                  title="Route Map Preview"
                  subtitle={route.label}
                  filenameStem="route-map"
                  showFailureOverlay={showFailureOverlay}
                />
              </div>
            ))}
          </div>
        ) : mapRoutes.length === 2 && mapLayout === "overlay" ? (
          <RouteViz
            data={mapRoutes[0].data}
            compareData={mapRoutes[1].data}
            primaryLabel={mapRoutes[0].label}
            compareLabel={mapRoutes[1].label}
            title="Route Map Preview"
            subtitle={`Day ${displayDay} · overlay compare`}
            filenameStem="route-map-overlay"
            showFailureOverlay={showFailureOverlay}
            showTourDiff={showRouteDiff}
          />
        ) : displayEntry ? (
          <RouteViz
            data={displayEntry.data}
            title="Route Map Preview"
            subtitle={`Day ${displayDay} · ${displayEntry.policy}`}
            filenameStem="route-map"
            showFailureOverlay={showFailureOverlay}
          />
        ) : null
      ) : null}
    </>
  );
}
