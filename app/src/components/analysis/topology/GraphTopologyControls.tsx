import type EChartsReact from "echarts-for-react";
import { RefreshCw } from "lucide-react";
import { CanvasExportButton } from "../../common/CanvasExportButton";
import { ChartExportButtons } from "../../common/ChartExportButtons";
import { OpenPathToolbar } from "../../common/OpenPathToolbar";
import {
  DENSE_GRAPH_NODE_THRESHOLD,
  type DistanceMatrixData,
  type TopologyLayoutMode,
} from "../../../utils/graph/graphTopology";

export type TopologyView = "echarts" | "sigma" | "cosmograph";
export type PheromoneTimelineMode = "day" | "iteration";

interface GraphTopologyControlsProps {
  logPath: string | null;
  projectRoot: string | null;
  loading: boolean;
  loadMatrix: () => void;
  matrix: DistanceMatrixData | null;
  matrixPath: string | null;
  suggestRadial: boolean;
  layoutMode: TopologyLayoutMode | "auto";
  setLayoutMode: (m: TopologyLayoutMode | "auto") => void;
  kNeighbors: number;
  setKNeighbors: (k: number) => void;
  topologyView: TopologyView;
  setTopologyView: (v: TopologyView) => void;
  relayoutOnFilter: boolean;
  setRelayoutOnFilter: (v: boolean) => void;
  showPheromone: boolean;
  setShowPheromone: (v: boolean) => void;
  resolvedLayout: TopologyLayoutMode;
  setLayoutSeed: (fn: (s: number) => number) => void;
  hasGraphPayload: boolean;
  hasChartOption: boolean;
  chartRef: React.RefObject<EChartsReact | null>;
  sigmaContainerRef: React.RefObject<HTMLDivElement | null>;
  cosmographContainerRef: React.RefObject<HTMLDivElement | null>;
  tourCount: number;
  edgeCount: number;
  phEdgeCount: number;
  pheromoneMode: PheromoneTimelineMode;
  setPheromoneMode: (m: PheromoneTimelineMode) => void;
  timelineSync: boolean;
  setTimelineSync: (v: boolean) => void;
  displayDay: number;
  pheromoneDay: number;
  setPheromoneDay: (d: number) => void;
  dayRange: { min: number; max: number };
  handlePheromoneDay: (d: number) => void;
  maxTourSteps: number;
  pheromoneStep: number;
  setPheromoneStep: (s: number) => void;
  fillMin: number;
  setFillMin: (v: number) => void;
  fillMax: number;
  setFillMax: (v: number) => void;
  clearFillBrush: () => void;
  duckdbProfitRange: [number, number] | null;
}

export function GraphTopologyControls({
  logPath,
  projectRoot,
  loading,
  loadMatrix,
  matrix,
  matrixPath,
  suggestRadial,
  layoutMode,
  setLayoutMode,
  kNeighbors,
  setKNeighbors,
  topologyView,
  setTopologyView,
  relayoutOnFilter,
  setRelayoutOnFilter,
  showPheromone,
  setShowPheromone,
  resolvedLayout,
  setLayoutSeed,
  hasGraphPayload,
  hasChartOption,
  chartRef,
  sigmaContainerRef,
  cosmographContainerRef,
  tourCount,
  edgeCount,
  phEdgeCount,
  pheromoneMode,
  setPheromoneMode,
  timelineSync,
  setTimelineSync,
  displayDay,
  pheromoneDay,
  setPheromoneDay,
  dayRange,
  handlePheromoneDay,
  maxTourSteps,
  pheromoneStep,
  setPheromoneStep,
  fillMin,
  setFillMin,
  fillMax,
  setFillMax,
  clearFillBrush,
  duckdbProfitRange,
}: GraphTopologyControlsProps) {
  return (
    <>
      {!logPath && (
        <p className="text-xs text-canvas-muted">
          Open a simulation log to load the graph.
        </p>
      )}

      {logPath && (
        <div className="flex flex-wrap items-center gap-2">
          <button
            onClick={loadMatrix}
            disabled={loading}
            className="btn-ghost text-xs flex items-center gap-1"
          >
            <RefreshCw size={12} className={loading ? "animate-spin" : ""} />
            {loading ? "Loading…" : "Reload matrix"}
          </button>
          {matrix && matrixPath && (
            <OpenPathToolbar
              path={matrixPath}
              projectRoot={projectRoot}
              kind="csv"
              label={matrixPath.split(/[/\\]/).slice(-3).join("/")}
              chipClassName="text-[10px] max-w-xs"
              className="min-w-0"
            />
          )}
          {suggestRadial && layoutMode === "auto" && (
            <span className="text-[10px] text-amber-400/90">
              Auto radial (N≥{DENSE_GRAPH_NODE_THRESHOLD})
            </span>
          )}
        </div>
      )}

      {matrix && (
        <>
          <div className="flex flex-wrap items-center gap-3 text-xs">
            <label className="flex items-center gap-1.5 text-canvas-muted">
              k-NN
              <select
                className="select-base text-xs py-0.5 w-14"
                value={kNeighbors}
                onChange={(e) => setKNeighbors(Number(e.target.value))}
              >
                {[2, 3, 4, 5].map((k) => (
                  <option key={k} value={k}>
                    {k}
                  </option>
                ))}
              </select>
            </label>
            <label className="flex items-center gap-1.5 text-canvas-muted">
              Layout
              <select
                className="select-base text-xs py-0.5 w-24"
                value={layoutMode}
                onChange={(e) =>
                  setLayoutMode(e.target.value as TopologyLayoutMode | "auto")
                }
              >
                <option value="auto">Auto</option>
                <option value="force">Force</option>
                <option value="radial">Radial dense</option>
              </select>
            </label>
            <label className="flex items-center gap-1.5 text-canvas-muted">
              View
              <select
                className="select-base text-xs py-0.5 w-36"
                value={topologyView}
                onChange={(e) => setTopologyView(e.target.value as TopologyView)}
              >
                <option value="echarts">ECharts</option>
                <option value="sigma">Sigma.js WebGL</option>
                <option value="cosmograph">Cosmograph (WebGL)</option>
              </select>
            </label>
            <label className="flex items-center gap-1.5 text-canvas-muted">
              <input
                type="checkbox"
                checked={relayoutOnFilter}
                onChange={(e) => setRelayoutOnFilter(e.target.checked)}
                className="accent-accent-primary"
              />
              Re-layout on filter
            </label>
            <label className="flex items-center gap-1.5 text-canvas-muted">
              <input
                type="checkbox"
                checked={showPheromone}
                onChange={(e) => setShowPheromone(e.target.checked)}
                className="accent-accent-primary"
              />
              ACO pheromone trails
            </label>
            <button
              onClick={() => setLayoutSeed((s) => s + 1)}
              className="btn-ghost text-xs"
              disabled={resolvedLayout === "radial"}
            >
              Re-run layout
            </button>
            {hasGraphPayload && topologyView === "echarts" && hasChartOption && (
              <ChartExportButtons
                chartRef={{ current: chartRef.current }}
                filenameStem="topology-graph"
              />
            )}
            {hasGraphPayload && topologyView !== "echarts" && (
              <CanvasExportButton
                container={() =>
                  topologyView === "sigma"
                    ? sigmaContainerRef.current
                    : cosmographContainerRef.current
                }
                filename={
                  topologyView === "sigma"
                    ? "topology-sigma.png"
                    : "topology-cosmograph.png"
                }
              />
            )}
            <span className="text-canvas-muted">
              {tourCount} on tour · {edgeCount} edges
              {showPheromone ? ` · ${phEdgeCount} τ edges` : ""}
            </span>
          </div>

          {showPheromone && (
            <div className="space-y-1">
              <div className="flex flex-wrap items-center justify-between gap-2 text-[10px] text-canvas-muted">
                <span>
                  {pheromoneMode === "day"
                    ? "Pheromone timeline (tour edges ≤ day)"
                    : "Tour iteration stepping (τ per ant step)"}
                </span>
                <div className="flex items-center gap-2">
                  <label className="flex items-center gap-1">
                    Mode
                    <select
                      className="select-base text-xs py-0.5 w-24"
                      value={pheromoneMode}
                      onChange={(e) =>
                        setPheromoneMode(e.target.value as PheromoneTimelineMode)
                      }
                    >
                      <option value="day">By day</option>
                      <option value="iteration">By tour step</option>
                    </select>
                  </label>
                  <label className="flex items-center gap-1">
                    <input
                      type="checkbox"
                      checked={timelineSync}
                      onChange={(e) => {
                        const on = e.target.checked;
                        setTimelineSync(on);
                        if (on) setPheromoneDay(displayDay);
                      }}
                      className="accent-accent-primary"
                    />
                    Sync day scrubber
                  </label>
                </div>
              </div>
              <div className="flex items-center gap-2">
                <input
                  type="range"
                  min={dayRange.min}
                  max={dayRange.max}
                  value={pheromoneDay}
                  onChange={(e) => handlePheromoneDay(Number(e.target.value))}
                  className="flex-1 accent-amber-500"
                />
                <span className="text-xs font-mono text-amber-400/90 w-16 text-right">
                  Day {pheromoneDay}
                </span>
              </div>
              {pheromoneMode === "iteration" && (
                <div className="flex items-center gap-2">
                  <input
                    type="range"
                    min={maxTourSteps > 0 ? 1 : 0}
                    max={Math.max(maxTourSteps, 1)}
                    value={pheromoneStep}
                    onChange={(e) => setPheromoneStep(Number(e.target.value))}
                    className="flex-1 accent-amber-400"
                    disabled={maxTourSteps === 0}
                  />
                  <span className="text-xs font-mono text-amber-400/90 w-24 text-right">
                    Step {pheromoneStep}/{maxTourSteps}
                  </span>
                </div>
              )}
              <p className="text-[10px] text-canvas-muted">
                {pheromoneMode === "day"
                  ? "Edge warmth ∝ accumulated τ from consecutive tour stops (decay 0.88/day)."
                  : "Step through tour edge construction; each step deposits τ on the next consecutive stop pair."}
              </p>
            </div>
          )}

          <div className="space-y-1">
            <div className="flex items-center justify-between text-[10px] text-canvas-muted">
              <span>Fill % cross-filter (click node to brush)</span>
              <div className="flex items-center gap-2">
                <span>
                  {fillMin}% – {fillMax}%
                </span>
                {(fillMin > 0 || fillMax < 100) && (
                  <button
                    onClick={clearFillBrush}
                    className="text-accent-primary hover:underline"
                  >
                    Clear
                  </button>
                )}
              </div>
            </div>
            <div className="flex items-center gap-2">
              <input
                type="range"
                min={0}
                max={100}
                value={fillMin}
                onChange={(e) =>
                  setFillMin(Math.min(Number(e.target.value), fillMax))
                }
                className="flex-1 accent-accent-primary"
              />
              <input
                type="range"
                min={0}
                max={100}
                value={fillMax}
                onChange={(e) =>
                  setFillMax(Math.max(Number(e.target.value), fillMin))
                }
                className="flex-1 accent-accent-primary"
              />
            </div>
            {duckdbProfitRange && (
              <p className="text-[10px] text-accent-primary">
                DuckDB profit brush: €{duckdbProfitRange[0].toFixed(0)} – €
                {duckdbProfitRange[1].toFixed(0)} (day-level; tour nodes follow
                current day)
              </p>
            )}
          </div>

          <div className="flex flex-wrap gap-3 text-[10px] text-canvas-muted">
            <span className="flex items-center gap-1">
              <span className="w-2 h-2 rounded-full bg-emerald-400" /> On tour
            </span>
            <span className="flex items-center gap-1">
              <span className="w-2 h-2 rounded-full bg-red-400" /> Mandatory
            </span>
            <span className="flex items-center gap-1">
              <span className="w-2 h-2 rounded-full bg-indigo-400" /> In fill
              range
            </span>
            {showPheromone && (
              <span className="flex items-center gap-1">
                <span className="w-2 h-2 rounded-full bg-amber-400" /> High τ
                edge
              </span>
            )}
            <span className="flex items-center gap-1">
              <span className="w-2 h-2 rounded-full bg-slate-400" /> Idle
            </span>
          </div>
        </>
      )}
    </>
  );
}
