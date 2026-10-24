/**
 * Topological graph analytics panel — ECharts force-directed graph (§G.4).
 */
import { lazy, Suspense, useCallback, useEffect, useMemo, useRef, useState } from "react";
import ReactECharts from "echarts-for-react";
import type EChartsReact from "echarts-for-react";
import { ChevronDown, ChevronUp } from "lucide-react";
import { toast } from "sonner";
import type { DayLogEntry, SimDayData } from "../../../types";
import {
  accumulateTourPheromone,
  accumulateTourPheromoneByStep,
  buildTopologyFromMatrix,
  countTourEdgeSteps,
  DENSE_GRAPH_NODE_THRESHOLD,
  loadDistanceMatrixForLog,
  resolveLayoutMode,
  type DistanceMatrixData,
  type TopologyLayoutMode,
  type TopologyNodeMeta,
} from "../../../utils/graph/graphTopology";
import {
  GraphTopologyControls,
  type PheromoneTimelineMode,
  type TopologyView,
} from "./GraphTopologyControls";

const TopologySigmaView = lazy(() =>
  import("./TopologySigmaView").then((m) => ({ default: m.TopologySigmaView }))
);
const TopologyCosmographView = lazy(() =>
  import("./TopologyCosmographView").then((m) => ({ default: m.TopologyCosmographView }))
);

interface Props {
  logPath: string | null;
  projectRoot: string | null;
  simData: SimDayData | null;
  theme: "dark" | "light";
  duckdbProfitRange?: [number, number] | null;
  entries?: DayLogEntry[];
  displayDay?: number;
  dayRange?: { min: number; max: number };
  syncTimeline?: boolean;
  onDaySelect?: (day: number) => void;
  logScale?: boolean;
}

export function GraphTopologyPanel({
  logPath,
  projectRoot,
  simData,
  theme,
  duckdbProfitRange = null,
  entries = [],
  displayDay = 1,
  dayRange = { min: 1, max: 1 },
  syncTimeline = true,
  onDaySelect,
  logScale = false,
}: Props) {
  const chartRef = useRef<EChartsReact | null>(null);
  const sigmaContainerRef = useRef<HTMLDivElement>(null);
  const cosmographContainerRef = useRef<HTMLDivElement>(null);
  const [open, setOpen] = useState(false);
  const [loading, setLoading] = useState(false);
  const [matrix, setMatrix] = useState<DistanceMatrixData | null>(null);
  const [matrixPath, setMatrixPath] = useState<string | null>(null);
  const [kNeighbors, setKNeighbors] = useState(3);
  const [relayoutOnFilter, setRelayoutOnFilter] = useState(false);
  const [fillMin, setFillMin] = useState(0);
  const [fillMax, setFillMax] = useState(100);
  const [layoutSeed, setLayoutSeed] = useState(0);
  const [layoutMode, setLayoutMode] = useState<TopologyLayoutMode | "auto">("auto");
  const [showPheromone, setShowPheromone] = useState(false);
  const [pheromoneDay, setPheromoneDay] = useState(displayDay);
  const [pheromoneMode, setPheromoneMode] = useState<PheromoneTimelineMode>("day");
  const [pheromoneStep, setPheromoneStep] = useState(1);
  const [timelineSync, setTimelineSync] = useState(syncTimeline);
  const [topologyView, setTopologyView] = useState<TopologyView>("echarts");

  const loadMatrix = useCallback(async () => {
    if (!logPath) return;
    setLoading(true);
    try {
      const { data, path } = await loadDistanceMatrixForLog(logPath, projectRoot);
      setMatrix(data);
      setMatrixPath(path);
      toast.success("Distance matrix loaded", {
        description: `${data.nodeIds.length} nodes`,
      });
    } catch (err) {
      setMatrix(null);
      setMatrixPath(null);
      toast.error("Failed to load distance matrix", { description: String(err) });
    } finally {
      setLoading(false);
    }
  }, [logPath, projectRoot]);

  useEffect(() => {
    if (open && logPath && !matrix && !loading) {
      void loadMatrix();
    }
  }, [open, logPath, matrix, loading, loadMatrix]);

  useEffect(() => {
    setTimelineSync(syncTimeline);
  }, [syncTimeline]);

  useEffect(() => {
    if (timelineSync) setPheromoneDay(displayDay);
  }, [displayDay, timelineSync]);

  const fillRange = useMemo((): [number, number] | null => {
    if (fillMin <= 0 && fillMax >= 100) return null;
    return [fillMin, fillMax];
  }, [fillMin, fillMax]);

  const pheromoneDayEntry = useMemo(
    () => entries.find((e) => e.day === pheromoneDay),
    [entries, pheromoneDay]
  );

  const maxTourSteps = useMemo(
    () => countTourEdgeSteps(pheromoneDayEntry, matrix?.nodeIds ?? []),
    [pheromoneDayEntry, matrix]
  );

  useEffect(() => {
    if (pheromoneStep > maxTourSteps) {
      setPheromoneStep(Math.max(1, maxTourSteps));
    }
  }, [maxTourSteps, pheromoneStep]);

  const pheromoneWeights = useMemo(() => {
    if (!showPheromone || !matrix || !entries.length) return new Map<string, number>();
    if (pheromoneMode === "iteration" && pheromoneDayEntry) {
      return accumulateTourPheromoneByStep(
        pheromoneDayEntry,
        matrix.nodeIds,
        pheromoneStep
      );
    }
    return accumulateTourPheromone(entries, matrix.nodeIds, pheromoneDay);
  }, [
    showPheromone,
    matrix,
    entries,
    pheromoneDay,
    pheromoneMode,
    pheromoneDayEntry,
    pheromoneStep,
  ]);

  const resolvedLayout = matrix
    ? resolveLayoutMode(matrix.nodeIds.length, layoutMode)
    : "force";

  const graphPayload = useMemo(() => {
    if (!matrix) return null;
    return buildTopologyFromMatrix(matrix, simData, {
      kNeighbors,
      relayoutOnFilter,
      fillRange,
      layoutIterations: 60 + (layoutSeed % 3) * 20,
      layoutMode,
      pheromoneWeights,
      showPheromone,
      logScale,
      theme,
    });
  }, [
    matrix,
    simData,
    kNeighbors,
    relayoutOnFilter,
    fillRange,
    layoutSeed,
    layoutMode,
    pheromoneWeights,
    showPheromone,
    logScale,
    theme,
  ]);

  const chartOption = useMemo(() => {
    if (!graphPayload || !matrix) return null;
    const layoutLabel = resolvedLayout === "radial" ? "radial dense" : "force";
    const phLabel = showPheromone
      ? pheromoneMode === "iteration"
        ? ` · τ step ${pheromoneStep}/${maxTourSteps || 0} (day ${pheromoneDay})`
        : ` · τ≤day ${pheromoneDay}`
      : "";
    const title = matrixPath
      ? `Topology · ${matrix.nodeIds.length} nodes · k=${kNeighbors} · ${layoutLabel}${phLabel}`
      : undefined;
    return {
      ...graphPayload.option,
      title: title
        ? {
            text: title,
            left: "center",
            top: 4,
            textStyle: { fontSize: 11, color: theme === "dark" ? "#9ca3af" : "#6b7280" },
          }
        : undefined,
    };
  }, [
    graphPayload,
    matrix,
    matrixPath,
    kNeighbors,
    resolvedLayout,
    showPheromone,
    pheromoneDay,
    pheromoneMode,
    pheromoneStep,
    maxTourSteps,
    theme,
  ]);

  const tourCount = graphPayload?.nodeMeta.filter((n) => n.onTour).length ?? 0;
  const phEdgeCount = showPheromone
    ? [...pheromoneWeights.values()].filter((v) => v > 0).length
    : 0;
  const suggestRadial = (matrix?.nodeIds.length ?? 0) >= DENSE_GRAPH_NODE_THRESHOLD;

  const handlePheromoneDay = (day: number) => {
    setPheromoneDay(day);
    if (timelineSync) onDaySelect?.(day);
  };

  const brushNodeFill = useCallback((meta: TopologyNodeMeta) => {
    if (meta.fillPct == null) return;
    const pad = 5;
    setFillMin(Math.max(0, meta.fillPct - pad));
    setFillMax(Math.min(100, meta.fillPct + pad));
  }, []);

  const clearFillBrush = useCallback(() => {
    setFillMin(0);
    setFillMax(100);
  }, []);

  const topologyClickEvents = useMemo(
    () => ({
      click: (params: { dataType?: string; data?: { id?: string } }) => {
        if (params.dataType !== "node" || params.data?.id == null) return;
        const meta = graphPayload?.nodeMeta[Number(params.data.id)];
        if (meta) brushNodeFill(meta);
      },
    }),
    [graphPayload, brushNodeFill]
  );

  return (
    <div className="card space-y-3">
      <button
        onClick={() => setOpen((v) => !v)}
        className="flex items-center justify-between w-full text-left"
      >
        <div>
          <p className="text-xs font-semibold text-gray-300">Graph Topology (§G.4)</p>
          <p className="text-[10px] text-canvas-muted">
            Distance matrix → k-NN edges · force/radial layout · ACO pheromone trails
            {logScale ? " · log-scale τ" : ""}
          </p>
        </div>
        {open ? <ChevronUp size={14} /> : <ChevronDown size={14} />}
      </button>

      {open && (
        <>
          <GraphTopologyControls
            logPath={logPath}
            projectRoot={projectRoot}
            loading={loading}
            loadMatrix={() => void loadMatrix()}
            matrix={matrix}
            matrixPath={matrixPath}
            suggestRadial={suggestRadial}
            layoutMode={layoutMode}
            setLayoutMode={setLayoutMode}
            kNeighbors={kNeighbors}
            setKNeighbors={setKNeighbors}
            topologyView={topologyView}
            setTopologyView={setTopologyView}
            relayoutOnFilter={relayoutOnFilter}
            setRelayoutOnFilter={setRelayoutOnFilter}
            showPheromone={showPheromone}
            setShowPheromone={setShowPheromone}
            resolvedLayout={resolvedLayout}
            setLayoutSeed={setLayoutSeed}
            hasGraphPayload={Boolean(graphPayload)}
            hasChartOption={Boolean(chartOption)}
            chartRef={chartRef}
            sigmaContainerRef={sigmaContainerRef}
            cosmographContainerRef={cosmographContainerRef}
            tourCount={tourCount}
            edgeCount={graphPayload?.edges.length ?? 0}
            phEdgeCount={phEdgeCount}
            pheromoneMode={pheromoneMode}
            setPheromoneMode={setPheromoneMode}
            timelineSync={timelineSync}
            setTimelineSync={setTimelineSync}
            displayDay={displayDay}
            pheromoneDay={pheromoneDay}
            setPheromoneDay={setPheromoneDay}
            dayRange={dayRange}
            handlePheromoneDay={handlePheromoneDay}
            maxTourSteps={maxTourSteps}
            pheromoneStep={pheromoneStep}
            setPheromoneStep={setPheromoneStep}
            fillMin={fillMin}
            setFillMin={setFillMin}
            fillMax={fillMax}
            setFillMax={setFillMax}
            clearFillBrush={clearFillBrush}
            duckdbProfitRange={duckdbProfitRange}
          />

          {matrix && (
            <>
              {graphPayload && topologyView === "sigma" && (
                <div ref={sigmaContainerRef}>
                  <Suspense
                    fallback={
                      <div className="h-[360px] flex items-center justify-center text-xs text-canvas-muted">
                        Loading Sigma.js…
                      </div>
                    }
                  >
                    <TopologySigmaView
                      nodeMeta={graphPayload.nodeMeta}
                      edges={graphPayload.edges}
                      positions={graphPayload.positions}
                      layoutMode={resolvedLayout}
                      fillRange={fillRange}
                      pheromoneWeights={pheromoneWeights}
                      showPheromone={showPheromone}
                      logScale={logScale}
                      theme={theme}
                      onNodeClick={brushNodeFill}
                    />
                  </Suspense>
                </div>
              )}
              {graphPayload && topologyView === "cosmograph" && (
                <div ref={cosmographContainerRef}>
                  <Suspense
                    fallback={
                      <div className="h-[360px] flex items-center justify-center text-xs text-canvas-muted">
                        Loading Cosmograph view…
                      </div>
                    }
                  >
                    <TopologyCosmographView
                      nodeMeta={graphPayload.nodeMeta}
                      edges={graphPayload.edges}
                      positions={graphPayload.positions}
                      fillRange={fillRange}
                      pheromoneWeights={pheromoneWeights}
                      showPheromone={showPheromone}
                      logScale={logScale}
                      theme={theme}
                      onNodeClick={brushNodeFill}
                    />
                  </Suspense>
                </div>
              )}
              {chartOption && topologyView === "echarts" && (
                <ReactECharts
                  ref={chartRef}
                  option={chartOption}
                  style={{ height: 360 }}
                  notMerge
                  onEvents={topologyClickEvents}
                />
              )}
            </>
          )}
        </>
      )}
    </div>
  );
}
