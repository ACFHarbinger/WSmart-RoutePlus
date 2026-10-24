/**
 * Simulation Digital Twin — real-time visualization of a running simulation (§G.16, Slice V5).
 *
 * Uses `useSimThrottle` (2 Hz visibility-aware throttle), `StalenessDot`,
 * `useLoadedRunStore` synchronization, and optional hexagonal spatial density overlay.
 */
import { useCallback, useEffect, useMemo, useState } from "react";
import { invoke } from "@tauri-apps/api/core";
import { open } from "@tauri-apps/plugin-dialog";
import { ChevronLeft, ChevronRight, FolderOpen, Pause, Play, RefreshCw } from "lucide-react";
import { GlobalFilterBar } from "../../components/layout/GlobalFilterBar";
import { KpiCard } from "../../components/ui/KpiCard";
import { StalenessDot } from "../../components/ui/telemetry/status";
import { useSimWatcher } from "../../hooks/process/useSimWatcher";
import { useSimThrottle } from "../../hooks/sim/useSimThrottle";
import { useAppStore } from "../../store/app";
import { useLoadedRunStore } from "../../store/loadedRun";
import { useRecentHandoff } from "../../hooks/files/useRecentHandoff";
import { useGlobalFiltersStore } from "../../store/filters";
import { useProcessStore } from "../../store/process";
import { useSimStore, uniquePolicies, uniqueSamples, filterEntries } from "../../store/sim";
import { OpenPathToolbar } from "../../components/common/OpenPathToolbar";
import { hexToRgb } from "../../utils/charts/colors";
import {
  enrichEntriesWithGraphCoords,
  GRAPH_PRESETS,
  guessGraphPreset,
  loadGraphCoordinates,
} from "../../utils/graph/graphCoords";
import { GraphTopologyPanel } from "../../components/analysis/topology/GraphTopologyPanel";
import { FailureAnalysisPanel } from "../../components/analysis/routes/FailureAnalysisPanel";
import { PolicyTelemetryPanel } from "../../components/analysis/telemetry/PolicyTelemetryPanel";
import { PolicyTelemetryTrendsPanel } from "../../components/analysis/telemetry/PolicyTelemetryTrendsPanel";
import { SqlQueryPanel } from "../../components/analysis/explorer/SqlQueryPanel";
import {
  formatPipelineTimingBadge,
  portfolioRunLabel,
  runSimulationArrowPipeline,
} from "../../utils/duckdb/arrowPipeline";
import { useDuckDbStore } from "../../store/duckdb";
import { toast } from "sonner";
import { filterFailureEntries } from "../../utils/sim/simFailure";
import { useLogPathRunLabelBrush } from "../../hooks/brush/useLogPathRunLabelBrush";
import {
  BinFillStrip,
  computeDelta,
  MetricTimeseries,
  POLICY_COLORS,
  PRIMARY_KPIS,
  SECONDARY_KPIS,
  TourTable,
} from "../../components/monitor/sim/SimMonitorPanels";
import { SimMonitorMapSection } from "../../components/monitor/sim/SimMonitorMapSection";
import type { DayLogEntry, PolicyVizEntry, SimFailureEntry } from "../../types";

export function SimulationMonitor() {
  const {
    entries: rawEntries,
    policyVizEntries,
    failureEntries,
    selectedDay,
    watchPath,
    isWatching,
    lastEventAt,
    loadEntries,
    loadPolicyVizEntries,
    loadFailureEntries,
    setSelectedPolicy,
    setSelectedSample,
    setSelectedDay,
    setWatchPath,
    reset,
  } = useSimStore();
  const entries = useSimThrottle(rawEntries, 500);
  const setLoadedRun = useLoadedRunStore((s) => s.setLoadedRun);

  const runningSimProcess = useProcessStore((s) =>
    Object.values(s.processes).find(
      (p) => p.status === "running" && p.command.includes("test_sim")
    )
  );
  const policyVizLive = isWatching || !!runningSimProcess;
  const [telemetryTrendsKey, setTelemetryTrendsKey] = useState(0);

  useEffect(() => {
    if (policyVizEntries.length > 0) {
      setTelemetryTrendsKey((k) => k + 1);
    }
  }, [policyVizEntries.length]);

  const {
    policy: selectedPolicy,
    sampleId: selectedSample,
    logScale,
    setPolicy,
    setSampleId,
  } = useGlobalFiltersStore();

  useEffect(() => {
    setSelectedPolicy(selectedPolicy);
    setSelectedSample(selectedSample);
  }, [selectedPolicy, selectedSample, setSelectedPolicy, setSelectedSample]);

  useSimWatcher(watchPath);

  const policies = useMemo(() => uniquePolicies(entries), [entries]);
  const samples = useMemo(() => uniqueSamples(entries), [entries]);

  const filteredEntries = useMemo(
    () => filterEntries(entries, selectedPolicy, selectedSample),
    [entries, selectedPolicy, selectedSample]
  );

  const dayRange = useMemo(() => {
    if (!filteredEntries.length) return { min: 0, max: 0 };
    const days = filteredEntries.map((e) => e.day);
    return { min: Math.min(...days), max: Math.max(...days) };
  }, [filteredEntries]);

  const displayDay = selectedDay ?? dayRange.max;
  const displayEntry = filteredEntries.find((e) => e.day === displayDay) ?? null;

  const displayFailureEntry = useMemo(
    () => filterFailureEntries(failureEntries, selectedPolicy, selectedSample, displayDay),
    [failureEntries, selectedPolicy, selectedSample, displayDay]
  );

  const {
    pendingLogPath,
    setPendingLogPath,
    pendingMapCompare,
    setPendingMapCompare,
    effectiveTheme: theme,
  } = useAppStore();
  const { projectRoot, handoff } = useRecentHandoff();
  const {
    ready: duckdbReady,
    loading: duckdbLoading,
    lastPipeline,
    setLastPipeline,
    setLoading: setDuckdbLoading,
  } = useDuckDbStore();

  const [activeLogPath, setActiveLogPath] = useState<string | null>(null);
  useLogPathRunLabelBrush(activeLogPath);
  const sourceRunLabel = useMemo(
    () => (activeLogPath ? portfolioRunLabel(activeLogPath, undefined, projectRoot) : null),
    [activeLogPath, projectRoot]
  );
  const [graphPreset, setGraphPreset] = useState(GRAPH_PRESETS[0].id);
  const [loadingGraphCoords, setLoadingGraphCoords] = useState(false);

  const loadLogFile = useCallback(
    async (path: string, watch = true) => {
      reset();
      const [historical, vizHistorical, failureHistorical] = await Promise.all([
        invoke<DayLogEntry[]>("load_simulation_log", { path }),
        invoke<PolicyVizEntry[]>("load_policy_viz_log", { path }).catch(() => []),
        invoke<SimFailureEntry[]>("load_sim_failure_log", { path }).catch(() => []),
      ]);
      loadEntries(historical);
      loadPolicyVizEntries(vizHistorical);
      loadFailureEntries(failureHistorical);
      setActiveLogPath(path);
      if (watch) setWatchPath(path);
      setLoadedRun({
        path,
        label: portfolioRunLabel(path, undefined, projectRoot) ?? "simulation_log",
        entries: historical,
        source: watch ? "live" : "viewing",
      });
      handoff(path, "log", { navigate: false });

      if (duckdbReady) {
        setDuckdbLoading(true);
        runSimulationArrowPipeline(path, "monitor_sim", projectRoot)
          .then(setLastPipeline)
          .catch((err) => console.warn("Simulation Arrow pipeline:", err))
          .finally(() => setDuckdbLoading(false));
      }
    },
    [
      reset,
      loadEntries,
      loadPolicyVizEntries,
      loadFailureEntries,
      setWatchPath,
      setLoadedRun,
      handoff,
      duckdbReady,
      projectRoot,
      setLastPipeline,
      setDuckdbLoading,
    ]
  );

  const openLog = useCallback(async () => {
    const path = (await open({
      filters: [{ name: "JSONL Logs", extensions: ["jsonl", "log", "txt"] }],
    })) as string | null;
    if (!path) return;
    await loadLogFile(path);
  }, [loadLogFile]);

  useEffect(() => {
    if (pendingLogPath) {
      void loadLogFile(pendingLogPath, false);
      setPendingLogPath(null);
    }
  }, [pendingLogPath, setPendingLogPath, loadLogFile]);

  const [showSecondary, setShowSecondary] = useState(false);
  const [showBinFill, setShowBinFill] = useState(true);
  const [showTourTable, setShowTourTable] = useState(false);
  const [showRouteMap, setShowRouteMap] = useState(true);
  const [routeMapMode, setRouteMapMode] = useState<"echarts" | "deckgl">("echarts");
  const [isPlaying, setIsPlaying] = useState(false);
  const [playbackSpeed, setPlaybackSpeed] = useState<1 | 2 | 4>(1);
  const [mapLayout, setMapLayout] = useState<"overlay" | "split">("overlay");
  const [showFailureOverlay, setShowFailureOverlay] = useState(true);
  const [showRouteDiff, setShowRouteDiff] = useState(false);
  const [showHexDensity, setShowHexDensity] = useState(false);
  const [duckdbProfitRange, setDuckdbProfitRange] = useState<[number, number] | null>(null);

  useEffect(() => {
    if (!activeLogPath || !entries.length) return;
    const guessed = guessGraphPreset(activeLogPath, entries);
    if (guessed) setGraphPreset(guessed);
  }, [activeLogPath, entries]);

  const applyGraphCoords = useCallback(async () => {
    if (!projectRoot) {
      toast.error("Set project root in Settings to load graph coordinates");
      return;
    }
    if (!entries.length) return;
    setLoadingGraphCoords(true);
    try {
      const sampleIndex = selectedSample ?? 0;
      const table = await loadGraphCoordinates(projectRoot, graphPreset, sampleIndex);
      loadEntries(enrichEntriesWithGraphCoords(entries, table));
      setRouteMapMode("deckgl");
      toast.success("Graph coordinates loaded", {
        description: GRAPH_PRESETS.find((p) => p.id === graphPreset)?.label,
      });
    } catch (err) {
      toast.error("Failed to load graph coordinates", { description: String(err) });
    } finally {
      setLoadingGraphCoords(false);
    }
  }, [projectRoot, entries, graphPreset, selectedSample, loadEntries]);

  const hasGeoCoords = useMemo(
    () => displayEntry?.data.all_bin_coords?.some((b) => b.lat != null && b.lng != null) ?? false,
    [displayEntry]
  );

  const hasBinCoords = useMemo(
    () => (displayEntry?.data.all_bin_coords?.length ?? 0) > 0,
    [displayEntry]
  );

  const [chartPolicies, setChartPolicies] = useState<string[]>([]);
  const activeChartPolicies = useMemo(
    () => (chartPolicies.length > 0 ? chartPolicies : policies),
    [chartPolicies, policies]
  );

  const toggleChartPolicy = useCallback(
    (p: string) => {
      setChartPolicies((prev) => {
        const active = prev.length > 0 ? prev : policies;
        if (active.includes(p)) {
          const next = active.filter((x) => x !== p);
          return next.length === 0 ? policies : next;
        }
        return [...active, p];
      });
    },
    [policies]
  );

  const [mapPolicies, setMapPolicies] = useState<string[]>([]);
  const activeMapPolicies = useMemo(
    () => (mapPolicies.length > 0 ? mapPolicies : policies),
    [mapPolicies, policies]
  );

  const toggleMapPolicy = useCallback(
    (p: string) => {
      setMapPolicies((prev) => {
        const active = prev.length > 0 ? prev : policies;
        if (active.includes(p)) {
          const next = active.filter((x) => x !== p);
          return next.length === 0 ? policies : next;
        }
        return [...active, p];
      });
    },
    [policies]
  );

  useEffect(() => {
    if (!pendingMapCompare) return;
    setShowRouteMap(true);
    setMapPolicies(pendingMapCompare.policies);
    setMapLayout(pendingMapCompare.layout);
    if (pendingMapCompare.mapMode) setRouteMapMode(pendingMapCompare.mapMode);
    setPendingMapCompare(null);
  }, [pendingMapCompare, setPendingMapCompare]);

  const mapRoutes = useMemo(() => {
    const dayEntries = entries.filter(
      (e) =>
        e.day === displayDay &&
        (selectedSample === null || e.sample_id === selectedSample) &&
        activeMapPolicies.includes(e.policy)
    );
    return dayEntries
      .filter((e) => e.data.all_bin_coords?.length)
      .map((e) => {
        const idx = policies.indexOf(e.policy);
        return {
          id: `${e.policy}-${e.sample_id}`,
          label: e.policy,
          data: e.data,
          color: hexToRgb(POLICY_COLORS[idx % POLICY_COLORS.length]),
        };
      });
  }, [entries, displayDay, selectedSample, activeMapPolicies, policies]);

  const mapHasFailureData = useMemo(
    () =>
      mapRoutes.some(
        (r) =>
          r.data.failure_analysis?.has_failure &&
          ((r.data.failure_analysis.overflow_bins?.length ?? 0) > 0 ||
            (r.data.failure_analysis.skipped_high_fill_bins?.length ?? 0) > 0)
      ),
    [mapRoutes]
  );

  useEffect(() => {
    if (!isPlaying || dayRange.max <= dayRange.min) return;
    const intervalMs = 800 / playbackSpeed;
    const timer = window.setInterval(() => {
      const current = selectedDay ?? dayRange.max;
      if (current >= dayRange.max) {
        setSelectedDay(dayRange.min);
      } else {
        setSelectedDay(current + 1);
      }
    }, intervalMs);
    return () => window.clearInterval(timer);
  }, [isPlaying, playbackSpeed, dayRange.min, dayRange.max, selectedDay, setSelectedDay]);

  const policySeries = useMemo(
    () =>
      activeChartPolicies.map((p, i) => ({
        policy: p,
        color: POLICY_COLORS[i % POLICY_COLORS.length],
        entries: filterEntries(entries, p, selectedSample),
      })),
    [activeChartPolicies, entries, selectedSample]
  );

  return (
    <div className="space-y-4">
      {entries.length > 0 && (
        <GlobalFilterBar
          showLogScale
          runLabels={sourceRunLabel ? [sourceRunLabel] : []}
        />
      )}

      {/* Header controls */}
      <div className="flex items-center gap-3 flex-wrap">
        <button onClick={openLog} className="btn-primary flex items-center gap-2">
          <FolderOpen size={14} />
          Open Log File
        </button>

        {isWatching && <StalenessDot lastEventAt={lastEventAt} />}

        {watchPath && (
          <OpenPathToolbar
            path={watchPath}
            projectRoot={projectRoot}
            kind="log"
            labeled
            labeledTargets={["summary"]}
            trailing={
              <>
                {isWatching && (
                  <RefreshCw size={11} className="animate-spin text-accent-success shrink-0" />
                )}
                {duckdbLoading && <span className="shrink-0">· DuckDB ingesting…</span>}
                {!duckdbLoading && lastPipeline?.tableName === "monitor_sim" && (
                  <span className="shrink-0">· {formatPipelineTimingBadge(lastPipeline)}</span>
                )}
              </>
            }
          />
        )}

        {policies.length > 0 && (
          <select
            className="select-base w-40"
            value={selectedPolicy ?? ""}
            onChange={(e) => setPolicy(e.target.value || null)}
          >
            <option value="">All policies</option>
            {policies.map((p) => (
              <option key={p} value={p}>{p}</option>
            ))}
          </select>
        )}

        {samples.length > 1 && (
          <select
            className="select-base w-32"
            value={selectedSample ?? ""}
            onChange={(e) => setSampleId(e.target.value ? Number(e.target.value) : null)}
          >
            <option value="">All samples</option>
            {samples.map((s) => (
              <option key={s} value={s}>Sample {s}</option>
            ))}
          </select>
        )}

        {dayRange.max > 0 && (
          <div className="flex items-center gap-1.5 ml-auto">
            <span className="text-xs text-canvas-muted">Day</span>
            <button
              onClick={() => setSelectedDay(Math.max(dayRange.min, displayDay - 1))}
              disabled={displayDay <= dayRange.min}
              className="btn-ghost p-1 disabled:opacity-30"
              title="Previous day"
            >
              <ChevronLeft size={12} />
            </button>
            <input
              type="range"
              min={dayRange.min}
              max={dayRange.max}
              value={displayDay}
              onChange={(e) => setSelectedDay(Number(e.target.value))}
              className="w-28 accent-accent-primary"
            />
            <button
              onClick={() => setSelectedDay(Math.min(dayRange.max, displayDay + 1))}
              disabled={displayDay >= dayRange.max}
              className="btn-ghost p-1 disabled:opacity-30"
              title="Next day"
            >
              <ChevronRight size={12} />
            </button>
            <span className="text-xs font-mono text-canvas-muted w-14 text-center tabular-nums">
              {displayDay}/{dayRange.max}
            </span>
            <button
              onClick={() => {
                if (isPlaying) {
                  setIsPlaying(false);
                } else {
                  if (selectedDay === null) setSelectedDay(dayRange.min);
                  setIsPlaying(true);
                }
              }}
              className="btn-ghost p-1"
              title={isPlaying ? "Pause day playback" : "Play through simulation days"}
            >
              {isPlaying ? <Pause size={12} /> : <Play size={12} />}
            </button>
            <select
              className="select-base text-[10px] py-0.5 w-14"
              value={playbackSpeed}
              onChange={(e) => setPlaybackSpeed(Number(e.target.value) as 1 | 2 | 4)}
              title="Playback speed"
            >
              <option value={1}>1×</option>
              <option value={2}>2×</option>
              <option value={4}>4×</option>
            </select>
            {selectedDay !== null ? (
              <button
                onClick={() => {
                  setIsPlaying(false);
                  setSelectedDay(null);
                }}
                className="btn-ghost text-xs text-accent-primary"
                title="Jump to latest day and follow live updates"
              >
                Latest ↓
              </button>
            ) : isWatching ? (
              <span className="flex items-center gap-1 text-xs text-accent-success">
                <span className="w-1.5 h-1.5 rounded-full bg-accent-success animate-pulse" />
                Following
              </span>
            ) : null}
          </div>
        )}
      </div>

      {policies.length > 1 && (
        <div className="flex items-center gap-2 flex-wrap">
          <span className="text-xs text-canvas-muted shrink-0">Charts:</span>
          {policies.map((p, i) => {
            const color = POLICY_COLORS[i % POLICY_COLORS.length];
            const active = activeChartPolicies.includes(p);
            return (
              <button
                key={p}
                onClick={() => toggleChartPolicy(p)}
                className={`text-xs px-2 py-0.5 rounded-full border transition-opacity ${
                  active ? "opacity-100" : "opacity-35"
                }`}
                style={{
                  borderColor: color,
                  color: active ? color : undefined,
                  background: active ? `${color}18` : undefined,
                }}
              >
                {p}
              </button>
            );
          })}
        </div>
      )}

      {!displayEntry && (
        <div className="flex flex-col items-center justify-center h-64 text-canvas-muted gap-3">
          <p className="text-sm">Open a simulation log file to begin monitoring.</p>
          <p className="text-xs">
            The watcher streams new days in real-time as the simulation runs.
          </p>
        </div>
      )}

      {displayEntry && (
        <>
          <div className="grid grid-cols-2 sm:grid-cols-4 gap-3">
            {PRIMARY_KPIS.map(({ key, label, lowerIsBetter }) => (
              <KpiCard
                key={key}
                label={label}
                value={(displayEntry.data as Record<string, number>)[key]}
                delta={computeDelta(filteredEntries, displayDay, key)}
                colorize
                lowerIsBetter={lowerIsBetter}
              />
            ))}
          </div>

          <button
            className="btn-ghost text-xs"
            onClick={() => setShowSecondary((v) => !v)}
          >
            {showSecondary ? "Hide" : "Show"} secondary metrics
          </button>
          {showSecondary && (
            <div className="grid grid-cols-2 sm:grid-cols-4 gap-3">
              {SECONDARY_KPIS.map(({ key, label, lowerIsBetter }) => (
                <KpiCard
                  key={key}
                  label={label}
                  value={(displayEntry.data as Record<string, number>)[key]}
                  delta={computeDelta(filteredEntries, displayDay, key)}
                  colorize
                  lowerIsBetter={lowerIsBetter}
                />
              ))}
            </div>
          )}

          <p className="text-[10px] text-canvas-muted">
            {logScale
              ? "Daily KPI timeseries · log-scale (symlog overflows, log profit/km/kg)"
              : "Daily KPI timeseries · linear scale"}
          </p>
          <div className="grid grid-cols-2 lg:grid-cols-4 gap-3">
            {PRIMARY_KPIS.map(({ key, label }) => (
              <MetricTimeseries
                key={key}
                policySeries={policySeries}
                metricKey={key}
                label={label}
                logScale={logScale}
              />
            ))}
          </div>

          <SimMonitorMapSection
            showBinFill={showBinFill}
            setShowBinFill={setShowBinFill}
            showTourTable={showTourTable}
            setShowTourTable={setShowTourTable}
            showRouteMap={showRouteMap}
            setShowRouteMap={setShowRouteMap}
            policies={policies}
            activeMapPolicies={activeMapPolicies}
            toggleMapPolicy={toggleMapPolicy}
            mapRoutes={mapRoutes}
            mapLayout={mapLayout}
            setMapLayout={setMapLayout}
            hasGeoCoords={hasGeoCoords}
            hasBinCoords={hasBinCoords}
            entries={entries}
            graphPreset={graphPreset}
            setGraphPreset={setGraphPreset}
            activeLogPath={activeLogPath}
            applyGraphCoords={applyGraphCoords}
            loadingGraphCoords={loadingGraphCoords}
            mapHasFailureData={mapHasFailureData}
            showFailureOverlay={showFailureOverlay}
            setShowFailureOverlay={setShowFailureOverlay}
            showRouteDiff={showRouteDiff}
            setShowRouteDiff={setShowRouteDiff}
            showHexDensity={showHexDensity}
            setShowHexDensity={setShowHexDensity}
            routeMapMode={routeMapMode}
            setRouteMapMode={setRouteMapMode}
            isPlaying={isPlaying}
            playbackSpeed={playbackSpeed}
            displayDay={displayDay}
            displayEntry={displayEntry}
          />

          {showBinFill && <BinFillStrip data={displayEntry.data} />}
          {showTourTable && (
            <TourTable
              data={displayEntry.data}
              day={displayDay}
              policy={displayEntry.policy}
            />
          )}

          <FailureAnalysisPanel
            entry={displayFailureEntry}
            embedded={displayEntry?.data.failure_analysis ?? null}
          />

          <PolicyTelemetryPanel
            entries={policyVizEntries}
            policy={selectedPolicy}
            sampleId={selectedSample}
            day={displayDay}
            theme={theme}
            logScale={logScale}
            live={policyVizLive}
          />

          <PolicyTelemetryTrendsPanel
            theme={theme}
            logScale={logScale}
            refreshKey={telemetryTrendsKey}
            initialPolicy={selectedPolicy}
            initialRunLabel={sourceRunLabel}
          />

          <GraphTopologyPanel
            logPath={activeLogPath}
            projectRoot={projectRoot}
            simData={displayEntry?.data ?? null}
            theme={theme}
            duckdbProfitRange={duckdbProfitRange}
            entries={filteredEntries}
            displayDay={displayDay}
            dayRange={dayRange}
            onDaySelect={setSelectedDay}
            logScale={logScale}
          />

          {duckdbReady && lastPipeline?.tableName === "monitor_sim" && (
            <SqlQueryPanel
              tableName={lastPipeline.tableName}
              theme={theme}
              onDaySelect={(day) => setSelectedDay(day)}
              onProfitRange={(min, max) => setDuckdbProfitRange([min, max])}
              highlightPolicies={selectedPolicy ? [selectedPolicy] : null}
              highlightRunLabels={sourceRunLabel ? [sourceRunLabel] : null}
              brushSqlSync
              autoRunOnBrushSync
              portfolioMode
              portfolioRunLabels={sourceRunLabel ? [sourceRunLabel] : []}
            />
          )}
        </>
      )}
    </div>
  );
}
