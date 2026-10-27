/**
 * Simulation Digital Twin — real-time visualization of a running simulation (§G.16, Slice V5).
 *
 * Uses `useSimThrottle` (2 Hz visibility-aware throttle), `StalenessDot`,
 * `useLoadedRunStore` synchronization, and optional hexagonal spatial density overlay.
 */
import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { invoke } from "@tauri-apps/api/core";
import { open } from "@tauri-apps/plugin-dialog";
import { FolderOpen, FolderTree, RefreshCw } from "lucide-react";
import { GlobalFilterBar } from "../../components/layout/GlobalFilterBar";
import { StalenessDot } from "../../components/ui/telemetry/status";
import { useSimWatcher } from "../../hooks/process/useSimWatcher";
import { useSimThrottle } from "../../hooks/sim/useSimThrottle";
import { useAppStore } from "../../store/app";
import { useLoadedRunStore } from "../../store/loadedRun";
import { useRecentHandoff } from "../../hooks/files/useRecentHandoff";
import { useGlobalFiltersStore } from "../../store/filters";
import { useProcessStore } from "../../store/process";
import { useSimStore, uniquePolicies, uniqueSamples, filterEntries } from "../../store/sim";
import { useCorpusSceneState, useSimulationCorpusStore } from "../../store/simulationCorpus";
import { OpenPathToolbar } from "../../components/common/OpenPathToolbar";
import {
  enrichEntriesWithGraphCoords,
  GRAPH_PRESETS,
  guessGraphPreset,
  loadGraphCoordinates,
} from "../../utils/graph/graphCoords";
import { formatPipelineTimingBadge, portfolioRunLabel, runSimulationArrowPipeline } from "../../utils/duckdb/arrowPipeline";
import { useDuckDbStore } from "../../store/duckdb";
import { toast } from "sonner";
import { filterFailureEntries } from "../../utils/sim/simFailure";
import { useLogPathRunLabelBrush } from "../../hooks/brush/useLogPathRunLabelBrush";
import { POLICY_COLORS } from "../../components/monitor/sim/SimMonitorPanels";
import { SimMonitorMapSection } from "../../components/monitor/sim/SimMonitorMapSection";
import { SimMonitorAdvancedPanels, SimMonitorKpiSection } from "../../components/monitor/sim/SimMonitorDetailPanels";
import { SimMonitorChartPolicySelector, SimMonitorDayScrubber } from "../../components/monitor/sim/SimMonitorHeaderControls";
import { useSimMonitorMapRoutes } from "../../components/monitor/sim/useSimMonitorMapRoutes";
import { CorpusRouteMap } from "../../components/analysis/summary/corpus/CorpusRouteMap";
import { applyCorpusMapFrame } from "../../utils/corpus/monitorRoutes";
import { corpusMapFrame } from "../../utils/corpus/routeFrame";
import type {
  DayLogEntry,
  DiscoveredSimulationRun,
  PolicyVizEntry,
  SimFailureEntry,
  SimulationDirectoryCatalog,
} from "../../types";

/** Catalog `key` is the run directory. Older payloads only have each log's `path`. */
function simulationRunPath(run: DiscoveredSimulationRun, rootDir: string): string {
  if (run.key) return run.key;
  const logPath = run.logs.find((log) => log.path)?.path;
  if (logPath) return logPath;
  if (!rootDir) {
    throw new Error("No simulation log path to load");
  }
  return rootDir;
}

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
    if (policyVizEntries.length > 0) setTelemetryTrendsKey((k) => k + 1);
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
      if (!path) {
        throw new Error("No simulation log path to load");
      }
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

  const [catalog, setCatalog] = useState<SimulationDirectoryCatalog | null>(null);
  const [selectedRunKey, setSelectedRunKey] = useState<string | null>(null);

  const loadDirectory = useCallback(
    async (dirPath: string) => {
      try {
        const cat = await invoke<SimulationDirectoryCatalog>("scan_simulation_directory", {
          dirPath,
        });
        if (!cat.runs || cat.runs.length === 0) {
          toast.error("No simulation logs found in directory", { description: dirPath });
          return;
        }
        setCatalog(cat);
        const firstRun = cat.runs[0];
        const runPath = simulationRunPath(firstRun, cat.root_dir);
        setSelectedRunKey(runPath);
        await loadLogFile(runPath);
        toast.success(`Loaded catalog: ${cat.runs.length} run(s), ${cat.total_logs} log(s)`);
      } catch (err) {
        toast.error("Failed to scan simulation directory", { description: String(err) });
      }
    },
    [loadLogFile]
  );

  const openLog = useCallback(async () => {
    const path = (await open({
      filters: [
        { name: "Simulation Logs", extensions: ["json", "jsonl", "log", "txt"] },
        { name: "All Files", extensions: ["*"] },
      ],
    })) as string | null;
    if (!path) return;
    setCatalog(null);
    setSelectedRunKey(null);
    await loadLogFile(path);
  }, [loadLogFile]);

  const openDirectory = useCallback(async () => {
    const selected = (await open({
      directory: true,
      multiple: false,
    })) as string | null;
    if (!selected) return;
    await loadDirectory(selected);
  }, [loadDirectory]);

  const handleRunChange = useCallback(
    async (runKey: string) => {
      setSelectedRunKey(runKey);
      if (runKey === "__ALL__" && catalog) {
        await loadLogFile(catalog.root_dir);
      } else if (runKey) {
        await loadLogFile(runKey);
      }
    },
    [catalog, loadLogFile]
  );

  useEffect(() => {
    if (pendingLogPath) {
      const p = pendingLogPath;
      setPendingLogPath(null);
      invoke<boolean>("path_exists", { path: p }).then((exists) => {
        if (!exists) return;
        invoke<SimulationDirectoryCatalog>("scan_simulation_directory", { dirPath: p })
          .then((cat) => {
            if (cat.runs && cat.runs.length > 0) {
              setCatalog(cat);
              const runPath = simulationRunPath(cat.runs[0], cat.root_dir);
              setSelectedRunKey(runPath);
              void loadLogFile(runPath, false);
            } else {
              void loadLogFile(p, false);
            }
          })
          .catch(() => {
            void loadLogFile(p, false);
          });
      });
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

  const lastEnrichedPathRef = useRef<string | null>(null);

  useEffect(() => {
    if (!activeLogPath || !entries.length) return;
    const guessed = guessGraphPreset(activeLogPath, entries);
    if (guessed) setGraphPreset(guessed);
  }, [activeLogPath, entries]);

  // Attempt auto-enrich if projectRoot is set and coordinates not yet loaded
  useEffect(() => {
    if (!projectRoot || !entries.length || !activeLogPath) return;
    if (lastEnrichedPathRef.current === activeLogPath) return;

    const day1 = entries.find((e) => e.day === 1);
    const hasAnyCoords = day1?.data.all_bin_coords?.some((b) => b.lat != null && b.lng != null);
    if (hasAnyCoords) {
      lastEnrichedPathRef.current = activeLogPath;
      return;
    }

    const presetId = guessGraphPreset(activeLogPath, entries) ?? graphPreset;
    loadGraphCoordinates(projectRoot, presetId, selectedSample ?? 0)
      .then((table) => {
        loadEntries(enrichEntriesWithGraphCoords(entries, table));
        setRouteMapMode("deckgl");
        lastEnrichedPathRef.current = activeLogPath;
      })
      .catch((err) => {
        console.warn("Auto-enrich graph coordinates skipped:", err);
      });
  }, [projectRoot, activeLogPath, entries, graphPreset, selectedSample, loadEntries]);

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

  const playbackDay = useSimulationCorpusStore((state) => state.playbackDay);
  const playbackPlaying = useSimulationCorpusStore((state) => state.playbackPlaying);
  const { view: corpusView } = useCorpusSceneState();
  const frame = useMemo(
    () => (corpusView == null ? null : corpusMapFrame(corpusView, playbackDay)),
    [corpusView, playbackDay]
  );
  const corpusOverride = useMemo(
    () => (playbackDay != null && frame != null && frame.routes.length > 0 ? frame.routes : null),
    [playbackDay, frame]
  );

  const { mapRoutes, mapHasFailureData } = useSimMonitorMapRoutes({
    entries,
    displayDay,
    selectedSample,
    activeMapPolicies,
    policies,
    pendingMapCompare,
    setPendingMapCompare,
    setShowRouteMap,
    setMapPolicies,
    setMapLayout,
    setRouteMapMode,
    corpusOverride,
  });

  const applied = applyCorpusMapFrame(mapRoutes, frame, playbackDay);

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

        <button onClick={openDirectory} className="btn-secondary flex items-center gap-2">
          <FolderTree size={14} />
          Open Directory
        </button>

        {catalog && catalog.runs.length > 1 && (
          <div className="flex items-center gap-1.5 bg-canvas-subtle px-2 py-1 rounded border border-border-default">
            <span className="text-xs text-canvas-muted whitespace-nowrap">Run:</span>
            <select
              className="select-base text-xs py-0.5 px-2 max-w-xs truncate"
              value={selectedRunKey ?? ""}
              onChange={(e) => void handleRunChange(e.target.value)}
              title={selectedRunKey ?? "Select run"}
            >
              <option value="__ALL__">All Runs ({catalog.runs.length})</option>
              {catalog.runs.map((r) => (
                <option key={r.key} value={r.key} title={r.label}>
                  {r.label} ({r.logs.length} log{r.logs.length === 1 ? "" : "s"})
                </option>
              ))}
            </select>
          </div>
        )}

        {catalog && catalog.runs.length === 1 && (
          <span
            className="text-xs bg-canvas-subtle text-canvas-muted px-2 py-1 rounded border border-border-default truncate max-w-xs"
            title={catalog.runs[0].key}
          >
            {catalog.runs[0].label} ({catalog.runs[0].logs.length} log{catalog.runs[0].logs.length === 1 ? "" : "s"})
          </span>
        )}

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
                {isWatching && <RefreshCw size={11} className="animate-spin text-accent-success shrink-0" />}
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

        <SimMonitorDayScrubber
          dayRange={dayRange}
          displayDay={displayDay}
          setSelectedDay={setSelectedDay}
          isPlaying={isPlaying}
          setIsPlaying={setIsPlaying}
          playbackSpeed={playbackSpeed}
          setPlaybackSpeed={setPlaybackSpeed}
          selectedDay={selectedDay}
          isWatching={isWatching}
        />
      </div>

      <SimMonitorChartPolicySelector
        policies={policies}
        activeChartPolicies={activeChartPolicies}
        toggleChartPolicy={toggleChartPolicy}
      />

      {!displayEntry && playbackDay == null && (
        <div className="flex flex-col items-center justify-center h-64 text-canvas-muted gap-3">
          <p className="text-sm">Open a simulation log file or directory to begin monitoring.</p>
          <p className="text-xs">
            Supports single log files (*.json, *.jsonl) or standard simulation directory hierarchies (&lt;days&gt;days/&lt;region&gt;&lt;bins&gt;_[waste]/[dist/]&lt;run&gt;/).
          </p>
        </div>
      )}

      {!displayEntry && playbackDay != null && <CorpusRouteMap />}

      {displayEntry && (
        <>
          <SimMonitorKpiSection
            displayEntry={displayEntry}
            displayDay={displayDay}
            filteredEntries={filteredEntries}
            showSecondary={showSecondary}
            setShowSecondary={setShowSecondary}
            policySeries={policySeries}
            logScale={logScale}
          />

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
            mapRoutes={applied.routes}
            mapReason={applied.reason}
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
            isPlaying={isPlaying || (playbackPlaying && applied.driven)}
            playbackSpeed={playbackSpeed}
            displayDay={displayDay}
            displayEntry={displayEntry}
          />

          <SimMonitorAdvancedPanels
            showBinFill={showBinFill}
            showTourTable={showTourTable}
            displayEntry={displayEntry}
            displayDay={displayDay}
            displayFailureEntry={displayFailureEntry}
            policyVizEntries={policyVizEntries}
            selectedPolicy={selectedPolicy}
            selectedSample={selectedSample}
            theme={theme}
            logScale={logScale}
            policyVizLive={policyVizLive}
            telemetryTrendsKey={telemetryTrendsKey}
            sourceRunLabel={sourceRunLabel}
            activeLogPath={activeLogPath}
            projectRoot={projectRoot}
            duckdbProfitRange={duckdbProfitRange}
            filteredEntries={filteredEntries}
            dayRange={dayRange}
            setSelectedDay={setSelectedDay}
            duckdbReady={duckdbReady}
            lastPipeline={lastPipeline}
            setDuckdbProfitRange={setDuckdbProfitRange}
          />
        </>
      )}
    </div>
  );
}
