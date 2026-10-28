/**
 * Simulation Digital Twin — real-time visualization of a running simulation (§G.16, Slice V5).
 *
 * Uses `useSimThrottle` (2 Hz visibility-aware throttle), `StalenessDot`,
 * `useLoadedRunStore` synchronization, and optional hexagonal spatial density overlay.
 */
import { useEffect, useMemo, useState } from "react";
import { CorpusRouteMap } from "../../components/analysis/summary/corpus/CorpusRouteMap";
import { GlobalFilterBar } from "../../components/layout/GlobalFilterBar";
import { SimMonitorAdvancedPanels, SimMonitorKpiSection } from "../../components/monitor/sim/SimMonitorDetailPanels";
import { SimMonitorChartPolicySelector } from "../../components/monitor/sim/SimMonitorHeaderControls";
import { SimMonitorMapSection } from "../../components/monitor/sim/SimMonitorMapSection";
import { POLICY_COLORS } from "../../components/monitor/sim/SimMonitorPanels";
import { useSimMonitorMapRoutes } from "../../components/monitor/sim/useSimMonitorMapRoutes";
import { useSimWatcher } from "../../hooks/process/useSimWatcher";
import { useSimThrottle } from "../../hooks/sim/useSimThrottle";
import { useAppStore } from "../../store/app";
import { useGlobalFiltersStore } from "../../store/filters";
import { useProcessStore } from "../../store/process";
import { useSimStore, uniquePolicies, uniqueSamples, filterEntries } from "../../store/sim";
import { useCorpusSceneState, useSimulationCorpusStore } from "../../store/simulationCorpus";
import { applyCorpusMapFrame } from "../../utils/corpus/monitorRoutes";
import { corpusMapFrame } from "../../utils/corpus/routeFrame";
import { filterFailureEntries } from "../../utils/sim/simFailure";
import { useGraphCoordinates } from "./simulation/useGraphCoordinates";
import { usePolicyVisibility } from "./simulation/usePolicyVisibility";
import { SimulationMonitorToolbar } from "./simulation/SimulationMonitorToolbar";
import { useSimulationLogSession } from "./simulation/useSimulationLogSession";

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

  const {
    activeLogPath,
    sourceRunLabel,
    catalog,
    selectedRunKey,
    openLog,
    openDirectory,
    handleRunChange,
    projectRoot,
    duckdbReady,
    duckdbLoading,
    lastPipeline,
  } = useSimulationLogSession({
    reset,
    loadEntries,
    loadPolicyVizEntries,
    loadFailureEntries,
    setWatchPath,
    pendingLogPath,
    setPendingLogPath,
  });

  const [showSecondary, setShowSecondary] = useState(false);
  const [showBinFill, setShowBinFill] = useState(true);
  const [showTourTable, setShowTourTable] = useState(false);
  const [showRouteMap, setShowRouteMap] = useState(true);
  const [isPlaying, setIsPlaying] = useState(false);
  const [playbackSpeed, setPlaybackSpeed] = useState<1 | 2 | 4>(1);
  const [mapLayout, setMapLayout] = useState<"overlay" | "split">("overlay");
  const [showFailureOverlay, setShowFailureOverlay] = useState(true);
  const [showRouteDiff, setShowRouteDiff] = useState(false);
  const [showHexDensity, setShowHexDensity] = useState(false);
  const [routeMapMode, setRouteMapMode] = useState<"echarts" | "deckgl">("echarts");
  const [duckdbProfitRange, setDuckdbProfitRange] = useState<[number, number] | null>(null);

  const {
    graphPreset,
    setGraphPreset,
    loadingGraphCoords,
    applyGraphCoords,
    hasGeoCoords,
    hasBinCoords,
  } = useGraphCoordinates({
    activeLogPath,
    entries,
    projectRoot,
    selectedSample,
    loadEntries,
    displayEntry,
    setRouteMapMode,
  });

  const {
    activeChartPolicies,
    toggleChartPolicy,
    activeMapPolicies,
    toggleMapPolicy,
    setMapPolicies,
  } = usePolicyVisibility(policies);

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
      <SimulationMonitorToolbar
        openLog={openLog}
        openDirectory={openDirectory}
        catalog={catalog}
        selectedRunKey={selectedRunKey}
        handleRunChange={handleRunChange}
        isWatching={isWatching}
        lastEventAt={lastEventAt}
        watchPath={watchPath}
        projectRoot={projectRoot}
        duckdbLoading={duckdbLoading}
        lastPipeline={lastPipeline}
        policies={policies}
        selectedPolicy={selectedPolicy}
        setPolicy={setPolicy}
        samples={samples}
        selectedSample={selectedSample}
        setSampleId={setSampleId}
        dayRange={dayRange}
        displayDay={displayDay}
        setSelectedDay={setSelectedDay}
        isPlaying={isPlaying}
        setIsPlaying={setIsPlaying}
        playbackSpeed={playbackSpeed}
        setPlaybackSpeed={setPlaybackSpeed}
        selectedDay={selectedDay}
      />

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
