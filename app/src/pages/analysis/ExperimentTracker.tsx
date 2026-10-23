/**
 * Experiment Tracker — MLflow run browser and metric comparison (§G.18).
 * Ports Streamlit `experiment_tracker` / `experiment_tracker_mlflow` modes.
 */
import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import ReactECharts from "echarts-for-react";
import { invoke } from "@tauri-apps/api/core";
import { open as openUrl } from "@tauri-apps/plugin-shell";
import { Download, ExternalLink, RefreshCw } from "lucide-react";
import { LoadedRunRow } from "../../components/common/LoadedRunRow";
import { OpenPathToolbar } from "../../components/common/OpenPathToolbar";
import { GlobalFilterBar } from "../../components/layout/GlobalFilterBar";
import { ProcessIdFooter } from "../../components/monitor/process/ProcessIdFooter";
import { TrainHpoLivePanel } from "../../components/monitor/live/TrainHpoLivePanel";
import { useProcessRunLabelBrush } from "../../hooks/brush/useProcessRunLabelBrush";
import { useAppStore } from "../../store/app";
import { useProcessStore } from "../../store/process";
import { collectAttentionVizFromLogLines } from "../../utils/graph/attentionViz";
import { collectTrainingHealthFromLogLines } from "../../utils/training/trainingHealth";
import {
  brushLogPathFromProcessLines,
  mlflowRunDirFromArtifactUri,
  outputRunPathFromLogLines,
  resolveLocalProjectPath,
} from "../../utils/runs/outputRunPath";
import { runLabelFromPath } from "../../utils/benchmark/policyTelemetryTrends";
import { trainingRunPathFromLogLines } from "../../utils/training/trainingRunPath";
import { collectTrainingMetricsFromLogLines } from "../../utils/training/trainingMetrics";
import { findRecentHpoProcessId, trainHpoLivePanelTitle } from "../../utils/training/trainingProcess";
import { useGlobalFiltersStore } from "../../store/filters";
import { MLIntrospectionPanel } from "../../components/analysis/training/MLIntrospectionPanel";
import {
  chartMetricDisplay,
  chartMetricYAxisType,
  isLogScaleMetric,
} from "../../utils/charts/chartLogScale";
import { ZenMLPipelineView } from "./ZenMLPipelineView";
import { ChartExportButtons } from "../../components/common/ChartExportButtons";
import { downloadCsv } from "../../utils/charts/tableExport";
import type { MlflowMetricPoint, MlflowRun, OutputDir } from "../../types";
import { simOutputRoots } from "../../utils/runs/outputRoots";

const DEFAULT_TRACKING_URI = "mlruns";
const DEFAULT_EXPERIMENT = "wsmart-route";
const DEFAULT_MLFLOW_UI = "http://localhost:5000";

type MlflowView = "runs" | "dashboard";
const RUN_COLORS = [
  "#667eea", "#f093fb", "#4fd1c5", "#f6ad55", "#fc8181",
  "#90cdf4", "#9ae6b4", "#fbd38d",
];

function formatBytes(b: number) {
  if (b < 1024) return `${b} B`;
  if (b < 1024 ** 2) return `${(b / 1024).toFixed(1)} KB`;
  return `${(b / 1024 ** 2).toFixed(1)} MB`;
}

function formatTime(ms: number | null) {
  if (ms == null) return "—";
  return new Date(ms).toLocaleString();
}

export function ExperimentTracker() {
  const { projectRoot, pythonPath, effectiveTheme } = useAppStore();
  const logScale = useGlobalFiltersStore((s) => s.logScale);
  const activeRunLabel = useGlobalFiltersStore((s) => s.runLabel);
  const processes = useProcessStore((s) => s.processes);
  const [trackingUri, setTrackingUri] = useState(DEFAULT_TRACKING_URI);
  const [experimentName, setExperimentName] = useState(DEFAULT_EXPERIMENT);
  const [runs, setRuns] = useState<MlflowRun[]>([]);
  const [selectedRunIds, setSelectedRunIds] = useState<string[]>([]);
  const [metricKeys, setMetricKeys] = useState<string[]>([]);
  const [selectedMetric, setSelectedMetric] = useState<string>("");
  const [metricHistory, setMetricHistory] = useState<Record<string, MlflowMetricPoint[]>>({});
  const [normalizeY, setNormalizeY] = useState(false);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [outputDirs, setOutputDirs] = useState<OutputDir[]>([]);
  const [mlflowView, setMlflowView] = useState<MlflowView>("runs");
  const [mlflowUiUrl, setMlflowUiUrl] = useState(DEFAULT_MLFLOW_UI);
  const chartRef = useRef<ReactECharts>(null);

  const recentHpoId = useMemo(() => findRecentHpoProcessId(processes), [processes]);
  const recentHpoProc = recentHpoId ? processes[recentHpoId] : null;
  const recentHpoRunning = recentHpoProc?.status === "running";
  const recentHpoDone = recentHpoProc != null && recentHpoProc.status !== "running";
  const processRunLabel = useProcessRunLabelBrush(
    recentHpoId,
    recentHpoProc?.logLines
  );

  const outputRunPath = useMemo(
    () => (recentHpoProc ? outputRunPathFromLogLines(recentHpoProc.logLines) : null),
    [recentHpoProc]
  );
  const trainingRunPath = useMemo(
    () => (recentHpoProc ? trainingRunPathFromLogLines(recentHpoProc.logLines) : null),
    [recentHpoProc]
  );
  const processLogPath = useMemo(
    () =>
      recentHpoProc
        ? brushLogPathFromProcessLines(recentHpoProc.logLines, "train")
        : null,
    [recentHpoProc]
  );

  const liveHealthEntries = useMemo(
    () =>
      recentHpoProc
        ? collectTrainingHealthFromLogLines(recentHpoProc.logLines)
        : [],
    [recentHpoProc]
  );

  const liveAttentionEntries = useMemo(
    () =>
      recentHpoProc
        ? collectAttentionVizFromLogLines(recentHpoProc.logLines)
        : [],
    [recentHpoProc]
  );

  const liveMetrics = useMemo(
    () =>
      recentHpoProc
        ? collectTrainingMetricsFromLogLines(recentHpoProc.logLines)
        : [],
    [recentHpoProc]
  );

  const mlflowRunBrushLabel = useCallback(
    (run: MlflowRun): string => {
      const runDir = mlflowRunDirFromArtifactUri(run.artifact_uri);
      const resolved = runDir
        ? resolveLocalProjectPath(runDir, projectRoot) ?? runDir
        : null;
      return run.run_name || (resolved ? runLabelFromPath(resolved) : run.run_id.slice(0, 8));
    },
    [projectRoot]
  );

  const trackingUriPath = useMemo(
    () => resolveLocalProjectPath(trackingUri, projectRoot),
    [trackingUri, projectRoot]
  );

  const filterRunLabels = useMemo(() => {
    if (processRunLabel) return [processRunLabel];
    const labels = selectedRunIds
      .map((id) => runs.find((r) => r.run_id === id))
      .filter((r): r is MlflowRun => r != null)
      .map(mlflowRunBrushLabel);
    return labels.length > 0 ? labels : [];
  }, [processRunLabel, selectedRunIds, runs, mlflowRunBrushLabel]);

  const refreshRuns = useCallback(async () => {
    if (!projectRoot) return;
    setLoading(true);
    setError(null);
    try {
      const found = await invoke<MlflowRun[]>("list_mlflow_runs", {
        trackingUri,
        experimentName: experimentName || null,
        projectRoot,
        pythonExecutable: pythonPath || null,
      });
      setRuns(found.sort((a, b) => (b.start_time ?? 0) - (a.start_time ?? 0)));
      setSelectedRunIds([]);
      setMetricKeys([]);
      setSelectedMetric("");
      setMetricHistory({});
    } catch (e) {
      setError(String(e));
      setRuns([]);
    } finally {
      setLoading(false);
    }
  }, [projectRoot, pythonPath, trackingUri, experimentName]);

  const refreshOutputDirs = useCallback(async () => {
    if (!projectRoot) return;
    try {
      const found = (
        await Promise.all(
          simOutputRoots(projectRoot).map((outputPath) =>
            invoke<OutputDir[]>("list_output_dirs", { outputPath })
          )
        )
      ).flat();
      setOutputDirs(found.sort((a, b) => b.created_at.localeCompare(a.created_at)));
    } catch {
      setOutputDirs([]);
    }
  }, [projectRoot]);

  useEffect(() => {
    if (projectRoot) {
      refreshRuns();
      refreshOutputDirs();
    }
  }, [projectRoot, refreshRuns, refreshOutputDirs]);

  const toggleRun = useCallback((runId: string) => {
    setSelectedRunIds((prev) =>
      prev.includes(runId) ? prev.filter((id) => id !== runId) : [...prev, runId]
    );
  }, []);

  // Load metric keys when first run is selected
  useEffect(() => {
    if (!projectRoot || selectedRunIds.length === 0) {
      setMetricKeys([]);
      setSelectedMetric("");
      return;
    }
    invoke<string[]>("list_mlflow_metric_keys", {
      runId: selectedRunIds[0],
      trackingUri,
      projectRoot,
      pythonExecutable: pythonPath || null,
    })
      .then((keys) => {
        setMetricKeys(keys);
        setSelectedMetric((prev) => (prev && keys.includes(prev) ? prev : keys[0] ?? ""));
      })
      .catch(() => setMetricKeys([]));
  }, [projectRoot, pythonPath, trackingUri, selectedRunIds]);

  // Load metric history for comparison chart
  useEffect(() => {
    if (!projectRoot || selectedRunIds.length === 0 || !selectedMetric) {
      setMetricHistory({});
      return;
    }
    invoke<Record<string, MlflowMetricPoint[]>>("load_mlflow_metric_history", {
      runIds: selectedRunIds,
      metricKey: selectedMetric,
      trackingUri,
      projectRoot,
      pythonExecutable: pythonPath || null,
    })
      .then(setMetricHistory)
      .catch(() => setMetricHistory({}));
  }, [projectRoot, pythonPath, trackingUri, selectedRunIds, selectedMetric]);

  const useMetricLogScale = logScale && !normalizeY && isLogScaleMetric(selectedMetric);

  const comparisonOption = useMemo(() => {
    const series = selectedRunIds.map((runId, i) => {
      const points = metricHistory[runId] ?? [];
      const values = points.map((p) => p.value);
      if (normalizeY && values.length > 0) {
        const min = Math.min(...values);
        const max = Math.max(...values);
        const range = max - min || 1;
        return {
          name: runId.slice(0, 8),
          type: "line" as const,
          smooth: true,
          symbolSize: 4,
          lineStyle: { color: RUN_COLORS[i % RUN_COLORS.length], width: 2 },
          itemStyle: { color: RUN_COLORS[i % RUN_COLORS.length] },
          data: points.map((p) => [p.step, (p.value - min) / range]),
        };
      }
      return {
        name: runId.slice(0, 8),
        type: "line" as const,
        smooth: true,
        symbolSize: 4,
        lineStyle: { color: RUN_COLORS[i % RUN_COLORS.length], width: 2 },
        itemStyle: { color: RUN_COLORS[i % RUN_COLORS.length] },
        data: points.map((p) => [
          p.step,
          chartMetricDisplay(p.value, selectedMetric, logScale && !normalizeY),
        ]),
      };
    });

    const yName = normalizeY
      ? "Normalized"
      : useMetricLogScale
        ? `${selectedMetric} (log)`
        : selectedMetric;

    return {
      backgroundColor: "transparent",
      grid: { left: 50, right: 10, top: 30, bottom: 40 },
      xAxis: {
        type: "value",
        name: "Step",
        axisLabel: { color: "#9090b0", fontSize: 10 },
      },
      yAxis: {
        type: useMetricLogScale
          ? chartMetricYAxisType(selectedMetric, true)
          : "value",
        logBase: 10,
        name: yName,
        axisLabel: { color: "#9090b0", fontSize: 10 },
        minorSplitLine: { show: false },
      },
      legend: { textStyle: { color: "#9090b0", fontSize: 10 }, top: 0 },
      series,
      tooltip: { trigger: "axis" },
    };
  }, [
    selectedRunIds,
    metricHistory,
    selectedMetric,
    normalizeY,
    logScale,
    useMetricLogScale,
  ]);

  if (!projectRoot) {
    return (
      <div className="flex items-center justify-center h-48 text-canvas-muted text-sm">
        Set project root to browse experiments.
      </div>
    );
  }

  return (
    <div className="space-y-4">
      <GlobalFilterBar showLogScale runLabels={filterRunLabels} />

      {recentHpoId && recentHpoProc && (
        <TrainHpoLivePanel
          cardClassName="border-accent-success/30"
          header={{
            status: recentHpoRunning ? "running" : recentHpoProc.status,
            title: trainHpoLivePanelTitle({
              isRunning: recentHpoRunning,
              status: recentHpoProc.status,
              processId: recentHpoId,
              command: recentHpoProc.command,
              kind: "hpo",
            }),
            runLabel: processRunLabel,
            logPath: processLogPath,
            projectRoot,
            showLiveSuffix: recentHpoRunning,
            metricCount: liveMetrics.length,
            healthCount: liveHealthEntries.length,
            attentionCount: liveAttentionEntries.length,
            navMesh: {
              showHpoLinks: true,
              showOutputBrowser: recentHpoDone && recentHpoProc.status === "completed",
              outputRunPath,
              trainingRunPath,
            },
          }}
          progress={
            recentHpoRunning
              ? {
                  processId: recentHpoId,
                }
              : undefined
          }
          analytics={{
            metrics: liveMetrics,
            healthEntries: liveHealthEntries,
            attentionEntries: liveAttentionEntries,
            logScale,
            theme: effectiveTheme,
            exportNamePrefix: "experiment-tracker",
            isPostRun: !recentHpoRunning,
            postRunFallback:
              "Post-run shortcuts — open Output Browser or Training Monitor for this sweep",
          }}
          logLines={recentHpoProc.logLines}
          logTailWaiting={recentHpoRunning}
          footer={
            <ProcessIdFooter
              processId={recentHpoId}
              logPath={processLogPath}
              runLabel={processRunLabel}
              projectRoot={projectRoot}
            />
          }
        />
      )}

      {/* MLflow connection */}
      <div className="card space-y-3">
        <div className="flex items-center justify-between flex-wrap gap-2">
          <h2 className="text-sm font-semibold text-gray-200">MLflow Tracking</h2>
          <div className="flex items-center gap-1 bg-canvas-elevated rounded-lg p-0.5">
            {(["runs", "dashboard"] as const).map((v) => (
              <button
                key={v}
                onClick={() => setMlflowView(v)}
                className={`text-xs px-2.5 py-1 rounded-md transition-colors ${
                  mlflowView === v
                    ? "bg-accent-primary text-white"
                    : "text-canvas-muted hover:text-gray-200"
                }`}
              >
                {v === "runs" ? "Runs" : "Dashboard"}
              </button>
            ))}
          </div>
        </div>

        {mlflowView === "dashboard" && (
          <div className="space-y-2">
            <div className="flex flex-wrap gap-2 items-center">
              <input
                className="input-base font-mono text-xs flex-1 min-w-[200px]"
                value={mlflowUiUrl}
                onChange={(e) => setMlflowUiUrl(e.target.value)}
                placeholder="http://localhost:5000"
              />
              <button
                onClick={() => openUrl(mlflowUiUrl).catch(() => {})}
                className="btn-ghost text-xs flex items-center gap-1"
              >
                <ExternalLink size={12} />
                Open in browser
              </button>
            </div>
            <iframe
              title="MLflow Dashboard"
              src={mlflowUiUrl}
              className="w-full h-[480px] rounded-lg border border-canvas-border bg-white"
              sandbox="allow-scripts allow-same-origin allow-forms"
            />
            <p className="text-[10px] text-canvas-muted">
              Embedded MLflow UI fallback (§G.18). Start the MLflow server locally to use this view.
            </p>
          </div>
        )}

        {mlflowView === "runs" && (
        <>
        <div className="flex flex-wrap gap-3 items-end">
          <div className="flex flex-col gap-1">
            <label className="text-xs text-canvas-muted">Tracking URI</label>
            <input
              className="input-base font-mono text-xs w-48"
              value={trackingUri}
              onChange={(e) => setTrackingUri(e.target.value)}
              placeholder="mlruns"
            />
            {trackingUriPath ? (
              <OpenPathToolbar
                path={trackingUriPath}
                chipClassName="max-w-full"
                handoff={false}
              />
            ) : null}
          </div>
          <div className="flex flex-col gap-1">
            <label className="text-xs text-canvas-muted">Experiment</label>
            <input
              className="input-base font-mono text-xs w-40"
              value={experimentName}
              onChange={(e) => setExperimentName(e.target.value)}
              placeholder="wsmart-route"
            />
          </div>
          <button
            onClick={refreshRuns}
            disabled={loading}
            className="btn-primary flex items-center gap-2"
          >
            <RefreshCw size={14} className={loading ? "animate-spin" : ""} />
            Refresh
          </button>
          <span className="text-xs text-canvas-muted">{runs.length} run(s)</span>
          {runs.length > 0 && (
            <button
              onClick={() =>
                downloadCsv(
                  "mlflow-runs.csv",
                  ["run_id", "run_name", "status", "start_time"],
                  runs.map((r) => [
                    r.run_id,
                    r.run_name,
                    r.status,
                    r.start_time != null ? new Date(r.start_time).toISOString() : "",
                  ])
                )
              }
              className="btn-ghost text-xs flex items-center gap-1"
            >
              <Download size={12} />
              Export CSV
            </button>
          )}
        </div>
        {error && (
          <p className="text-xs text-accent-danger">{error}</p>
        )}

      {/* MLflow run table */}
      <div className="card overflow-auto">
        <table className="w-full text-xs">
          <thead>
            <tr className="border-b border-canvas-border">
              <th className="py-2 px-2 text-canvas-muted font-medium w-8" />
              <th className="text-left py-2 px-2 text-canvas-muted font-medium">Run</th>
              <th className="text-left py-2 px-2 text-canvas-muted font-medium">Status</th>
              <th className="text-left py-2 px-2 text-canvas-muted font-medium">Started</th>
              <th className="text-left py-2 px-2 text-canvas-muted font-medium">Metrics</th>
            </tr>
          </thead>
          <tbody className="divide-y divide-canvas-border">
            {runs.map((r) => {
              const rawRunDir = mlflowRunDirFromArtifactUri(r.artifact_uri);
              const runDir = rawRunDir
                ? resolveLocalProjectPath(rawRunDir, projectRoot) ?? rawRunDir
                : null;
              const brushLabel = mlflowRunBrushLabel(r);
              const brushActive = Boolean(activeRunLabel && activeRunLabel === brushLabel);
              return (
              <tr
                key={r.run_id}
                className={`hover:bg-canvas-hover cursor-pointer ${
                  selectedRunIds.includes(r.run_id) || brushActive ? "bg-canvas-hover/60" : ""
                } ${brushActive ? "ring-1 ring-inset ring-accent-secondary/30" : ""}`}
                onClick={() => toggleRun(r.run_id)}
              >
                <td className="py-2 px-2">
                  <input
                    type="checkbox"
                    className="accent-accent-primary"
                    checked={selectedRunIds.includes(r.run_id)}
                    onChange={() => toggleRun(r.run_id)}
                    onClick={(e) => e.stopPropagation()}
                  />
                </td>
                <td className="py-2 px-2">
                  {runDir ? (
                    <OpenPathToolbar
                      path={runDir}
                      kind="training"
                      label={brushLabel}
                      brushLabel={brushLabel}
                      storedLabel={brushLabel}
                      chipClassName="flex-1 min-w-0 max-w-none"
                      className="flex-1 min-w-0"
                    >
                      <span className="text-canvas-muted font-mono text-[10px] shrink-0 truncate max-w-[6rem]">
                        {r.run_id.slice(0, 12)}…
                      </span>
                    </OpenPathToolbar>
                  ) : (
                    <>
                      <div className="font-mono text-gray-300">{brushLabel}</div>
                      <div className="text-canvas-muted font-mono text-[10px]">
                        {r.run_id.slice(0, 12)}…
                      </div>
                    </>
                  )}
                </td>
                <td className="py-2 px-2 text-canvas-muted">{r.status}</td>
                <td className="py-2 px-2 text-canvas-muted">{formatTime(r.start_time)}</td>
                <td className="py-2 px-2 text-canvas-muted font-mono">
                  {Object.entries(r.metrics)
                    .slice(0, 3)
                    .map(([k, v]) => `${k}=${v.toFixed(3)}`)
                    .join(", ") || "—"}
                </td>
              </tr>
            );
            })}
            {runs.length === 0 && !loading && (
              <tr>
                <td colSpan={5} className="py-8 text-center text-canvas-muted">
                  No MLflow runs found. Enable MLflow tracking and run training to populate this view.
                </td>
              </tr>
            )}
          </tbody>
        </table>
      </div>

      {/* Metric comparison chart */}
      {selectedRunIds.length > 0 && metricKeys.length > 0 && (
        <div className="card space-y-3">
          <div className="flex flex-wrap items-center gap-3">
            <h3 className="text-sm font-semibold text-gray-200">
              Metric Comparison ({selectedRunIds.length} run{selectedRunIds.length > 1 ? "s" : ""})
            </h3>
            <select
              className="select-base text-xs w-48"
              value={selectedMetric}
              onChange={(e) => setSelectedMetric(e.target.value)}
            >
              {metricKeys.map((k) => (
                <option key={k} value={k}>{k}</option>
              ))}
            </select>
            <label className="flex items-center gap-1.5 text-xs text-canvas-muted cursor-pointer">
              <input
                type="checkbox"
                className="accent-accent-primary"
                checked={normalizeY}
                onChange={(e) => setNormalizeY(e.target.checked)}
              />
              Normalize Y-axis
            </label>
            <ChartExportButtons
              chartRef={chartRef}
              filenameStem={`mlflow-${selectedMetric}`}
              className="ml-auto"
            />
          </div>
          <p className="text-[10px] text-canvas-muted">
            {useMetricLogScale
              ? "Log-scale y-axis (disabled when Normalize is on)"
              : "Linear y-axis"}
          </p>
          <ReactECharts ref={chartRef} option={comparisonOption} style={{ height: 320 }} />
        </div>
      )}

      {/* Selected run params */}
      {selectedRunIds.length === 1 && (
        <div className="card space-y-2">
          <h3 className="text-sm font-semibold text-gray-200">Parameters</h3>
          <div className="grid grid-cols-2 sm:grid-cols-3 gap-2 text-xs">
            {Object.entries(runs.find((r) => r.run_id === selectedRunIds[0])?.params ?? {}).map(
              ([k, v]) => (
                <div key={k} className="flex gap-2">
                  <span className="text-canvas-muted shrink-0">{k}</span>
                  <span className="font-mono text-gray-300 truncate">{v}</span>
                </div>
              )
            )}
          </div>
        </div>
      )}
        </>
        )}
      </div>

      {/* ML introspection — TensorDict, attention, loss landscape (§G.5) */}
      <MLIntrospectionPanel logScale={logScale} />

      {/* ZenML pipeline runs (§G.18) */}
      <ZenMLPipelineView logScale={logScale} />

      {/* Legacy output dirs */}
      <div className="card space-y-2">
        <h3 className="text-sm font-semibold text-gray-200">Output Directories</h3>
        <table className="w-full text-xs">
          <thead>
            <tr className="border-b border-canvas-border">
              <th className="text-left py-2 px-3 text-canvas-muted font-medium">Run Name</th>
              <th className="text-left py-2 px-3 text-canvas-muted font-medium">Created</th>
              <th className="text-right py-2 px-3 text-canvas-muted font-medium">Size</th>
            </tr>
          </thead>
          <tbody className="divide-y divide-canvas-border">
            {outputDirs.slice(0, 10).map((d) => (
              <tr key={d.path} className="hover:bg-canvas-hover">
                <td className="py-2 px-3">
                  <LoadedRunRow
                    path={d.path}
                    projectRoot={projectRoot}
                    label={d.name}
                    activeRunLabel={activeRunLabel}
                    pathHandoffs
                    handoffKind="run"
                  />
                </td>
                <td className="py-2 px-3 text-canvas-muted">
                  {new Date(d.created_at).toLocaleString()}
                </td>
                <td className="py-2 px-3 text-right text-canvas-muted">{formatBytes(d.size_bytes)}</td>
              </tr>
            ))}
            {outputDirs.length === 0 && (
              <tr>
                <td colSpan={3} className="py-4 text-center text-canvas-muted">
                  No output directories found.
                </td>
              </tr>
            )}
          </tbody>
        </table>
      </div>
    </div>
  );
}
