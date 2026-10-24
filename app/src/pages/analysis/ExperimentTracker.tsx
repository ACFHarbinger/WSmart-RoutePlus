/**
 * Experiment Tracker — MLflow run browser and metric comparison (§G.18).
 * Ports Streamlit `experiment_tracker` / `experiment_tracker_mlflow` modes.
 */
import { useCallback, useEffect, useMemo, useState } from "react";
import { invoke } from "@tauri-apps/api/core";
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
import {
  findRecentHpoProcessId,
  trainHpoLivePanelTitle,
} from "../../utils/training/trainingProcess";
import { useGlobalFiltersStore } from "../../store/filters";
import { MLIntrospectionPanel } from "../../components/analysis/training/MLIntrospectionPanel";
import {
  chartMetricDisplay,
  chartMetricYAxisType,
  isLogScaleMetric,
} from "../../utils/charts/chartLogScale";
import { ZenMLPipelineView } from "./ZenMLPipelineView";
import {
  ExperimentMlflowSection,
  ExperimentOutputDirsSection,
  type MlflowView,
} from "../../components/analysis/experiment/ExperimentMlflowSection";
import type { MlflowMetricPoint, MlflowRun, OutputDir } from "../../types";
import { simOutputRoots } from "../../utils/runs/outputRoots";

const DEFAULT_TRACKING_URI = "mlruns";
const DEFAULT_EXPERIMENT = "wsmart-route";
const DEFAULT_MLFLOW_UI = "http://localhost:5000";

const RUN_COLORS = [
  "#667eea",
  "#f093fb",
  "#4fd1c5",
  "#f6ad55",
  "#fc8181",
  "#90cdf4",
  "#9ae6b4",
  "#fbd38d",
];

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
  const [metricHistory, setMetricHistory] = useState<
    Record<string, MlflowMetricPoint[]>
  >({});
  const [normalizeY, setNormalizeY] = useState(false);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [outputDirs, setOutputDirs] = useState<OutputDir[]>([]);
  const [mlflowView, setMlflowView] = useState<MlflowView>("runs");
  const [mlflowUiUrl, setMlflowUiUrl] = useState(DEFAULT_MLFLOW_UI);

  const recentHpoId = useMemo(
    () => findRecentHpoProcessId(processes),
    [processes]
  );
  const recentHpoProc = recentHpoId ? processes[recentHpoId] : null;
  const recentHpoRunning = recentHpoProc?.status === "running";
  const recentHpoDone =
    recentHpoProc != null && recentHpoProc.status !== "running";
  const processRunLabel = useProcessRunLabelBrush(
    recentHpoId,
    recentHpoProc?.logLines
  );

  const outputRunPath = useMemo(
    () =>
      recentHpoProc ? outputRunPathFromLogLines(recentHpoProc.logLines) : null,
    [recentHpoProc]
  );
  const trainingRunPath = useMemo(
    () =>
      recentHpoProc
        ? trainingRunPathFromLogLines(recentHpoProc.logLines)
        : null,
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
      return (
        run.run_name ||
        (resolved ? runLabelFromPath(resolved) : run.run_id.slice(0, 8))
      );
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
      setOutputDirs(
        found.sort((a, b) => b.created_at.localeCompare(a.created_at))
      );
    } catch {
      setOutputDirs([]);
    }
  }, [projectRoot]);

  useEffect(() => {
    if (projectRoot) {
      void refreshRuns();
      void refreshOutputDirs();
    }
  }, [projectRoot, refreshRuns, refreshOutputDirs]);

  const toggleRun = useCallback((runId: string) => {
    setSelectedRunIds((prev) =>
      prev.includes(runId)
        ? prev.filter((id) => id !== runId)
        : [...prev, runId]
    );
  }, []);

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
        setSelectedMetric((prev) =>
          prev && keys.includes(prev) ? prev : keys[0] ?? ""
        );
      })
      .catch(() => setMetricKeys([]));
  }, [projectRoot, pythonPath, trackingUri, selectedRunIds]);

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

  const useMetricLogScale =
    logScale && !normalizeY && isLogScaleMetric(selectedMetric);

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
              showOutputBrowser:
                recentHpoDone && recentHpoProc.status === "completed",
              outputRunPath,
              trainingRunPath,
            },
          }}
          progress={
            recentHpoRunning ? { processId: recentHpoId } : undefined
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

      <ExperimentMlflowSection
        projectRoot={projectRoot}
        mlflowView={mlflowView}
        setMlflowView={setMlflowView}
        mlflowUiUrl={mlflowUiUrl}
        setMlflowUiUrl={setMlflowUiUrl}
        trackingUri={trackingUri}
        setTrackingUri={setTrackingUri}
        trackingUriPath={trackingUriPath}
        experimentName={experimentName}
        setExperimentName={setExperimentName}
        refreshRuns={refreshRuns}
        loading={loading}
        runs={runs}
        error={error}
        selectedRunIds={selectedRunIds}
        toggleRun={toggleRun}
        mlflowRunBrushLabel={mlflowRunBrushLabel}
        activeRunLabel={activeRunLabel}
        metricKeys={metricKeys}
        selectedMetric={selectedMetric}
        setSelectedMetric={setSelectedMetric}
        normalizeY={normalizeY}
        setNormalizeY={setNormalizeY}
        useMetricLogScale={useMetricLogScale}
        comparisonOption={comparisonOption}
      />

      <MLIntrospectionPanel logScale={logScale} />

      <ZenMLPipelineView logScale={logScale} />

      <ExperimentOutputDirsSection
        outputDirs={outputDirs}
        projectRoot={projectRoot}
        activeRunLabel={activeRunLabel}
      />
    </div>
  );
}
