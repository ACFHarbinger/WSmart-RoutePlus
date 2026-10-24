/**
 * Training Monitor — live and historical training metrics (§G.17, Slice V5).
 *
 * Subscribes to both `"process:stdout_batch"` (batched 50ms Rust emitter)
 * and `"process:stdout"` for backward compatibility.
 */
import { useCallback, useEffect, useMemo, useState } from "react";
import { invoke } from "@tauri-apps/api/core";
import { listen } from "@tauri-apps/api/event";
import { FolderOpen, RefreshCw } from "lucide-react";
import { LoadedRunRow } from "../../components/common/LoadedRunRow";
import { OpenPathToolbar } from "../../components/common/OpenPathToolbar";
import { GlobalFilterBar } from "../../components/layout/GlobalFilterBar";
import { ProcessIdFooter } from "../../components/monitor/process/ProcessIdFooter";
import { TrainHpoLivePanel } from "../../components/monitor/live/TrainHpoLivePanel";
import { RuntimeAttentionPanel } from "../../components/analysis/topology/RuntimeAttentionPanel";
import { TrainingHealthPanel } from "../../components/analysis/training/TrainingHealthPanel";
import {
  MultiRunChart,
  RunPanel,
  RUN_COLORS,
} from "../../components/monitor/training/TrainingMonitorPanels";
import { useProcessRunLabelBrush } from "../../hooks/brush/useProcessRunLabelBrush";
import { useAppStore } from "../../store/app";
import { useGlobalFiltersStore } from "../../store/filters";
import { useProcessStore } from "../../store/process";
import {
  collectAttentionVizFromLogLines,
  parseAttentionVizLine,
} from "../../utils/graph/attentionViz";
import {
  collectTrainingHealthFromLogLines,
  parseTrainingHealthLine,
} from "../../utils/training/trainingHealth";
import {
  collectTrainingMetricsFromLogLines,
  normalizeTrainingMetricRow,
  parseTrainingMetricLine,
} from "../../utils/training/trainingMetrics";
import { portfolioRunLabel } from "../../utils/duckdb/arrowPipeline";
import {
  brushLogPathFromProcessLines,
  outputRunPathFromLogLines,
} from "../../utils/runs/outputRunPath";
import { useRecentHandoff } from "../../hooks/files/useRecentHandoff";
import { trainingRunPathFromLogLines } from "../../utils/training/trainingRunPath";
import {
  findActiveLiveTrainProcessId,
  findRecentTrainOrHpoProcessId,
  isHpoProcess,
  trainHpoLivePanelTitle,
} from "../../utils/training/trainingProcess";
import type {
  AttentionVizEntry,
  StdoutBatch,
  StdoutLine,
  TrainingHealthEntry,
  TrainingRun,
  TrainingMetricsRow,
} from "../../types";

const LIVE_KEY = "__live__";
const LIVE_HEALTH_KEY = "__live_health__";
const LIVE_ATTENTION_KEY = "__live_attention__";

export function TrainingMonitor() {
  const {
    effectiveTheme,
    pendingTrainingRunPath,
    setPendingTrainingRunPath,
  } = useAppStore();
  const { projectRoot, handoff } = useRecentHandoff();
  const { logScale, runLabel: activeRunLabel } = useGlobalFiltersStore();
  const [runs, setRuns] = useState<TrainingRun[]>([]);
  const [selected, setSelected] = useState<string[]>([]);
  const [metricsMap, setMetricsMap] = useState<Record<string, TrainingMetricsRow[]>>({});
  const [healthMap, setHealthMap] = useState<Record<string, TrainingHealthEntry[]>>({});
  const [attentionMap, setAttentionMap] = useState<Record<string, AttentionVizEntry[]>>({});
  const [loading, setLoading] = useState(false);

  const logsPath = projectRoot ? `${projectRoot}/logs` : "";

  const processes = useProcessStore((s) => s.processes);
  const activeTrainId = useMemo(
    () => findActiveLiveTrainProcessId(processes),
    [processes]
  );
  const activeIsHpo = activeTrainId
    ? isHpoProcess(activeTrainId, processes[activeTrainId]?.command ?? "")
    : false;
  const activeTrainRunning =
    activeTrainId != null && processes[activeTrainId]?.status === "running";

  const recentTrainId = useMemo(
    () => findRecentTrainOrHpoProcessId(processes),
    [processes]
  );
  const recentTrainProc = recentTrainId ? processes[recentTrainId] : null;
  const recentTrainDone =
    recentTrainProc != null && recentTrainProc.status !== "running";
  const recentTrainCompleted = recentTrainProc?.status === "completed";
  const recentOutputRunPath = useMemo(
    () =>
      recentTrainProc ? outputRunPathFromLogLines(recentTrainProc.logLines) : null,
    [recentTrainProc]
  );
  const recentTrainingRunPath = useMemo(
    () =>
      recentTrainProc ? trainingRunPathFromLogLines(recentTrainProc.logLines) : null,
    [recentTrainProc]
  );
  const recentTrainLogLines = recentTrainProc?.logLines ?? [];
  const processRunLabel = useProcessRunLabelBrush(recentTrainId, recentTrainLogLines);
  const processLogPath = useMemo(
    () => brushLogPathFromProcessLines(recentTrainLogLines, "train"),
    [recentTrainLogLines]
  );
  const liveRunLabel = useMemo(() => {
    if (activeTrainId) {
      return trainHpoLivePanelTitle({
        isRunning: activeTrainRunning,
        status: processes[activeTrainId]?.status,
        processId: activeTrainId,
        command: processes[activeTrainId]?.command,
      });
    }
    if (recentTrainId && recentTrainProc) {
      return trainHpoLivePanelTitle({
        isRunning: false,
        status: recentTrainProc.status,
        processId: recentTrainId,
        command: recentTrainProc.command,
      });
    }
    return "Live Training";
  }, [
    activeTrainId,
    activeTrainRunning,
    processes,
    recentTrainId,
    recentTrainProc,
  ]);

  const effectiveLiveMetrics = useMemo(() => {
    if (activeTrainId) return metricsMap[LIVE_KEY] ?? [];
    if (recentTrainProc) {
      return collectTrainingMetricsFromLogLines(recentTrainLogLines);
    }
    return [];
  }, [activeTrainId, metricsMap, recentTrainProc, recentTrainLogLines]);

  const effectiveLiveHealth = useMemo(() => {
    if (activeTrainId) return healthMap[LIVE_HEALTH_KEY] ?? [];
    if (recentTrainProc) {
      return collectTrainingHealthFromLogLines(recentTrainLogLines);
    }
    return [];
  }, [activeTrainId, healthMap, recentTrainProc, recentTrainLogLines]);

  const effectiveLiveAttention = useMemo(() => {
    if (activeTrainId) return attentionMap[LIVE_ATTENTION_KEY] ?? [];
    if (recentTrainProc) {
      return collectAttentionVizFromLogLines(recentTrainLogLines);
    }
    return [];
  }, [activeTrainId, attentionMap, recentTrainProc, recentTrainLogLines]);

  const latestLiveMetric = effectiveLiveMetrics[effectiveLiveMetrics.length - 1];

  useEffect(() => {
    if (!activeTrainId) {
      setMetricsMap((m) => {
        if (!m[LIVE_KEY]) return m;
        const next = { ...m };
        delete next[LIVE_KEY];
        return next;
      });
      setHealthMap((h) => {
        if (!h[LIVE_HEALTH_KEY]) return h;
        const next = { ...h };
        delete next[LIVE_HEALTH_KEY];
        return next;
      });
      setAttentionMap((a) => {
        if (!a[LIVE_ATTENTION_KEY]) return a;
        const next = { ...a };
        delete next[LIVE_ATTENTION_KEY];
        return next;
      });
      return;
    }
    setMetricsMap((m) => ({ ...m, [LIVE_KEY]: [] }));
    setHealthMap((h) => ({ ...h, [LIVE_HEALTH_KEY]: [] }));
    setAttentionMap((a) => ({ ...a, [LIVE_ATTENTION_KEY]: [] }));

    const processLines = (lines: string[]) => {
      const newRows: TrainingMetricsRow[] = [];
      const newAlerts: TrainingHealthEntry[] = [];
      const newAttentions: AttentionVizEntry[] = [];
      for (const line of lines) {
        const row = parseTrainingMetricLine(line);
        if (row) newRows.push(row);
        const alert = parseTrainingHealthLine(line);
        if (alert) newAlerts.push(alert);
        const attention = parseAttentionVizLine(line);
        if (attention) newAttentions.push(attention);
      }
      if (newRows.length > 0) {
        setMetricsMap((m) => ({
          ...m,
          [LIVE_KEY]: [...(m[LIVE_KEY] ?? []), ...newRows],
        }));
      }
      if (newAlerts.length > 0) {
        setHealthMap((h) => ({
          ...h,
          [LIVE_HEALTH_KEY]: [...(h[LIVE_HEALTH_KEY] ?? []), ...newAlerts],
        }));
      }
      if (newAttentions.length > 0) {
        setAttentionMap((a) => ({
          ...a,
          [LIVE_ATTENTION_KEY]: [...(a[LIVE_ATTENTION_KEY] ?? []), ...newAttentions],
        }));
      }
    };

    let unlistenBatch: (() => void) | null = null;
    let unlistenSingle: (() => void) | null = null;
    listen<StdoutBatch>("process:stdout_batch", (event) => {
      if (event.payload.id === activeTrainId) {
        processLines(event.payload.lines);
      }
    }).then((fn) => {
      unlistenBatch = fn;
    });
    listen<StdoutLine>("process:stdout", (event) => {
      if (event.payload.id === activeTrainId) {
        processLines([event.payload.line]);
      }
    }).then((fn) => {
      unlistenSingle = fn;
    });

    return () => {
      unlistenBatch?.();
      unlistenSingle?.();
    };
  }, [activeTrainId]);

  useEffect(() => {
    if (activeTrainId && !selected.includes(LIVE_KEY)) {
      setSelected((s) => [LIVE_KEY, ...s]);
    } else if (
      !activeTrainId &&
      recentTrainId &&
      effectiveLiveMetrics.length > 0 &&
      !selected.includes(LIVE_KEY)
    ) {
      setSelected((s) => [LIVE_KEY, ...s]);
    }
  }, [activeTrainId, recentTrainId, effectiveLiveMetrics.length]); // eslint-disable-line react-hooks/exhaustive-deps

  const discover = useCallback(async () => {
    if (!logsPath) return;
    setLoading(true);
    try {
      const found = await invoke<TrainingRun[]>("list_training_runs", { logsPath });
      setRuns(found);
    } finally {
      setLoading(false);
    }
  }, [logsPath]);

  useEffect(() => {
    if (logsPath) discover();
  }, [logsPath, discover]);

  const loadMetrics = useCallback(
    async (run: TrainingRun) => {
      if (metricsMap[run.name]) return;
      const rows = await invoke<Record<string, unknown>[]>("load_training_metrics", {
        runPath: run.path,
      });
      setMetricsMap((m) => ({ ...m, [run.name]: rows.map(normalizeTrainingMetricRow) }));
    },
    [metricsMap]
  );

  const loadHealth = useCallback(
    async (run: TrainingRun) => {
      if (healthMap[run.name]) return;
      const healthPath = `${run.path}/training_health.jsonl`;
      try {
        const entries = await invoke<TrainingHealthEntry[]>("load_training_health_log", {
          path: healthPath,
        });
        setHealthMap((h) => ({ ...h, [run.name]: entries }));
      } catch {
        setHealthMap((h) => ({ ...h, [run.name]: [] }));
      }
    },
    [healthMap]
  );

  const loadAttention = useCallback(
    async (run: TrainingRun) => {
      if (attentionMap[run.name]) return;
      const attentionPath = `${run.path}/attention_viz.jsonl`;
      try {
        const entries = await invoke<AttentionVizEntry[]>("load_attention_viz_log", {
          path: attentionPath,
        });
        setAttentionMap((a) => ({ ...a, [run.name]: entries }));
      } catch {
        setAttentionMap((a) => ({ ...a, [run.name]: [] }));
      }
    },
    [attentionMap]
  );

  useEffect(() => {
    if (!pendingTrainingRunPath) return;
    const run = runs.find((r) => r.path === pendingTrainingRunPath);
    if (run) {
      handoff(run.path, "training", { storedLabel: run.name, navigate: false });
      setSelected((s) => (s.includes(run.name) ? s : [...s, run.name]));
      void loadMetrics(run);
      void loadHealth(run);
      void loadAttention(run);
      setPendingTrainingRunPath(null);
      return;
    }
    if (!loading && logsPath) {
      void discover();
    }
  }, [
    pendingTrainingRunPath,
    runs,
    loading,
    logsPath,
    discover,
    loadMetrics,
    loadHealth,
    loadAttention,
    setPendingTrainingRunPath,
    handoff,
  ]);

  useEffect(() => {
    if (!recentTrainCompleted || !recentTrainingRunPath) return;
    const run = runs.find((r) => r.path === recentTrainingRunPath);
    if (run) {
      handoff(run.path, "training", { storedLabel: run.name, navigate: false });
      setSelected((s) => (s.includes(run.name) ? s : [...s, run.name]));
      void loadMetrics(run);
      void loadHealth(run);
      void loadAttention(run);
      return;
    }
    if (!loading && logsPath) {
      void discover();
    }
  }, [
    recentTrainCompleted,
    recentTrainingRunPath,
    runs,
    loading,
    logsPath,
    discover,
    loadMetrics,
    loadHealth,
    loadAttention,
    handoff,
  ]);

  const toggleRun = useCallback(
    (run: TrainingRun) => {
      setSelected((s) => {
        if (s.includes(run.name)) return s.filter((r) => r !== run.name);
        return [...s, run.name];
      });
      if (!selected.includes(run.name)) {
        handoff(run.path, "training", { storedLabel: run.name, navigate: false });
      }
      loadMetrics(run);
      loadHealth(run);
      loadAttention(run);
    },
    [selected, handoff, loadMetrics, loadHealth, loadAttention]
  );

  const selectedRunObjects = useMemo(
    () => runs.filter((r) => selected.includes(r.name)),
    [runs, selected]
  );

  const filterRunLabels = useMemo(() => {
    if (processRunLabel) return [processRunLabel];
    const labels = selectedRunObjects.map((r) =>
      portfolioRunLabel(r.path, r.name, projectRoot)
    );
    return labels.length > 0 ? labels : [];
  }, [processRunLabel, projectRoot, selectedRunObjects]);

  const displayedHealth = useMemo(() => {
    const merged: TrainingHealthEntry[] = [];
    if (effectiveLiveHealth.length > 0) {
      merged.push(...effectiveLiveHealth);
    }
    for (const run of selectedRunObjects) {
      const entries = healthMap[run.name];
      if (entries?.length) merged.push(...entries);
    }
    return merged;
  }, [effectiveLiveHealth, healthMap, selectedRunObjects]);

  const displayedAttention = useMemo(() => {
    const merged: AttentionVizEntry[] = [];
    if (effectiveLiveAttention.length > 0) {
      merged.push(...effectiveLiveAttention);
    }
    for (const run of selectedRunObjects) {
      const entries = attentionMap[run.name];
      if (entries?.length) merged.push(...entries);
    }
    return merged;
  }, [effectiveLiveAttention, attentionMap, selectedRunObjects]);

  const runsMetrics = useMemo(() => {
    const result: { name: string; metrics: TrainingMetricsRow[] }[] = [];
    if (selected.includes(LIVE_KEY) && effectiveLiveMetrics.length > 0) {
      result.push({ name: liveRunLabel, metrics: effectiveLiveMetrics });
    }
    for (const r of selectedRunObjects) {
      const metrics = metricsMap[r.name] ?? [];
      if (metrics.length > 0) result.push({ name: r.name, metrics });
    }
    return result;
  }, [selected, selectedRunObjects, effectiveLiveMetrics, liveRunLabel, metricsMap]);

  if (!projectRoot) {
    return (
      <div className="flex flex-col items-center justify-center h-64 text-canvas-muted gap-2">
        <p className="text-sm">Project root not configured.</p>
        <p className="text-xs">Set the project root in Settings to discover training runs.</p>
      </div>
    );
  }

  return (
    <div className="space-y-4">
      <GlobalFilterBar showLogScale runLabels={filterRunLabels} />

      <div className="flex items-center gap-3">
        <button
          onClick={discover}
          disabled={loading}
          className="btn-primary flex items-center gap-2"
        >
          {loading ? <RefreshCw size={14} className="animate-spin" /> : <FolderOpen size={14} />}
          Discover Runs
        </button>
        {logsPath ? (
          <OpenPathToolbar
            path={logsPath}
            projectRoot={projectRoot}
            label="logs"
            chipClassName="max-w-xs"
            handoff={false}
          />
        ) : null}
      </div>

      {runs.length === 0 && !loading && (
        <div className="card text-canvas-muted text-sm flex flex-wrap items-center gap-1">
          <span>No training runs found in</span>
          {logsPath ? (
            <OpenPathToolbar
              path={logsPath}
              projectRoot={projectRoot}
              label="logs"
              handoff={false}
            />
          ) : null}
          <span>.</span>
        </div>
      )}

      {recentTrainId && recentTrainProc && (
        <TrainHpoLivePanel
          cardClassName="border-accent-success/30"
          header={{
            status: activeTrainRunning
              ? "running"
              : recentTrainCompleted
              ? "completed"
              : recentTrainProc.status,
            title: trainHpoLivePanelTitle({
              isRunning: activeTrainRunning,
              status: recentTrainProc.status,
              processId: recentTrainId,
              command: recentTrainProc.command,
            }),
            runLabel: processRunLabel,
            logPath: processLogPath,
            projectRoot,
            showLiveSuffix: activeTrainRunning,
            metricCount: effectiveLiveMetrics.length,
            healthCount: effectiveLiveHealth.length,
            attentionCount: effectiveLiveAttention.length,
            overlaySelect:
              activeTrainId || effectiveLiveMetrics.length > 0
                ? {
                    checked: selected.includes(LIVE_KEY),
                    onChange: () =>
                      setSelected((s) =>
                        s.includes(LIVE_KEY)
                          ? s.filter((k) => k !== LIVE_KEY)
                          : [LIVE_KEY, ...s]
                      ),
                  }
                : undefined,
            navMesh: {
              showHpoLinks:
                activeIsHpo || isHpoProcess(recentTrainId, recentTrainProc.command),
              showOutputBrowser: recentTrainDone && recentTrainCompleted,
              outputRunPath: recentOutputRunPath,
              trainingRunPath: recentTrainingRunPath,
            },
          }}
          progress={
            activeTrainRunning && activeTrainId
              ? {
                  processId: activeTrainId,
                  fallbackValue: latestLiveMetric?.epoch,
                }
              : undefined
          }
          analytics={{
            metrics: effectiveLiveMetrics,
            healthEntries: effectiveLiveHealth,
            attentionEntries: effectiveLiveAttention,
            logScale,
            theme: effectiveTheme,
            exportNamePrefix: "training-monitor",
            isPostRun: recentTrainDone,
            postRunFallback:
              "Post-run shortcuts — open Output Browser or refresh metrics from the completed run",
            showHealthAttention: false,
          }}
          logLines={recentTrainLogLines}
          logTailWaiting={activeTrainRunning}
          footer={
            <ProcessIdFooter
              processId={recentTrainId}
              logPath={processLogPath}
              runLabel={processRunLabel}
              projectRoot={projectRoot}
            />
          }
        />
      )}

      {runs.length > 0 && (
        <div className="card">
          <p className="text-xs font-semibold text-canvas-muted uppercase tracking-wider mb-3">
            Training Runs — select to overlay
          </p>
          <div className="space-y-1">
            {runs.map((run, i) => {
              const liveOffset = activeTrainId ? 1 : 0;
              const color = RUN_COLORS[(i + liveOffset) % RUN_COLORS.length];
              const trailing = (
                <span className="flex gap-2 text-xs text-canvas-muted shrink-0">
                  {run.has_metrics && <span className="text-accent-success">metrics</span>}
                  {run.has_hparams && <span>hparams</span>}
                </span>
              );
              return (
                <div
                  key={run.name}
                  className="flex items-center gap-2 py-1.5 px-2 rounded-lg hover:bg-canvas-hover"
                >
                  <input
                    type="checkbox"
                    checked={selected.includes(run.name)}
                    onChange={() => toggleRun(run)}
                    className="accent-accent-primary shrink-0"
                  />
                  <span
                    className="w-2 h-2 rounded-full shrink-0"
                    style={{ backgroundColor: selected.includes(run.name) ? color : "#444" }}
                  />
                  <LoadedRunRow
                    path={run.path}
                    projectRoot={projectRoot}
                    label={run.name}
                    activeRunLabel={activeRunLabel}
                    pathHandoffs
                    handoffKind="training"
                    trailing={trailing}
                    className="flex-1 min-w-0"
                  />
                </div>
              );
            })}
          </div>
        </div>
      )}

      {displayedHealth.length > 0 && <TrainingHealthPanel entries={displayedHealth} />}

      {displayedAttention.length > 0 && (
        <RuntimeAttentionPanel
          entries={displayedAttention}
          theme={effectiveTheme}
          logScale={logScale}
        />
      )}

      {runsMetrics.length > 0 && <MultiRunChart runsMetrics={runsMetrics} logScale={logScale} />}

      {selectedRunObjects.map((run, i) => {
        const liveOffset = activeTrainId ? 1 : 0;
        const metrics = metricsMap[run.name];
        if (!metrics) {
          return (
            <div key={run.name} className="card text-canvas-muted text-sm">
              Loading {run.name}…
            </div>
          );
        }
        return (
          <RunPanel
            key={run.name}
            run={run}
            metrics={metrics}
            color={RUN_COLORS[(i + liveOffset) % RUN_COLORS.length]}
            projectRoot={projectRoot}
            logScale={logScale}
          />
        );
      })}
    </div>
  );
}
