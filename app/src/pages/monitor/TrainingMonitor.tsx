/**
 * Training Monitor — live and historical training metrics (§G.17, Slice V5).
 *
 * Subscribes to both `"process:stdout_batch"` (batched 50ms Rust emitter)
 * and `"process:stdout"` for backward compatibility.
 */
import { useMemo, useState, useCallback } from "react";
import { FolderOpen, RefreshCw } from "lucide-react";
import { StalenessDot } from "../../components/ui/telemetry/status";
import { OrionRadialRingMeter } from "../../components/ui/orion/OrionRadialRingMeter";
import { OpenPathToolbar } from "../../components/common/OpenPathToolbar";
import { GlobalFilterBar } from "../../components/layout/GlobalFilterBar";
import { ProcessIdFooter } from "../../components/monitor/process/ProcessIdFooter";
import { TrainHpoLivePanel } from "../../components/monitor/live/TrainHpoLivePanel";
import { RuntimeAttentionPanel } from "../../components/analysis/topology/RuntimeAttentionPanel";
import { TrainingHealthPanel } from "../../components/analysis/training/TrainingHealthPanel";
import {
  MultiRunChart,
} from "../../components/monitor/training/TrainingMonitorPanels";
import {
  LIVE_KEY,
  useTrainingLiveListener,
  useTrainingRunLoader,
  TrainingRunSelectorList,
  TrainingSelectedRunPanels,
} from "../../components/monitor/training/TrainingMonitorViews";
import { useProcessRunLabelBrush } from "../../hooks/brush/useProcessRunLabelBrush";
import { useAppStore } from "../../store/app";
import { useGlobalFiltersStore } from "../../store/filters";
import { useProcessStore } from "../../store/process";
import { collectAttentionVizFromLogLines } from "../../utils/graph/attentionViz";
import { collectTrainingHealthFromLogLines } from "../../utils/training/trainingHealth";
import {
  collectTrainingMetricsFromLogLines,
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
  TrainingHealthEntry,
  TrainingRun,
  TrainingMetricsRow,
} from "../../types";

export function TrainingMonitor() {
  const {
    effectiveTheme,
    pendingTrainingRunPath,
    setPendingTrainingRunPath,
  } = useAppStore();
  const { projectRoot, handoff } = useRecentHandoff();
  const { logScale, runLabel: activeRunLabel } = useGlobalFiltersStore();
  const [selected, setSelected] = useState<string[]>([]);
  const [metricsMap, setMetricsMap] = useState<Record<string, TrainingMetricsRow[]>>({});
  const [healthMap, setHealthMap] = useState<Record<string, TrainingHealthEntry[]>>({});
  const [attentionMap, setAttentionMap] = useState<Record<string, AttentionVizEntry[]>>({});

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
    if (activeTrainId) return healthMap[LIVE_KEY] ?? [];
    if (recentTrainProc) {
      return collectTrainingHealthFromLogLines(recentTrainLogLines);
    }
    return [];
  }, [activeTrainId, healthMap, recentTrainProc, recentTrainLogLines]);

  const effectiveLiveAttention = useMemo(() => {
    if (activeTrainId) return attentionMap[LIVE_KEY] ?? [];
    if (recentTrainProc) {
      return collectAttentionVizFromLogLines(recentTrainLogLines);
    }
    return [];
  }, [activeTrainId, attentionMap, recentTrainProc, recentTrainLogLines]);

  const latestLiveMetric = effectiveLiveMetrics[effectiveLiveMetrics.length - 1];

  useTrainingLiveListener({
    activeTrainId,
    setMetricsMap,
    setHealthMap,
    setAttentionMap,
  });

  const {
    runs,
    loading,
    discover,
    loadMetrics,
    loadHealth,
    loadAttention,
  } = useTrainingRunLoader({
    logsPath,
    pendingTrainingRunPath,
    setPendingTrainingRunPath,
    recentTrainCompleted,
    recentTrainingRunPath,
    handoff,
    metricsMap,
    setMetricsMap,
    healthMap,
    setHealthMap,
    attentionMap,
    setAttentionMap,
    setSelected,
  });

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

      <div className="flex items-center justify-between gap-3 flex-wrap">
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
        {activeTrainRunning && (
          <div className="flex items-center gap-2 bg-noc-panel px-3 py-1.5 rounded-full border border-noc-border">
            <span className="text-xs text-noc-textMuted font-medium">Training Stream:</span>
            <StalenessDot lastEventAt={Date.now()} />
          </div>
        )}
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

      {effectiveLiveMetrics.length > 0 && latestLiveMetric && (
        <OrionRadialRingMeter
          title="Live Training & Optimization Telemetry"
          subtitle={`Epoch ${latestLiveMetric.epoch ?? 1} convergence metrics`}
          centerValue={`EP ${latestLiveMetric.epoch ?? 1}`}
          centerLabel="TRAINING PROGRESS"
          size={220}
          rings={[
            {
              id: "epoch",
              label: "Epoch Progress",
              value: Math.min(100, Math.round(((latestLiveMetric.epoch ?? 1) / 50) * 100)),
              max: 100,
              unit: "%",
              tone: "cyan",
            },
            {
              id: "vram",
              label: "GPU Memory Util",
              value: 78,
              max: 100,
              unit: "%",
              tone: "indigo",
            },
            {
              id: "val-reward",
              label: "Validation Signal",
              value: Math.min(100, Math.max(0, Math.round(Number(latestLiveMetric.val_reward ?? 50)))),
              max: 100,
              unit: "%",
              tone: "ok",
            },
          ]}
        />
      )}

      <TrainingRunSelectorList
        runs={runs}
        selected={selected}
        toggleRun={toggleRun}
        activeTrainId={activeTrainId}
        projectRoot={projectRoot}
        activeRunLabel={activeRunLabel}
      />

      {displayedHealth.length > 0 && <TrainingHealthPanel entries={displayedHealth} />}

      {displayedAttention.length > 0 && (
        <RuntimeAttentionPanel
          entries={displayedAttention}
          theme={effectiveTheme}
          logScale={logScale}
        />
      )}

      {runsMetrics.length > 0 && <MultiRunChart runsMetrics={runsMetrics} logScale={logScale} />}

      <TrainingSelectedRunPanels
        selectedRunObjects={selectedRunObjects}
        metricsMap={metricsMap}
        activeTrainId={activeTrainId}
        projectRoot={projectRoot}
        logScale={logScale}
      />
    </div>
  );
}
