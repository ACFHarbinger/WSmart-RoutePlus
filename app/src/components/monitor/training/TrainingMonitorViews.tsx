import { useEffect, useCallback, useState } from "react";
import { invoke } from "@tauri-apps/api/core";
import { listen } from "@tauri-apps/api/event";
import { LoadedRunRow } from "../../common/LoadedRunRow";
import { RunPanel, RUN_COLORS } from "./TrainingMonitorPanels";
import {
  parseAttentionVizLine,
} from "../../../utils/graph/attentionViz";
import {
  parseTrainingHealthLine,
} from "../../../utils/training/trainingHealth";
import {
  parseTrainingMetricLine,
  normalizeTrainingMetricRow,
} from "../../../utils/training/trainingMetrics";
import type { useRecentHandoff } from "../../../hooks/files/useRecentHandoff";
import type {
  AttentionVizEntry,
  StdoutBatch,
  StdoutLine,
  TrainingHealthEntry,
  TrainingMetricsRow,
  TrainingRun,
} from "../../../types";

export const LIVE_KEY = "__live__";
export const LIVE_HEALTH_KEY = "__live_health__";
export const LIVE_ATTENTION_KEY = "__live_attention__";

interface UseTrainingLiveListenerProps {
  activeTrainId: string | null;
  setMetricsMap: React.Dispatch<React.SetStateAction<Record<string, TrainingMetricsRow[]>>>;
  setHealthMap: React.Dispatch<React.SetStateAction<Record<string, TrainingHealthEntry[]>>>;
  setAttentionMap: React.Dispatch<React.SetStateAction<Record<string, AttentionVizEntry[]>>>;
}

export function useTrainingLiveListener({
  activeTrainId,
  setMetricsMap,
  setHealthMap,
  setAttentionMap,
}: UseTrainingLiveListenerProps) {
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
  }, [activeTrainId, setMetricsMap, setHealthMap, setAttentionMap]);
}

interface UseTrainingRunLoaderProps {
  logsPath: string;
  pendingTrainingRunPath: string | null;
  setPendingTrainingRunPath: (path: string | null) => void;
  recentTrainCompleted: boolean;
  recentTrainingRunPath: string | null;
  handoff: ReturnType<typeof useRecentHandoff>["handoff"];
  metricsMap: Record<string, TrainingMetricsRow[]>;
  setMetricsMap: React.Dispatch<React.SetStateAction<Record<string, TrainingMetricsRow[]>>>;
  healthMap: Record<string, TrainingHealthEntry[]>;
  setHealthMap: React.Dispatch<React.SetStateAction<Record<string, TrainingHealthEntry[]>>>;
  attentionMap: Record<string, AttentionVizEntry[]>;
  setAttentionMap: React.Dispatch<React.SetStateAction<Record<string, AttentionVizEntry[]>>>;
  setSelected: React.Dispatch<React.SetStateAction<string[]>>;
}

export function useTrainingRunLoader({
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
}: UseTrainingRunLoaderProps) {
  const [runs, setRuns] = useState<TrainingRun[]>([]);
  const [loading, setLoading] = useState(false);

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
    [metricsMap, setMetricsMap]
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
    [healthMap, setHealthMap]
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
    [attentionMap, setAttentionMap]
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
    setSelected,
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
    setSelected,
  ]);

  return { runs, loading, discover, loadMetrics, loadHealth, loadAttention };
}

interface TrainingRunSelectorListProps {
  runs: TrainingRun[];
  selected: string[];
  toggleRun: (run: TrainingRun) => void;
  activeTrainId: string | null;
  projectRoot: string;
  activeRunLabel: string | null;
}

export function TrainingRunSelectorList({
  runs,
  selected,
  toggleRun,
  activeTrainId,
  projectRoot,
  activeRunLabel,
}: TrainingRunSelectorListProps) {
  if (runs.length === 0) return null;

  return (
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
  );
}

interface TrainingSelectedRunPanelsProps {
  selectedRunObjects: TrainingRun[];
  metricsMap: Record<string, TrainingMetricsRow[]>;
  activeTrainId: string | null;
  projectRoot: string;
  logScale: boolean;
}

export function TrainingSelectedRunPanels({
  selectedRunObjects,
  metricsMap,
  activeTrainId,
  projectRoot,
  logScale,
}: TrainingSelectedRunPanelsProps) {
  return (
    <>
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
    </>
  );
}
