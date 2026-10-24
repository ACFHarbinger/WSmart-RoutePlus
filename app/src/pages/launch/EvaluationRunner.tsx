/**
 * Evaluation Runner — multi-checkpoint evaluation and comparison (§G.12).
 */
import { useCallback, useEffect, useMemo, useState } from "react";
import { ChevronDown, ChevronUp, FolderOpen, Play, Plus, Terminal } from "lucide-react";
import { OpenPathToolbar } from "../../components/common/OpenPathToolbar";
import { open } from "@tauri-apps/plugin-dialog";
import { useAppStore } from "../../store/app";
import { useGlobalFiltersStore } from "../../store/filters";
import { useLaunchTriggerStore } from "../../store/launchTrigger";
import { TRAIN_PROBLEMS } from "../../store/launchers";
import { useProcessStore } from "../../store/process";
import { useRecentHandoff } from "../../hooks/files/useRecentHandoff";
import { useSpawnProcess } from "../../hooks/process/useSpawnProcess";
import type { ProcessStatus } from "../../types";
import {
  type EvalResult,
  checkpointLabelFromEvalProcess,
  checkpointPathFromEvalCommand,
  collectEvalResultFromLogLines,
  hasEvalMetrics,
  toEvalAnalyticsRows,
} from "../../utils/benchmark/evalResults";
import { useProcessRunLabelBrush } from "../../hooks/brush/useProcessRunLabelBrush";
import { findRecentEvalProcessIds } from "../../utils/process/launcherProcess";
import { brushLogPathFromProcessLines, outputRunPathFromLogLines } from "../../utils/runs/outputRunPath";
import { CheckpointRow, EvalRunnerLiveSection, ResultsGrid, type CheckpointEntry } from "../../components/launch/EvalRunnerSections";

const PROBLEMS = TRAIN_PROBLEMS;
const STRATEGIES = ["greedy", "sampling", "beam"] as const;
const DEVICES = ["cpu", "cuda:0", "cuda:1"] as const;

export function EvaluationRunner() {
  const { pendingCheckpoint, setPendingCheckpoint, setMode, setPendingEvalResults } =
    useAppStore();
  const { projectRoot, handoff } = useRecentHandoff();
  const logScale = useGlobalFiltersStore((s) => s.logScale);
  const { spawn, launching } = useSpawnProcess();

  const [checkpoints, setCheckpoints] = useState<CheckpointEntry[]>([
    { id: "ckpt_0", path: "" },
  ]);

  useEffect(() => {
    if (pendingCheckpoint) {
      handoff(pendingCheckpoint, "checkpoint", { navigate: false });
      setCheckpoints([{ id: "ckpt_pending", path: pendingCheckpoint }]);
      setPendingCheckpoint(null);
    }
  }, [pendingCheckpoint, handoff, setPendingCheckpoint]);

  const [problem, setProblem] = useState<string>("ptp");
  const [datasetPath, setDatasetPath] = useState("");
  const [valSize, setValSize] = useState(10);
  const [strategy, setStrategy] = useState("greedy");
  const [device, setDevice] = useState("cpu");
  const [showAdvanced, setShowAdvanced] = useState(false);
  const [extraOverrides, setExtraOverrides] = useState("");

  const [liveProcessIds, setLiveProcessIds] = useState<string[]>([]);
  const processes = useProcessStore((s) => s.processes);
  const displayProcessIds = useMemo(
    () =>
      liveProcessIds.length > 0
        ? liveProcessIds
        : findRecentEvalProcessIds(processes),
    [liveProcessIds, processes]
  );

  const results = useMemo(() => {
    return displayProcessIds
      .map((id) => {
        const proc = processes[id];
        const checkpointName = checkpointLabelFromEvalProcess(
          id,
          proc?.command ?? ""
        );
        const result = collectEvalResultFromLogLines(
          proc?.logLines ?? [],
          checkpointName
        );
        const checkpointPath = checkpointPathFromEvalCommand(proc?.command ?? "");
        if (checkpointPath) result.checkpointPath = checkpointPath;
        return hasEvalMetrics(result) ? result : null;
      })
      .filter((r): r is EvalResult => r != null);
  }, [displayProcessIds, processes]);

  const validCheckpoints = useMemo(
    () => checkpoints.filter((c) => c.path.trim() !== ""),
    [checkpoints]
  );

  const liveRunSummary = useMemo(() => {
    if (displayProcessIds.length === 0) return null;
    const statuses = displayProcessIds.map(
      (id) => processes[id]?.status ?? "running"
    );
    const running = statuses.filter((s) => s === "running").length;
    const completed = statuses.filter((s) => s === "completed").length;
    const failed = statuses.filter(
      (s) => s === "failed" || s === "cancelled"
    ).length;
    const allDone = running === 0;
    const aggregate: ProcessStatus | "running" = allDone
      ? failed > 0 && completed === 0
        ? "failed"
        : failed > 0
          ? "cancelled"
          : "completed"
      : "running";
    return { running, completed, failed, allDone, aggregate };
  }, [displayProcessIds, processes]);

  const completedCheckpointPath = useMemo(() => {
    if (validCheckpoints.length !== 1) return null;
    if (!liveRunSummary?.allDone || liveRunSummary.aggregate !== "completed") {
      return null;
    }
    return validCheckpoints[0].path;
  }, [validCheckpoints, liveRunSummary]);

  const outputRunPath = useMemo(() => {
    const lines = displayProcessIds.flatMap(
      (id) => processes[id]?.logLines ?? []
    );
    return outputRunPathFromLogLines(lines);
  }, [displayProcessIds, processes]);

  const primaryEvalProcessId = displayProcessIds[0] ?? null;
  const primaryEvalLogLines = primaryEvalProcessId
    ? processes[primaryEvalProcessId]?.logLines
    : undefined;
  const liveRunLabel = useProcessRunLabelBrush(
    primaryEvalProcessId,
    primaryEvalLogLines
  );
  const liveLogPath = useMemo(
    () => brushLogPathFromProcessLines(primaryEvalLogLines ?? [], "sim"),
    [primaryEvalLogLines]
  );

  const singleCheckpointEval = displayProcessIds.length === 1;
  const singleEvalProc = singleCheckpointEval
    ? processes[displayProcessIds[0]]
    : undefined;
  const singleEvalLogLines = singleEvalProc?.logLines ?? [];
  const singleEvalLogTailWaiting = singleEvalProc?.status === "running";

  const addCheckpoint = () => {
    setCheckpoints((prev) => [
      ...prev,
      { id: `ckpt_${Date.now()}`, path: "" },
    ]);
  };

  const removeCheckpoint = (id: string) => {
    setCheckpoints((prev) => prev.filter((c) => c.id !== id));
  };

  const updateCheckpoint = (id: string, path: string) => {
    setCheckpoints((prev) =>
      prev.map((c) => (c.id === id ? { ...c, path } : c))
    );
  };

  const pickCheckpointFile = async (id: string) => {
    const path = (await open({
      filters: [{ name: "Checkpoint", extensions: ["pt", "ckpt", "pth"] }],
    })) as string | null;
    if (!path) return;
    handoff(path, "checkpoint", { navigate: false });
    updateCheckpoint(id, path);
  };

  const pickDataset = async () => {
    const path = (await open({
      filters: [{ name: "Dataset", extensions: ["pkl", "json", "csv"] }],
    })) as string | null;
    if (!path) return;
    if (/\.csv$/i.test(path)) {
      handoff(path, "csv", { navigate: false });
    }
    setDatasetPath(path);
  };

  const previewArgs = useMemo(() => {
    const ckpt = validCheckpoints[0];
    const base = [
      `eval.problem=${problem}`,
      `eval.val_size=${valSize}`,
      `eval.decoding.strategy=${strategy}`,
      `eval.device=${device}`,
      ...(ckpt ? [`eval.policy.model.load_path=${ckpt.path}`] : []),
      ...(datasetPath ? [`eval.datasets=[${datasetPath}]`] : []),
    ];
    const extra = extraOverrides
      .split("\n")
      .map((l) => l.trim())
      .filter(Boolean);
    return [...base, ...extra];
  }, [
    problem,
    valSize,
    strategy,
    device,
    validCheckpoints,
    datasetPath,
    extraOverrides,
  ]);

  const commandPreview =
    validCheckpoints.length > 1
      ? `# ${validCheckpoints.length} checkpoints — one process per checkpoint\npython main.py eval \\\n  ${previewArgs.join(" \\\n  ")}`
      : `python main.py eval \\\n  ${previewArgs.join(" \\\n  ")}`;

  const openInAnalytics = useCallback(() => {
    setPendingEvalResults(toEvalAnalyticsRows(results));
    setMode("benchmark");
  }, [results, setPendingEvalResults, setMode]);

  const launch = useCallback(async () => {
    if (!projectRoot || validCheckpoints.length === 0) return;

    const extra = extraOverrides
      .split("\n")
      .map((l) => l.trim())
      .filter(Boolean);

    const launchedIds: string[] = [];
    const baseTs = Date.now();

    for (const [idx, ckpt] of validCheckpoints.entries()) {
      const ckptName = ckpt.path.split(/[/\\]/).pop() ?? ckpt.id;
      const procId = `eval_${ckptName}_${baseTs}_${idx}`;
      launchedIds.push(procId);
      const hydraArgs = [
        `eval.problem=${problem}`,
        `eval.val_size=${valSize}`,
        `eval.decoding.strategy=${strategy}`,
        `eval.device=${device}`,
        `eval.policy.model.load_path=${ckpt.path}`,
        ...(datasetPath ? [`eval.datasets=[${datasetPath}]`] : []),
        ...extra,
      ];
      await spawn({
        id: procId,
        pythonArgs: ["main.py", "eval", ...hydraArgs],
        workingDir: projectRoot,
      });
    }

    setLiveProcessIds(launchedIds);
  }, [
    projectRoot,
    validCheckpoints,
    problem,
    valSize,
    strategy,
    device,
    datasetPath,
    extraOverrides,
    spawn,
  ]);

  const evalNonce = useLaunchTriggerStore((s) => s.evalNonce);
  useEffect(() => {
    if (evalNonce > 0) void launch();
  }, [evalNonce, launch]);

  return (
    <div className="space-y-4 max-w-2xl">
      {/* Checkpoints */}
      <div className="card space-y-3">
        <div className="flex items-center justify-between">
          <h2 className="text-sm font-semibold text-gray-200">Checkpoints</h2>
          <button
            onClick={addCheckpoint}
            className="btn-ghost text-xs flex items-center gap-1"
          >
            <Plus size={12} />
            Add
          </button>
        </div>

        <div className="space-y-2">
          {checkpoints.map((c) => (
            <CheckpointRow
              key={c.id}
              entry={c}
              projectRoot={projectRoot}
              onRemove={() => removeCheckpoint(c.id)}
              onPickFile={() => void pickCheckpointFile(c.id)}
              onChange={(path) => updateCheckpoint(c.id, path)}
            />
          ))}
        </div>

        {validCheckpoints.length > 1 && (
          <p className="text-xs text-canvas-muted">
            {validCheckpoints.length} checkpoints — each will be evaluated in a
            separate process.
          </p>
        )}
      </div>

      {/* Eval parameters */}
      <div className="card space-y-4">
        <h2 className="text-sm font-semibold text-gray-200">Evaluation</h2>

        <div className="flex flex-col gap-1.5">
          <label className="text-xs text-canvas-muted">
            Dataset (optional — uses config default if empty)
          </label>
          <div className="flex gap-2">
            <input
              type="text"
              className="input-base font-mono text-xs flex-1"
              value={datasetPath}
              onChange={(e) => setDatasetPath(e.target.value)}
              placeholder="path/to/dataset.pkl"
            />
            <button
              onClick={() => void pickDataset()}
              className="btn-ghost p-1.5 text-canvas-muted hover:text-gray-200"
            >
              <FolderOpen size={13} />
            </button>
          </div>
          {datasetPath.trim() ? (
            <OpenPathToolbar
              path={datasetPath.trim()}
              projectRoot={projectRoot}
              chipClassName="max-w-full"
              handoff={false}
            />
          ) : null}
        </div>

        <div className="flex flex-wrap gap-4">
          <div className="flex flex-col gap-1.5">
            <label className="text-xs text-canvas-muted">Problem</label>
            <select
              className="select-base w-36"
              value={problem}
              onChange={(e) => setProblem(e.target.value)}
            >
              {PROBLEMS.map((p) => (
                <option key={p} value={p}>
                  {p}
                </option>
              ))}
            </select>
          </div>

          <div className="flex flex-col gap-1.5">
            <label className="text-xs text-canvas-muted">Decoding Strategy</label>
            <select
              className="select-base w-36"
              value={strategy}
              onChange={(e) => setStrategy(e.target.value)}
            >
              {STRATEGIES.map((s) => (
                <option key={s} value={s}>
                  {s}
                </option>
              ))}
            </select>
          </div>

          <div className="flex flex-col gap-1.5">
            <label className="text-xs text-canvas-muted">Device</label>
            <select
              className="select-base w-28"
              value={device}
              onChange={(e) => setDevice(e.target.value)}
            >
              {DEVICES.map((d) => (
                <option key={d} value={d}>
                  {d}
                </option>
              ))}
            </select>
          </div>

          <div className="flex flex-col gap-1.5">
            <label className="text-xs text-canvas-muted">Val Samples</label>
            <input
              type="number"
              className="input-base font-mono text-sm w-24"
              value={valSize}
              min={1}
              onChange={(e) => setValSize(Number(e.target.value))}
            />
          </div>
        </div>
      </div>

      {/* Advanced */}
      <div className="card">
        <button
          className="w-full flex items-center justify-between text-sm font-medium text-gray-300"
          onClick={() => setShowAdvanced((v) => !v)}
        >
          <span>Extra Hydra Overrides</span>
          {showAdvanced ? <ChevronUp size={14} /> : <ChevronDown size={14} />}
        </button>
        {showAdvanced && (
          <textarea
            className="input-base w-full font-mono text-xs h-20 resize-y mt-3"
            value={extraOverrides}
            onChange={(e) => setExtraOverrides(e.target.value)}
            placeholder="eval.batch_size=128"
            spellCheck={false}
          />
        )}
      </div>

      {/* Command preview */}
      <div className="card space-y-2">
        <div className="flex items-center gap-2 text-xs text-canvas-muted">
          <Terminal size={12} />
          Command preview
        </div>
        <pre className="font-mono text-xs text-accent-secondary whitespace-pre-wrap bg-canvas-bg rounded-lg p-3">
          {validCheckpoints.length > 0
            ? commandPreview
            : "# Add a checkpoint path above to see the command"}
        </pre>
      </div>

      {results.length > 0 && (
        <ResultsGrid
          results={results}
          logScale={logScale}
          runLabels={liveRunLabel ? [liveRunLabel] : []}
          projectRoot={projectRoot}
          onOpenAnalytics={openInAnalytics}
        />
      )}

      {/* Launch */}
      <div className="flex items-center gap-3">
        {!projectRoot && (
          <p className="text-xs text-accent-warning">
            Configure Project Root in Settings first.
          </p>
        )}
        {validCheckpoints.length === 0 && (
          <p className="text-xs text-accent-warning">
            Add at least one checkpoint path.
          </p>
        )}
        <button
          onClick={() => void launch()}
          disabled={launching || !projectRoot || validCheckpoints.length === 0}
          className="btn-primary flex items-center gap-2"
        >
          <Play size={14} />
          {launching
            ? "Launching…"
            : validCheckpoints.length > 1
              ? `Evaluate ${validCheckpoints.length} Checkpoints`
              : "Evaluate"}
        </button>
      </div>

      {liveRunSummary && (
        <EvalRunnerLiveSection
          liveRunSummary={liveRunSummary}
          displayProcessIds={displayProcessIds}
          processes={processes}
          results={results}
          liveRunLabel={liveRunLabel}
          liveLogPath={liveLogPath}
          projectRoot={projectRoot}
          outputRunPath={outputRunPath}
          completedCheckpointPath={completedCheckpointPath}
          openInAnalytics={openInAnalytics}
          singleCheckpointEval={singleCheckpointEval}
          singleEvalLogLines={singleEvalLogLines}
          singleEvalLogTailWaiting={singleEvalLogTailWaiting}
        />
      )}
    </div>
  );
}
