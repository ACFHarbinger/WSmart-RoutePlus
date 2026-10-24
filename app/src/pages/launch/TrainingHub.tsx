/**
 * Training Hub — full-featured training / HPO / eval launcher (§G.10).
 *
 * Form parameters mirror the controller justfile and Hydra configs:
 *   train:    envs@train.env, models, model.encoder.type, train.batch_size, seed
 *   hpo:      hpo.n_trials, hpo.method, hpo.num_workers
 *   eval:     eval.policy.model.load_path, eval.datasets, eval.val_size, eval.decoding.strategy
 */
import { useCallback, useEffect, useMemo, useState } from "react";
import { Play, ChevronDown, ChevronUp, Terminal } from "lucide-react";
import { open } from "@tauri-apps/plugin-dialog";
import { useAppStore } from "../../store/app";
import { useGlobalFiltersStore } from "../../store/filters";
import { useLaunchTriggerStore } from "../../store/launchTrigger";
import { useTrainHubStore } from "../../store/launchers";
import { useProcessStore } from "../../store/process";
import { useRecentHandoff } from "../../hooks/files/useRecentHandoff";
import { useSpawnProcess } from "../../hooks/process/useSpawnProcess";
import { brushLogPathFromProcessLines, outputRunPathFromLogLines } from "../../utils/runs/outputRunPath";
import { trainingRunPathFromLogLines } from "../../utils/training/trainingRunPath";
import { collectAttentionVizFromLogLines } from "../../utils/graph/attentionViz";
import { collectTrainingHealthFromLogLines } from "../../utils/training/trainingHealth";
import { collectTrainingMetricsFromLogLines } from "../../utils/training/trainingMetrics";
import {
  checkpointLabelFromEvalProcess,
  collectEvalResultFromLogLines,
  evalLivePanelTitle,
  hasEvalMetrics,
  toEvalAnalyticsRows,
} from "../../utils/benchmark/evalResults";
import { useProcessRunLabelBrush } from "../../hooks/brush/useProcessRunLabelBrush";
import { findRecentLauncherProcessId } from "../../utils/process/launcherProcess";
import { findRecentHpoProcessId, findRecentMetaTrainProcessId, findRecentTrainProcessId, trainHpoLivePanelTitle } from "../../utils/training/trainingProcess";
import { buildTrainHubHydraArgs, PROBLEMS, TrainHubLiveSection, type TrainHubMode } from "../../components/launch/TrainHubSections";
import { TrainHubModeSpecificCard } from "../../components/launch/TrainHubForms";
import type { ProcessEntry } from "../../types";

function findRecentHubProcessId(
  processes: Record<string, ProcessEntry>,
  hubMode: TrainHubMode
): string | null {
  if (hubMode === "hpo") return findRecentHpoProcessId(processes);
  if (hubMode === "meta") return findRecentMetaTrainProcessId(processes);
  if (hubMode === "eval") return findRecentLauncherProcessId(processes, "eval");
  return findRecentTrainProcessId(processes);
}

function SelectField<T extends string>({
  label,
  value,
  onChange,
  options,
}: {
  label: string;
  value: T;
  onChange: (v: T) => void;
  options: readonly T[];
}) {
  return (
    <div className="flex flex-col gap-1">
      <label className="text-xs text-canvas-muted">{label}</label>
      <select
        className="select-base w-36"
        value={value}
        onChange={(e) => onChange(e.target.value as T)}
      >
        {options.map((o) => (
          <option key={o} value={o}>
            {o}
          </option>
        ))}
      </select>
    </div>
  );
}

export function TrainingHub() {
  const { effectiveTheme, setMode, setPendingEvalResults } = useAppStore();
  const { projectRoot, handoff } = useRecentHandoff();
  const logScale = useGlobalFiltersStore((s) => s.logScale);
  const { spawn, launching } = useSpawnProcess();

  const {
    trainMode: mode,
    problem,
    seed,
    wandb,
    extraOverrides,
    model,
    encoder,
    batchSize,
    epochs,
    hpoTrials,
    hpoMethod,
    hpoWorkers,
    metaStrategy,
    metaLr,
    metaHistoryLength,
    mrlBatchSize,
    mrlStep,
    checkpointPath,
    evalDataset,
    evalSamples,
    evalStrategy,
    patch,
  } = useTrainHubStore();

  const setTrainMode = (v: TrainHubMode) => patch({ trainMode: v });
  const setProblem = (v: string) => patch({ problem: v });
  const setSeed = (v: number) => patch({ seed: v });
  const setWandb = (v: boolean) => patch({ wandb: v });
  const setExtraOverrides = (v: string) => patch({ extraOverrides: v });
  const setModel = (v: string) => patch({ model: v });
  const setEncoder = (v: string) => patch({ encoder: v });
  const setBatchSize = (v: number) => patch({ batchSize: v });
  const setEpochs = (v: number) => patch({ epochs: v });
  const setHpoTrials = (v: number) => patch({ hpoTrials: v });
  const setHpoMethod = (v: string) => patch({ hpoMethod: v });
  const setHpoWorkers = (v: number) => patch({ hpoWorkers: v });
  const setMetaStrategy = (v: string) => patch({ metaStrategy: v });
  const setMetaLr = (v: number) => patch({ metaLr: v });
  const setMetaHistoryLength = (v: number) => patch({ metaHistoryLength: v });
  const setMrlBatchSize = (v: number) => patch({ mrlBatchSize: v });
  const setMrlStep = (v: number) => patch({ mrlStep: v });
  const setCheckpointPath = (v: string) => patch({ checkpointPath: v });
  const setEvalDataset = (v: string) => patch({ evalDataset: v });
  const setEvalSamples = (v: number) => patch({ evalSamples: v });
  const setEvalStrategy = (v: string) => patch({ evalStrategy: v });

  const [showAdvanced, setShowAdvanced] = useState(false);

  const [liveProcessId, setLiveProcessId] = useState<string | null>(null);
  const processes = useProcessStore((s) => s.processes);
  const displayProcessId = useMemo(
    () => liveProcessId ?? findRecentHubProcessId(processes, mode),
    [liveProcessId, processes, mode]
  );
  const displayProc = displayProcessId ? processes[displayProcessId] : null;
  const runStatus = displayProc?.status ?? null;

  const pickCheckpoint = async () => {
    const path = (await open({
      filters: [{ name: "Checkpoint", extensions: ["pt", "ckpt", "pth"] }],
    })) as string | null;
    if (!path) return;
    handoff(path, "checkpoint", { navigate: false });
    setCheckpointPath(path);
  };

  const pickDataset = async () => {
    const path = (await open({
      filters: [{ name: "Dataset", extensions: ["pkl", "json", "csv"] }],
    })) as string | null;
    if (!path) return;
    if (/\.csv$/i.test(path)) {
      handoff(path, "csv", { navigate: false });
    }
    setEvalDataset(path);
  };

  const hydraArgs = useMemo(
    () =>
      buildTrainHubHydraArgs({
        mode,
        seed,
        wandb,
        problem,
        model,
        encoder,
        batchSize,
        epochs,
        hpoTrials,
        hpoMethod,
        hpoWorkers,
        metaStrategy,
        metaLr,
        metaHistoryLength,
        mrlBatchSize,
        mrlStep,
        checkpointPath,
        evalDataset,
        evalSamples,
        evalStrategy,
        extraOverrides,
      }),
    [
      mode,
      seed,
      wandb,
      problem,
      model,
      encoder,
      batchSize,
      epochs,
      hpoTrials,
      hpoMethod,
      hpoWorkers,
      metaStrategy,
      metaLr,
      metaHistoryLength,
      mrlBatchSize,
      mrlStep,
      checkpointPath,
      evalDataset,
      evalSamples,
      evalStrategy,
      extraOverrides,
    ]
  );

  const entrypoint =
    mode === "meta"
      ? "meta_train"
      : mode === "train" || mode === "hpo"
        ? "train"
        : "eval";
  const commandPreview = `python main.py ${entrypoint} \\\n  ${hydraArgs.join(" \\\n  ")}`;

  const launch = useCallback(async () => {
    if (!projectRoot) return;
    const procId = `${mode}_${Date.now()}`;
    setLiveProcessId(procId);
    await spawn({
      id: procId,
      pythonArgs: ["main.py", entrypoint, ...hydraArgs],
      workingDir: projectRoot,
    });
  }, [projectRoot, mode, entrypoint, hydraArgs, spawn]);

  const trainNonce = useLaunchTriggerStore((s) => s.trainNonce);
  useEffect(() => {
    if (trainNonce > 0) void launch();
  }, [trainNonce, launch]);

  const liveLogLines = displayProc?.logLines ?? [];
  const liveRunLabel = useProcessRunLabelBrush(displayProcessId, liveLogLines);
  const liveMetrics = useMemo(
    () => collectTrainingMetricsFromLogLines(liveLogLines),
    [liveLogLines]
  );
  const liveHealth = useMemo(
    () => collectTrainingHealthFromLogLines(liveLogLines),
    [liveLogLines]
  );
  const liveAttention = useMemo(
    () => collectAttentionVizFromLogLines(liveLogLines),
    [liveLogLines]
  );
  const outputRunPath = useMemo(
    () => outputRunPathFromLogLines(liveLogLines),
    [liveLogLines]
  );
  const trainingRunPath = useMemo(
    () => trainingRunPathFromLogLines(liveLogLines),
    [liveLogLines]
  );
  const liveLogPath = useMemo(() => {
    const kind = mode === "eval" ? "sim" : "train";
    return brushLogPathFromProcessLines(liveLogLines, kind);
  }, [liveLogLines, mode]);

  const isDone = runStatus !== null && runStatus !== "running";
  const latestMetric = liveMetrics[liveMetrics.length - 1];
  const showTrainingAnalytics =
    displayProcessId != null &&
    (mode === "train" || mode === "hpo" || mode === "meta");
  const trainHpoLiveTitle = trainHpoLivePanelTitle({
    isRunning: !isDone,
    status: runStatus ?? undefined,
    processId: displayProcessId ?? undefined,
    command: displayProc?.command,
    kind: mode === "hpo" ? "hpo" : mode === "train" ? "train" : undefined,
  });

  const evalCheckpointName = useMemo(() => {
    if (!displayProcessId || !displayProc) {
      return checkpointPath.split(/[/\\]/).pop() ?? "checkpoint";
    }
    return checkpointLabelFromEvalProcess(
      displayProcessId,
      displayProc.command ?? ""
    );
  }, [displayProcessId, displayProc, checkpointPath]);

  const evalResult = useMemo(() => {
    if (mode !== "eval" || !displayProc) return null;
    const result = collectEvalResultFromLogLines(liveLogLines, evalCheckpointName);
    const trimmed = checkpointPath.trim();
    if (trimmed) result.checkpointPath = trimmed;
    return result;
  }, [mode, displayProc, liveLogLines, evalCheckpointName, checkpointPath]);

  const openEvalInAnalytics = useCallback(() => {
    if (!evalResult || !hasEvalMetrics(evalResult)) return;
    setPendingEvalResults(toEvalAnalyticsRows([evalResult]));
    setMode("benchmark");
  }, [evalResult, setPendingEvalResults, setMode]);

  const evalLiveTitle = evalLivePanelTitle({
    isRunning: !isDone,
    status: runStatus ?? undefined,
  });

  const completedEvalCheckpointPath =
    mode === "eval" && isDone && runStatus === "completed"
      ? checkpointPath || null
      : null;

  return (
    <div className="space-y-4 max-w-2xl">
      {/* Mode selector */}
      <div className="card space-y-3">
        <h2 className="text-sm font-semibold text-gray-200">Mode</h2>
        <div className="flex gap-2">
          {(["train", "hpo", "meta", "eval"] as const).map((m) => (
            <button
              key={m}
              onClick={() => setTrainMode(m)}
              className={
                mode === m
                  ? "btn-primary text-sm py-1.5 px-4"
                  : "btn-ghost text-sm py-1.5 px-4"
              }
            >
              {m === "train"
                ? "Train"
                : m === "hpo"
                  ? "HPO Sweep"
                  : m === "meta"
                    ? "Meta-RL"
                    : "Evaluate"}
            </button>
          ))}
        </div>
      </div>

      {/* Shared: problem + seed + WandB */}
      <div className="card space-y-4">
        <h2 className="text-sm font-semibold text-gray-200">Common</h2>
        <div className="flex flex-wrap gap-4 items-end">
          {mode !== "meta" && (
            <SelectField
              label="Problem"
              value={problem}
              onChange={setProblem}
              options={PROBLEMS}
            />
          )}
          <div className="flex flex-col gap-1">
            <label className="text-xs text-canvas-muted">Seed</label>
            <input
              type="number"
              className="input-base font-mono text-sm w-24"
              value={seed}
              min={0}
              onChange={(e) => setSeed(Number(e.target.value))}
            />
          </div>
          <label className="flex items-center gap-2 cursor-pointer text-sm text-gray-300 self-end pb-2">
            <input
              type="checkbox"
              className="accent-accent-primary"
              checked={wandb}
              onChange={(e) => setWandb(e.target.checked)}
            />
            WandB logging
          </label>
        </div>
      </div>

      <TrainHubModeSpecificCard
        mode={mode}
        model={model}
        setModel={setModel}
        encoder={encoder}
        setEncoder={setEncoder}
        batchSize={batchSize}
        setBatchSize={setBatchSize}
        epochs={epochs}
        setEpochs={setEpochs}
        hpoMethod={hpoMethod}
        setHpoMethod={setHpoMethod}
        hpoTrials={hpoTrials}
        setHpoTrials={setHpoTrials}
        hpoWorkers={hpoWorkers}
        setHpoWorkers={setHpoWorkers}
        metaStrategy={metaStrategy}
        setMetaStrategy={setMetaStrategy}
        metaLr={metaLr}
        setMetaLr={setMetaLr}
        metaHistoryLength={metaHistoryLength}
        setMetaHistoryLength={setMetaHistoryLength}
        mrlBatchSize={mrlBatchSize}
        setMrlBatchSize={setMrlBatchSize}
        mrlStep={mrlStep}
        setMrlStep={setMrlStep}
        checkpointPath={checkpointPath}
        setCheckpointPath={setCheckpointPath}
        pickCheckpoint={() => void pickCheckpoint()}
        evalDataset={evalDataset}
        setEvalDataset={setEvalDataset}
        pickDataset={() => void pickDataset()}
        evalStrategy={evalStrategy}
        setEvalStrategy={setEvalStrategy}
        evalSamples={evalSamples}
        setEvalSamples={setEvalSamples}
        projectRoot={projectRoot}
      />

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
            placeholder="some.param=value"
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
          {commandPreview}
        </pre>
      </div>

      {/* Launch */}
      <div className="flex items-center gap-3">
        {!projectRoot && (
          <p className="text-xs text-accent-warning">
            Configure Project Root in Settings first.
          </p>
        )}
        <button
          onClick={() => void launch()}
          disabled={launching || !projectRoot}
          className="btn-primary flex items-center gap-2"
        >
          <Play size={14} />
          {launching ? "Launching…" : `Start ${mode}`}
        </button>
      </div>

      {displayProcessId && (
        <TrainHubLiveSection
          displayProcessId={displayProcessId}
          mode={mode}
          displayProc={displayProc}
          isDone={isDone}
          runStatus={runStatus}
          evalLiveTitle={evalLiveTitle}
          trainHpoLiveTitle={trainHpoLiveTitle}
          liveRunLabel={liveRunLabel}
          liveLogPath={liveLogPath}
          projectRoot={projectRoot}
          outputRunPath={outputRunPath}
          trainingRunPath={trainingRunPath}
          completedEvalCheckpointPath={completedEvalCheckpointPath}
          evalResult={evalResult}
          openEvalInAnalytics={openEvalInAnalytics}
          liveLogLines={liveLogLines}
          evalCheckpointName={evalCheckpointName}
          checkpointPath={checkpointPath}
          liveMetrics={liveMetrics}
          liveHealth={liveHealth}
          liveAttention={liveAttention}
          showTrainingAnalytics={showTrainingAnalytics}
          epochs={epochs}
          latestMetric={latestMetric}
          logScale={logScale}
          effectiveTheme={effectiveTheme}
        />
      )}
    </div>
  );
}
