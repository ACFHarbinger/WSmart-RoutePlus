import { useMemo, useRef } from "react";
import ReactECharts from "echarts-for-react";
import type EChartsReact from "echarts-for-react";
import { ChartExportButtons } from "../common/ChartExportButtons";
import { GlobalFilterBar } from "../layout/GlobalFilterBar";
import { EvalCheckpointLiveCard } from "../monitor/eval/EvalCheckpointLiveCard";
import { EvalResultCard } from "../monitor/eval/EvalResultCard";
import { LauncherLivePanel } from "../monitor/live/LauncherLivePanel";
import { TrainHpoLivePanel } from "../monitor/live/TrainHpoLivePanel";
import { ProcessIdFooter } from "../monitor/process/ProcessIdFooter";
import { TRAIN_MODELS, TRAIN_PROBLEMS } from "../../store/launchers";
import { hasEvalMetrics, type EvalResult } from "../../utils/benchmark/evalResults";
import type {
  AttentionVizEntry,
  ProcessEntry,
  ProcessStatus,
  TrainingHealthEntry,
  TrainingMetricsRow,
} from "../../types";

export type TrainHubMode = "train" | "hpo" | "meta" | "eval";

export const PROBLEMS = TRAIN_PROBLEMS;
export const MODELS = TRAIN_MODELS;
export const ENCODERS = ["gat", "gcn", "mha"] as const;
export const HPO_METHODS = ["nsgaii", "tpe", "dehb", "random"] as const;
export const META_STRATEGIES = [
  "hrl",
  "weight_learning",
  "hypernet",
  "maml",
] as const;
export const EVAL_STRATEGIES = ["greedy", "sampling", "beam"] as const;

export function buildTrainHubHydraArgs(params: {
  mode: TrainHubMode;
  seed: number;
  wandb: boolean;
  problem: string;
  model: string;
  encoder: string;
  batchSize: number;
  epochs: number;
  hpoTrials: number;
  hpoMethod: string;
  hpoWorkers: number;
  metaStrategy: string;
  metaLr: number;
  metaHistoryLength: number;
  mrlBatchSize: number;
  mrlStep: number;
  checkpointPath: string;
  evalDataset: string;
  evalSamples: number;
  evalStrategy: string;
  extraOverrides: string;
}): string[] {
  const base = [`seed=${params.seed}`];
  if (!params.wandb) base.push("tracker.enabled=false");

  const extra = params.extraOverrides
    .split("\n")
    .map((l) => l.trim())
    .filter(Boolean);

  if (params.mode === "train") {
    return [
      ...base,
      `envs@train.env=${params.problem}`,
      `models=${params.model}`,
      `model.encoder.type=${params.encoder}`,
      `train.batch_size=${params.batchSize}`,
      `train.max_epochs=${params.epochs}`,
      `hpo.n_trials=0`,
      ...extra,
    ];
  }
  if (params.mode === "hpo") {
    return [
      ...base,
      `envs@train.env=${params.problem}`,
      `models=${params.model}`,
      `model.encoder.type=${params.encoder}`,
      `hpo.n_trials=${params.hpoTrials}`,
      `hpo.method=${params.hpoMethod}`,
      `hpo.num_workers=${params.hpoWorkers}`,
      ...extra,
    ];
  }
  if (params.mode === "meta") {
    return [
      `seed=${params.seed}`,
      ...(params.wandb ? ["tracking.wandb_mode=online"] : []),
      `envs@meta_rl.env=${params.problem}`,
      `meta_rl.env.name=${params.problem}`,
      `meta_rl.meta_strategy=${params.metaStrategy}`,
      `meta_rl.meta_lr=${params.metaLr}`,
      `meta_rl.meta_history_length=${params.metaHistoryLength}`,
      `meta_rl.mrl_batch_size=${params.mrlBatchSize}`,
      `meta_rl.mrl_step=${params.mrlStep}`,
      ...extra,
    ];
  }
  return [
    ...base,
    ...(params.checkpointPath
      ? [`eval.policy.model.load_path=${params.checkpointPath}`]
      : []),
    ...(params.evalDataset ? [`eval.datasets=[${params.evalDataset}]`] : []),
    `eval.problem=${params.problem}`,
    `eval.val_size=${params.evalSamples}`,
    `eval.decoding.strategy=${params.evalStrategy}`,
    ...extra,
  ];
}

export function LiveChart({
  metrics,
  logScale,
}: {
  metrics: TrainingMetricsRow[];
  logScale: boolean;
}) {
  const chartRef = useRef<EChartsReact | null>(null);
  const option = useMemo(() => {
    const xs = metrics.map((_, i) => i + 1);
    const scaleLoss = (v: number | null | undefined) =>
      v == null ? null : logScale ? Math.max(v, 1e-8) : v;
    const trainLoss = metrics.map((m) => scaleLoss(m.train_loss));
    const valLoss = metrics.map((m) => scaleLoss(m.val_loss));
    const reward = metrics.map((m) => m.reward ?? null);
    const hasReward = reward.some((v) => v !== null);

    return {
      backgroundColor: "transparent",
      tooltip: { trigger: "axis" as const },
      legend: {
        data: ["train_loss", "val_loss", ...(hasReward ? ["reward"] : [])],
        textStyle: { color: "#9090b0", fontSize: 10 },
        top: 0,
      },
      grid: { left: 50, right: hasReward ? 55 : 10, top: 30, bottom: 30 },
      xAxis: {
        type: "category" as const,
        data: xs,
        axisLabel: { color: "#9090b0", fontSize: 9 },
        name: "update",
        nameTextStyle: { color: "#9090b0" },
      },
      yAxis: [
        {
          type: (logScale ? "log" : "value") as "log" | "value",
          logBase: 10,
          name: logScale ? "Loss (log)" : "Loss",
          nameTextStyle: { color: "#9090b0", fontSize: 9 },
          axisLabel: { color: "#9090b0", fontSize: 9 },
          minorSplitLine: { show: false },
        },
        ...(hasReward
          ? [
              {
                type: "value" as const,
                name: "Reward",
                nameTextStyle: { color: "#9090b0", fontSize: 9 },
                axisLabel: { color: "#9090b0", fontSize: 9 },
                splitLine: { show: false },
              },
            ]
          : []),
      ],
      series: [
        {
          name: "train_loss",
          type: "line" as const,
          smooth: true,
          symbol: "none",
          color: "#6366f1",
          data: trainLoss,
        },
        {
          name: "val_loss",
          type: "line" as const,
          smooth: true,
          symbol: "none",
          lineStyle: { type: "dashed" as const },
          color: "#34d399",
          data: valLoss,
        },
        ...(hasReward
          ? [
              {
                name: "reward",
                type: "line" as const,
                smooth: true,
                symbol: "none",
                yAxisIndex: 1,
                lineStyle: { type: "dotted" as const },
                color: "#fbbf24",
                data: reward,
              },
            ]
          : []),
      ],
    };
  }, [metrics, logScale]);

  return (
    <div className="space-y-1">
      <p className="text-[10px] text-canvas-muted">
        {logScale
          ? "Log-scale loss axis — reward stays linear on right axis"
          : "Linear loss · reward on right axis when present"}
      </p>
      <div className="flex justify-end">
        <ChartExportButtons
          chartRef={{ current: chartRef.current }}
          filenameStem="training-live"
        />
      </div>
      <ReactECharts ref={chartRef} option={option} style={{ height: 200 }} />
    </div>
  );
}

interface TrainHubLiveSectionProps {
  displayProcessId: string;
  mode: TrainHubMode;
  displayProc: ProcessEntry | null;
  isDone: boolean;
  runStatus: ProcessStatus | null;
  evalLiveTitle: string;
  trainHpoLiveTitle: string;
  liveRunLabel: string | null;
  liveLogPath: string | null;
  projectRoot: string | null;
  outputRunPath: string | null;
  trainingRunPath: string | null;
  completedEvalCheckpointPath: string | null;
  evalResult: EvalResult | null;
  openEvalInAnalytics: () => void;
  liveLogLines: string[];
  evalCheckpointName: string;
  checkpointPath: string;
  liveMetrics: TrainingMetricsRow[];
  liveHealth: TrainingHealthEntry[];
  liveAttention: AttentionVizEntry[];
  showTrainingAnalytics: boolean;
  epochs: number;
  latestMetric: TrainingMetricsRow | undefined;
  logScale: boolean;
  effectiveTheme: "dark" | "light";
}

export function TrainHubLiveSection({
  displayProcessId,
  mode,
  displayProc,
  isDone,
  runStatus,
  evalLiveTitle,
  trainHpoLiveTitle,
  liveRunLabel,
  liveLogPath,
  projectRoot,
  outputRunPath,
  trainingRunPath,
  completedEvalCheckpointPath,
  evalResult,
  openEvalInAnalytics,
  liveLogLines,
  evalCheckpointName,
  checkpointPath,
  liveMetrics,
  liveHealth,
  liveAttention,
  showTrainingAnalytics,
  epochs,
  latestMetric,
  logScale,
  effectiveTheme,
}: TrainHubLiveSectionProps) {
  return (
    <>
      <GlobalFilterBar
        showLogScale
        runLabels={liveRunLabel ? [liveRunLabel] : []}
      />

      {mode === "eval" && displayProc && (
        <LauncherLivePanel
          header={{
            status: isDone ? (runStatus ?? "running") : "running",
            title: evalLiveTitle,
            runLabel: liveRunLabel,
            logPath: liveLogPath,
            projectRoot,
            navMesh: {
              kind: "eval",
              hideSelf: true,
              hideHub: true,
              showPostRun: isDone && runStatus === "completed",
              showOutputBrowser: isDone && runStatus === "completed",
              outputRunPath,
              checkpointPath: completedEvalCheckpointPath,
              onOpenAnalytics:
                evalResult && hasEvalMetrics(evalResult)
                  ? openEvalInAnalytics
                  : undefined,
            },
          }}
          footer={
            <ProcessIdFooter
              processId={displayProcessId}
              logPath={liveLogPath}
              runLabel={liveRunLabel}
              projectRoot={projectRoot}
            />
          }
          logLines={liveLogLines}
          logTailWaiting={!isDone}
        >
          {!isDone || !evalResult || !hasEvalMetrics(evalResult) ? (
            <EvalCheckpointLiveCard
              procId={displayProcessId}
              checkpointName={evalCheckpointName}
              checkpointPath={
                completedEvalCheckpointPath ?? (checkpointPath.trim() || null)
              }
              projectRoot={projectRoot}
              status={displayProc.status}
              isRunning={!isDone}
              result={evalResult ?? undefined}
              logLines={liveLogLines}
              showLogTail={false}
            />
          ) : (
            <EvalResultCard
              result={evalResult}
              projectRoot={projectRoot}
              onOpenAnalytics={openEvalInAnalytics}
            />
          )}
        </LauncherLivePanel>
      )}

      {mode !== "eval" && (
        <TrainHpoLivePanel
          header={{
            status: isDone ? (runStatus ?? "running") : "running",
            title: trainHpoLiveTitle,
            runLabel: liveRunLabel,
            logPath: liveLogPath,
            projectRoot,
            showLiveSuffix: !isDone,
            metricCount: liveMetrics.length,
            healthCount: liveHealth.length,
            attentionCount: liveAttention.length,
            layout: "split",
            runningIcon: "activity",
            titleTone: "heading",
            navMesh: {
              hideHub: true,
              showTrainLinks: showTrainingAnalytics,
              showHpoLinks: mode === "hpo",
              showOutputBrowser: isDone && runStatus === "completed",
              outputRunPath,
              trainingRunPath,
            },
          }}
          progress={
            !isDone && displayProcessId
              ? {
                  processId: displayProcessId,
                  fallbackTotal: mode === "train" ? epochs : undefined,
                  fallbackValue: latestMetric?.epoch,
                }
              : undefined
          }
          showAnalytics={showTrainingAnalytics}
          analyticsWrapperClassName={isDone ? undefined : "pt-0"}
          analytics={{
            metrics: liveMetrics,
            healthEntries: liveHealth,
            attentionEntries: liveAttention,
            logScale,
            theme: effectiveTheme,
            exportNamePrefix: "training-hub",
            isPostRun: isDone,
            postRunFallback:
              "Post-run shortcuts — open Training Monitor or Output Browser for this run",
            middleContent: (
              <>
                {liveMetrics.length >= 2 && <GlobalFilterBar showLogScale />}
                {liveMetrics.length >= 2 ? (
                  <LiveChart metrics={liveMetrics} logScale={logScale} />
                ) : (
                  <p className="text-xs text-canvas-muted">
                    {isDone
                      ? liveMetrics.length === 0
                        ? "No JSON metric lines detected in stdout."
                        : "Only one metric update received."
                      : "Waiting for metric JSON lines on stdout…"}
                  </p>
                )}
              </>
            ),
          }}
          logLines={liveLogLines}
          logTailWaiting={!isDone}
          footer={
            <ProcessIdFooter
              processId={displayProcessId}
              logPath={liveLogPath}
              runLabel={liveRunLabel}
              projectRoot={projectRoot}
            />
          }
        />
      )}
    </>
  );
}
