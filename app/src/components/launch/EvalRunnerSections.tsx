import { useRef } from "react";
import ReactECharts from "echarts-for-react";
import type EChartsReact from "echarts-for-react";
import { BarChart3, Download, FolderOpen, Trash2 } from "lucide-react";
import { GlobalFilterBar } from "../layout/GlobalFilterBar";
import { ChartExportButtons } from "../common/ChartExportButtons";
import { OpenPathToolbar } from "../common/OpenPathToolbar";
import { EvalCheckpointLiveCard } from "../monitor/eval/EvalCheckpointLiveCard";
import { LauncherLivePanel } from "../monitor/live/LauncherLivePanel";
import { ProcessIdFooter } from "../monitor/process/ProcessIdFooter";
import { parentRunBrushLabelFromCheckpointPath } from "../../utils/training/checkpoints";
import {
  checkpointLabelFromEvalProcess,
  checkpointPathFromEvalCommand,
  evalLivePanelTitle,
  type EvalResult,
} from "../../utils/benchmark/evalResults";
import type { ProcessEntry, ProcessStatus } from "../../types";

export interface CheckpointEntry {
  id: string;
  path: string;
}

export const EVAL_CHART_METRICS = [
  { key: "cost", label: "Tour Cost" },
  { key: "gap", label: "Optimality Gap (%)" },
  { key: "time", label: "Time (s)" },
] as const;

export const EVAL_COLORS = [
  "#6366f1",
  "#34d399",
  "#fbbf24",
  "#f87171",
  "#818cf8",
  "#a3e635",
];

export function ResultsGrid({
  results,
  logScale,
  runLabels,
  projectRoot,
  onOpenAnalytics,
}: {
  results: EvalResult[];
  logScale: boolean;
  runLabels?: string[];
  projectRoot?: string | null;
  onOpenAnalytics: () => void;
}) {
  const chartRefs = useRef<Record<string, EChartsReact | null>>({});
  const allKeys = Array.from(
    new Set(
      results.flatMap((r) =>
        Object.keys(r).filter((k) => k !== "checkpointName")
      )
    )
  );
  const numKeys = allKeys.filter((k) =>
    results.some((r) => typeof r[k] === "number")
  );
  const checkpoints = results.map((r) => r.checkpointName);

  const makeBarOption = (metricKey: string, metricLabel: string) => ({
    backgroundColor: "transparent",
    grid: { left: 50, right: 10, top: 20, bottom: 55 },
    xAxis: {
      type: "category",
      data: checkpoints,
      axisLabel: { color: "#9090b0", fontSize: 9, rotate: 25 },
    },
    yAxis: {
      type: (logScale ? "log" : "value") as "log" | "value",
      logBase: 10,
      name: metricLabel,
      nameTextStyle: { color: "#9090b0" },
      axisLabel: { color: "#9090b0", fontSize: 10 },
      minorSplitLine: { show: false },
    },
    series: [
      {
        type: "bar",
        data: results.map((r, i) => {
          const raw = (r[metricKey] as number | undefined) ?? 0;
          return {
            value: logScale ? Math.max(raw, 0.001) : raw,
            itemStyle: { color: EVAL_COLORS[i % EVAL_COLORS.length] },
          };
        }),
      },
    ],
    tooltip: { trigger: "axis" },
  });

  const exportCsv = () => {
    const cols = ["checkpoint", ...numKeys];
    const rows = results.map((r) =>
      cols
        .map((c) => (c === "checkpoint" ? r.checkpointName : String(r[c] ?? "")))
        .join(",")
    );
    const csv = [cols.join(","), ...rows].join("\n");
    const blob = new Blob([csv], { type: "text/csv" });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = "eval_results.csv";
    a.click();
    URL.revokeObjectURL(url);
  };

  return (
    <div className="space-y-3">
      <GlobalFilterBar showLogScale runLabels={runLabels} />

      <div className="card space-y-3">
        <div className="flex items-center justify-between">
          <h2 className="text-sm font-semibold text-gray-200">
            Results ({results.length})
          </h2>
          <div className="flex items-center gap-2">
            <button
              onClick={onOpenAnalytics}
              className="btn-ghost text-xs flex items-center gap-1"
            >
              <BarChart3 size={12} />
              Open in Analytics →
            </button>
            <button
              onClick={exportCsv}
              className="btn-ghost text-xs flex items-center gap-1"
            >
              <Download size={12} />
              Export CSV
            </button>
          </div>
        </div>
        <p className="text-[10px] text-canvas-muted">
          {logScale
            ? "Log-scale bars — tour cost · optimality gap · eval time per checkpoint"
            : "Linear bars — tour cost · optimality gap · eval time per checkpoint"}
        </p>
        <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-3">
          {EVAL_CHART_METRICS.filter(({ key }) => numKeys.includes(key)).map(
            ({ key, label }) => (
              <div
                key={key}
                className="rounded-lg border border-canvas-border/40 p-2"
              >
                <div className="flex items-center justify-between mb-1">
                  <p className="text-xs text-canvas-muted">{label}</p>
                  <ChartExportButtons
                    chartRef={{ current: chartRefs.current[key] }}
                    filenameStem={`eval-runner-${key}`}
                    size={10}
                  />
                </div>
                <ReactECharts
                  ref={(el) => {
                    chartRefs.current[key] = el;
                  }}
                  option={makeBarOption(key, label)}
                  style={{ height: 180 }}
                />
              </div>
            )
          )}
        </div>

        <div className="overflow-x-auto">
          <table className="w-full text-xs">
            <thead>
              <tr className="border-b border-canvas-border">
                <th className="text-left py-1.5 pr-4 text-canvas-muted font-medium">
                  Checkpoint
                </th>
                {numKeys.map((k) => (
                  <th
                    key={k}
                    className="text-right py-1.5 px-3 text-canvas-muted font-medium"
                  >
                    {k}
                  </th>
                ))}
              </tr>
            </thead>
            <tbody className="divide-y divide-canvas-border/30">
              {results.map((r) => (
                <tr key={r.checkpointName} className="hover:bg-canvas-hover/40">
                  <td className="py-1.5 pr-4 max-w-[240px]">
                    {r.checkpointPath ? (
                      <OpenPathToolbar
                        path={r.checkpointPath}
                        projectRoot={projectRoot}
                        kind="checkpoint"
                        label={r.checkpointName}
                        brushLabel={parentRunBrushLabelFromCheckpointPath(
                          r.checkpointPath,
                          projectRoot
                        )}
                        chipClassName="max-w-full"
                        className="max-w-full min-w-0"
                      />
                    ) : (
                      <span className="font-mono text-gray-300 truncate">
                        {r.checkpointName}
                      </span>
                    )}
                  </td>
                  {numKeys.map((k) => (
                    <td
                      key={k}
                      className="py-1.5 px-3 text-right font-mono text-gray-400"
                    >
                      {typeof r[k] === "number"
                        ? (r[k] as number).toFixed(4)
                        : "—"}
                    </td>
                  ))}
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>
    </div>
  );
}

export function CheckpointRow({
  entry,
  projectRoot,
  onRemove,
  onPickFile,
  onChange,
}: {
  entry: CheckpointEntry;
  projectRoot?: string | null;
  onRemove: () => void;
  onPickFile: () => void;
  onChange: (path: string) => void;
}) {
  const trimmedPath = entry.path.trim();

  return (
    <div className="space-y-1">
      <div className="flex items-center gap-2">
        <input
          type="text"
          className="input-base font-mono text-xs flex-1"
          value={entry.path}
          onChange={(e) => onChange(e.target.value)}
          placeholder="path/to/checkpoint.pt"
        />
        <button
          onClick={onPickFile}
          className="btn-ghost p-1.5 text-canvas-muted hover:text-gray-200"
          title="Browse"
        >
          <FolderOpen size={13} />
        </button>
        <button
          onClick={onRemove}
          className="btn-ghost p-1.5 text-accent-danger/60 hover:text-accent-danger"
          title="Remove"
        >
          <Trash2 size={13} />
        </button>
      </div>
      {trimmedPath ? (
        <OpenPathToolbar
          path={trimmedPath}
          projectRoot={projectRoot}
          kind="checkpoint"
          brushLabel={parentRunBrushLabelFromCheckpointPath(
            trimmedPath,
            projectRoot
          )}
          chipClassName="max-w-full"
        />
      ) : null}
    </div>
  );
}

export function EvalRunnerLiveSection({
  liveRunSummary,
  displayProcessIds,
  processes,
  results,
  liveRunLabel,
  liveLogPath,
  projectRoot,
  outputRunPath,
  completedCheckpointPath,
  openInAnalytics,
  singleCheckpointEval,
  singleEvalLogLines,
  singleEvalLogTailWaiting,
}: {
  liveRunSummary: {
    running: number;
    completed: number;
    failed: number;
    allDone: boolean;
    aggregate: ProcessStatus | "running";
  };
  displayProcessIds: string[];
  processes: Record<string, ProcessEntry>;
  results: EvalResult[];
  liveRunLabel: string | null;
  liveLogPath: string | null;
  projectRoot: string | null;
  outputRunPath: string | null;
  completedCheckpointPath: string | null;
  openInAnalytics: () => void;
  singleCheckpointEval: boolean;
  singleEvalLogLines: string[];
  singleEvalLogTailWaiting: boolean;
}) {
  return (
    <LauncherLivePanel
      header={{
        status: liveRunSummary.allDone ? liveRunSummary.aggregate : "running",
        title: evalLivePanelTitle({
          isRunning: !liveRunSummary.allDone,
          status: liveRunSummary.aggregate,
          total: displayProcessIds.length,
          runningCount: liveRunSummary.running,
          completedCount: liveRunSummary.completed,
        }),
        runLabel: liveRunLabel,
        logPath: liveLogPath,
        projectRoot,
        navMesh: {
          kind: "eval",
          hideSelf: true,
          showPostRun:
            liveRunSummary.allDone && liveRunSummary.aggregate === "completed",
          showOutputBrowser:
            liveRunSummary.allDone && liveRunSummary.aggregate === "completed",
          outputRunPath,
          checkpointPath: completedCheckpointPath,
          onOpenAnalytics: results.length > 0 ? openInAnalytics : undefined,
        },
      }}
      footer={
        <ProcessIdFooter
          processIds={displayProcessIds}
          logPath={liveLogPath}
          runLabel={liveRunLabel}
          projectRoot={projectRoot}
        />
      }
      logLines={singleCheckpointEval ? singleEvalLogLines : undefined}
      logTailWaiting={singleCheckpointEval ? singleEvalLogTailWaiting : false}
    >
      <div className="space-y-2">
        {displayProcessIds.map((procId) => {
          const proc = processes[procId];
          const ckptName = checkpointLabelFromEvalProcess(
            procId,
            proc?.command ?? ""
          );
          const isRunning = proc?.status === "running";
          const ckptResult = results.find((r) => r.checkpointName === ckptName);
          const ckptPath = checkpointPathFromEvalCommand(proc?.command ?? "");

          return (
            <EvalCheckpointLiveCard
              key={procId}
              procId={procId}
              checkpointName={ckptName}
              checkpointPath={ckptPath}
              projectRoot={projectRoot}
              status={proc?.status}
              isRunning={isRunning}
              result={ckptResult}
              logLines={proc?.logLines ?? []}
              showLogTail={!singleCheckpointEval}
            />
          );
        })}
      </div>
    </LauncherLivePanel>
  );
}
