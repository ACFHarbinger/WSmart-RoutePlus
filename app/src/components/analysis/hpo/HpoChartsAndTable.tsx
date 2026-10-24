import { useMemo, useRef } from "react";
import ReactECharts from "echarts-for-react";
import { ChartExportButtons } from "../../common/ChartExportButtons";
import { OpenPathToolbar } from "../../common/OpenPathToolbar";
import { OrionHoneycombMatrix } from "../../ui/orion/OrionHoneycombMatrix";
import { buildHpoHoneycomb } from "./hpoHoneycomb";
import { resolveLocalProjectPath } from "../../../utils/runs/outputRunPath";
import { trialLogDirFromUserAttrs } from "../../../utils/training/trialRunPath";
import type { OptunaStudyData, OptunaTrial } from "../../../types";

const COMPARE_COLORS = ["#6366f1", "#f87171"];

export function buildCrossStudyHistoryOption(
  studies: Array<{ name: string; trials: OptunaStudyData["trials"] }>,
  logScale = false
) {
  const series = studies.map((s, i) => {
    const completed = s.trials.filter(
      (t) => t.state === "COMPLETE" && t.value != null
    );
    let best = Infinity;
    const bestSoFar = completed.map((t) => {
      best = Math.min(best, t.value as number);
      return logScale ? Math.max(best, 1e-8) : best;
    });
    return {
      name: s.name,
      type: "line",
      data: bestSoFar,
      lineStyle: { color: COMPARE_COLORS[i % COMPARE_COLORS.length], width: 2 },
      showSymbol: false,
    };
  });
  const maxLen = Math.max(...series.map((s) => s.data.length), 1);
  return {
    backgroundColor: "transparent",
    grid: { left: 50, right: 10, top: 30, bottom: 40 },
    xAxis: {
      type: "category",
      data: Array.from({ length: maxLen }, (_, i) => i + 1),
      name: "Trial #",
      axisLabel: { color: "#9090b0", fontSize: 10 },
    },
    yAxis: {
      type: logScale ? "log" : "value",
      logBase: 10,
      name: logScale ? "Best objective (log)" : "Best objective",
      axisLabel: { color: "#9090b0", fontSize: 10 },
      minorSplitLine: { show: false },
    },
    legend: { textStyle: { color: "#9090b0", fontSize: 11 } },
    series,
    tooltip: { trigger: "axis" },
  };
}

export function buildHistoryOption(
  trials: OptunaStudyData["trials"],
  logScale = false
) {
  const completed = trials.filter(
    (t) => t.state === "COMPLETE" && t.value != null
  );
  const numbers = completed.map((t) => t.number);
  const values = completed.map((t) => t.value as number);
  const displayValue = (v: number) => (logScale ? Math.max(v, 1e-8) : v);
  let best = Infinity;
  const bestSoFar = values.map((v) => {
    best = Math.min(best, v);
    return displayValue(best);
  });

  return {
    backgroundColor: "transparent",
    grid: { left: 50, right: 10, top: 20, bottom: 40 },
    xAxis: {
      type: "category",
      data: numbers,
      name: "Trial",
      nameTextStyle: { color: "#9090b0" },
      axisLabel: { color: "#9090b0", fontSize: 10 },
    },
    yAxis: {
      type: logScale ? "log" : "value",
      logBase: 10,
      name: logScale ? "Objective (log)" : "Objective",
      nameTextStyle: { color: "#9090b0" },
      axisLabel: { color: "#9090b0", fontSize: 10 },
      minorSplitLine: { show: false },
    },
    series: [
      {
        name: "Objective",
        type: "scatter",
        data: values.map(displayValue),
        symbolSize: 7,
        itemStyle: { color: "#6366f1" },
      },
      {
        name: "Best so far",
        type: "line",
        data: bestSoFar,
        lineStyle: { color: "#34d399", width: 2 },
        showSymbol: false,
      },
    ],
    legend: { textStyle: { color: "#9090b0", fontSize: 11 } },
    tooltip: { trigger: "axis" },
  };
}

export function buildImportanceOption(importances: Record<string, number>) {
  const sorted = Object.entries(importances).sort((a, b) => b[1] - a[1]);
  return {
    backgroundColor: "transparent",
    grid: { left: 120, right: 20, top: 10, bottom: 30 },
    xAxis: {
      type: "value",
      axisLabel: { color: "#9090b0", fontSize: 10 },
    },
    yAxis: {
      type: "category",
      data: sorted.map(([k]) => k),
      axisLabel: { color: "#9090b0", fontSize: 10 },
    },
    series: [
      {
        type: "bar",
        data: sorted.map(([, v]) => v),
        itemStyle: { color: "#818cf8" },
      },
    ],
    tooltip: { trigger: "axis" },
  };
}

export function buildParallelOption(study: OptunaStudyData, logScale = false) {
  const completed = study.trials.filter(
    (t) => t.state === "COMPLETE" && t.value != null
  );
  if (completed.length === 0) return null;

  const paramKeys = [
    ...new Set(completed.flatMap((t) => Object.keys(t.params))),
  ].slice(0, 8);

  const objectiveLabel = logScale ? "objective (log)" : "objective";
  const schema = [
    { dim: 0, name: objectiveLabel, type: "value" },
    ...paramKeys.map((k, i) => ({
      dim: i + 1,
      name: k,
      type: "value" as const,
    })),
  ];

  const displayObjective = (v: number) => (logScale ? Math.max(v, 1e-8) : v);
  const data = completed.map((t) => [
    displayObjective(t.value as number),
    ...paramKeys.map((k) => {
      const v = t.params[k];
      return typeof v === "number" ? v : 0;
    }),
  ]);

  return {
    backgroundColor: "transparent",
    parallelAxis: schema.map((s, i) => ({
      dim: i,
      name: s.name,
      nameTextStyle: { color: "#9090b0", fontSize: 10 },
      axisLine: { lineStyle: { color: "#2d2d50" } },
    })),
    series: [
      {
        type: "parallel",
        lineStyle: { width: 1, opacity: 0.5, color: "#6366f1" },
        data,
      },
    ],
    tooltip: { trigger: "item" },
  };
}

export function HpoStudyVisualsSection({
  studyData,
  compareStudyData,
  historyOption,
  crossStudyOption,
  parallelOption,
  logScale,
  projectRoot,
  activeRunLabel,
  selectedTrialNumbers,
  toggleTrial,
  trialBrushLabel,
}: {
  studyData: OptunaStudyData;
  compareStudyData: OptunaStudyData | null;
  historyOption: object | null;
  crossStudyOption: object | null;
  parallelOption: object | null;
  logScale: boolean;
  projectRoot: string;
  activeRunLabel: string | null;
  selectedTrialNumbers: number[];
  toggleTrial: (num: number) => void;
  trialBrushLabel: (trial: OptunaTrial) => string | null;
}) {
  const historyChartRef = useRef<ReactECharts>(null);
  const importanceChartRef = useRef<ReactECharts>(null);
  const crossStudyChartRef = useRef<ReactECharts>(null);
  const parallelChartRef = useRef<ReactECharts>(null);

  const completedTrials = useMemo(
    () =>
      studyData.trials.filter((t) => t.state === "COMPLETE" && t.value != null),
    [studyData.trials]
  );

  const honeycombConfig = useMemo(
    () => buildHpoHoneycomb(completedTrials, studyData.importances),
    [completedTrials, studyData.importances]
  );

  return (
    <>
      {completedTrials.length >= 2 && (
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
          <div className="card">
            <div className="flex items-center justify-between mb-2">
              <p className="text-xs text-canvas-muted">Optimisation History</p>
              <ChartExportButtons
                chartRef={historyChartRef}
                filenameStem="hpo-history"
                size={11}
              />
            </div>
            <p className="text-[10px] text-canvas-muted mb-1">
              {logScale
                ? "Log-scale objective axis — trial scatter + best-so-far line"
                : "Linear objective — trial scatter + best-so-far line"}
            </p>
            <ReactECharts
              ref={historyChartRef}
              option={historyOption ?? {}}
              style={{ height: 260 }}
            />
          </div>
          {Object.keys(studyData.importances).length > 0 && (
            <div className="card">
              <div className="flex items-center justify-between mb-2">
                <p className="text-xs text-canvas-muted">
                  Parameter Importance (FANOVA)
                </p>
                <ChartExportButtons
                  chartRef={importanceChartRef}
                  filenameStem="hpo-importance"
                  size={11}
                />
              </div>
              <ReactECharts
                ref={importanceChartRef}
                option={buildImportanceOption(studyData.importances)}
                style={{ height: 260 }}
              />
            </div>
          )}
        </div>
      )}

      {honeycombConfig && (
        <OrionHoneycombMatrix
          title={`2D Hyperparameter Sweep Honeycomb (${honeycombConfig.xKey} × ${honeycombConfig.yKey})`}
          subtitle="Each cell is one pair of hyperparameter values. When several trials share a pair, the cell keeps the lower objective."
          xAxisLabel={honeycombConfig.xKey}
          yAxisLabel={honeycombConfig.yKey}
          cells={honeycombConfig.cells}
          higherIsBetter={false}
        />
      )}

      {crossStudyOption && (
        <div className="card">
          <div className="flex items-center justify-between mb-2">
            <p className="text-xs text-canvas-muted">
              Cross-Study Comparison — Best-So-Far
            </p>
            <ChartExportButtons
              chartRef={crossStudyChartRef}
              filenameStem="hpo-cross-study"
              size={11}
            />
          </div>
          <ReactECharts
            ref={crossStudyChartRef}
            option={crossStudyOption}
            style={{ height: 280 }}
          />
          {compareStudyData && (
            <div className="grid grid-cols-2 gap-3 mt-3 text-xs">
              <div className="kpi-card">
                <p className="text-canvas-muted">{studyData.name}</p>
                <p className="font-mono text-accent-success tabular-nums">
                  {studyData.best_value != null
                    ? studyData.best_value.toFixed(6)
                    : "—"}
                </p>
              </div>
              <div className="kpi-card">
                <p className="text-canvas-muted">{compareStudyData.name}</p>
                <p className="font-mono text-accent-danger tabular-nums">
                  {compareStudyData.best_value != null
                    ? compareStudyData.best_value.toFixed(6)
                    : "—"}
                </p>
              </div>
            </div>
          )}
        </div>
      )}

      {parallelOption && (
        <div className="card">
          <div className="flex items-center justify-between mb-2">
            <p className="text-xs text-canvas-muted">Parallel Coordinates</p>
            <ChartExportButtons
              chartRef={parallelChartRef}
              filenameStem="hpo-parallel"
              size={11}
            />
          </div>
          <p className="text-[10px] text-canvas-muted mb-1">
            {logScale
              ? "Log-scale objective axis — hyperparameter polylines"
              : "Linear objective — hyperparameter polylines"}
          </p>
          <ReactECharts
            ref={parallelChartRef}
            option={parallelOption}
            style={{ height: 320 }}
          />
        </div>
      )}

      {studyData.trials.length > 0 && (
        <div className="card space-y-2">
          <p className="text-xs text-canvas-muted">Trial Health (§A.4)</p>
          <p className="text-[10px] text-canvas-muted">
            Gradient norm and entropy captured per trial via HPO health metrics —
            unhealthy trials are pruned early when thresholds are exceeded.
          </p>
          <div className="overflow-x-auto">
            <table className="w-full text-xs">
              <thead>
                <tr className="text-left text-canvas-muted border-b border-canvas-border">
                  <th className="py-2 pr-3 w-8" />
                  <th className="py-2 pr-3">Trial</th>
                  <th className="py-2 pr-3">State</th>
                  <th className="py-2 pr-3">Objective</th>
                  <th className="py-2 pr-3">Grad Norm</th>
                  <th className="py-2 pr-3">Entropy</th>
                  <th className="py-2">Health</th>
                </tr>
              </thead>
              <tbody>
                {studyData.trials.map((trial) => {
                  const attrs = trial.user_attrs ?? {};
                  const rawTrialLogDir = trialLogDirFromUserAttrs(attrs);
                  const trialLogDir = rawTrialLogDir
                    ? resolveLocalProjectPath(rawTrialLogDir, projectRoot) ??
                      rawTrialLogDir
                    : null;
                  const brushLabel = trialBrushLabel(trial);
                  const brushActive = Boolean(
                    brushLabel && activeRunLabel && activeRunLabel === brushLabel
                  );
                  const gradNorm = attrs.last_grad_norm;
                  const entropy = attrs.last_entropy;
                  const healthPruned = attrs.health_pruned;
                  const gradHigh =
                    typeof gradNorm === "number" && gradNorm > 100;
                  const entropyLow =
                    typeof entropy === "number" && entropy < 0.01;
                  return (
                    <tr
                      key={trial.number}
                      className={`border-b border-canvas-border/50 text-gray-200 hover:bg-canvas-hover cursor-pointer ${
                        selectedTrialNumbers.includes(trial.number) ||
                        brushActive
                          ? "bg-canvas-hover/60"
                          : ""
                      } ${
                        brushActive
                          ? "ring-1 ring-inset ring-accent-secondary/30"
                          : ""
                      }`}
                      onClick={() => toggleTrial(trial.number)}
                    >
                      <td className="py-1.5 pr-3">
                        <input
                          type="checkbox"
                          className="accent-accent-primary"
                          checked={selectedTrialNumbers.includes(trial.number)}
                          onChange={() => toggleTrial(trial.number)}
                          onClick={(e) => e.stopPropagation()}
                        />
                      </td>
                      <td className="py-1.5 pr-3">
                        {trialLogDir ? (
                          <OpenPathToolbar
                            path={trialLogDir}
                            kind="training"
                            brushLabel={brushLabel ?? undefined}
                            storedLabel={brushLabel ?? undefined}
                            chipClassName="flex-1 min-w-0 max-w-none"
                            className="flex-1 min-w-0"
                          >
                            <span className="text-canvas-muted font-mono text-[10px] shrink-0">
                              #{trial.number}
                            </span>
                          </OpenPathToolbar>
                        ) : (
                          <span className="font-mono">#{trial.number}</span>
                        )}
                      </td>
                      <td className="py-1.5 pr-3">{trial.state}</td>
                      <td className="py-1.5 pr-3 font-mono tabular-nums">
                        {trial.value != null ? trial.value.toFixed(4) : "—"}
                      </td>
                      <td
                        className={`py-1.5 pr-3 font-mono tabular-nums ${
                          gradHigh ? "text-accent-danger" : ""
                        }`}
                      >
                        {typeof gradNorm === "number"
                          ? gradNorm.toFixed(3)
                          : "—"}
                      </td>
                      <td
                        className={`py-1.5 pr-3 font-mono tabular-nums ${
                          entropyLow ? "text-accent-warning" : ""
                        }`}
                      >
                        {typeof entropy === "number" ? entropy.toFixed(4) : "—"}
                      </td>
                      <td className="py-1.5">
                        {typeof healthPruned === "string" ? (
                          <span className="text-accent-danger">
                            {healthPruned}
                          </span>
                        ) : gradHigh || entropyLow ? (
                          <span className="text-accent-warning">unhealthy</span>
                        ) : (
                          <span className="text-canvas-muted">ok</span>
                        )}
                      </td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
          </div>
        </div>
      )}
    </>
  );
}
