import { useRef } from "react";
import ReactECharts from "echarts-for-react";
import type EChartsReact from "echarts-for-react";
import { X } from "lucide-react";
import { ChartExportButtons } from "../../common/ChartExportButtons";
import { OpenPathToolbar } from "../../common/OpenPathToolbar";
import { barOpacity } from "../../../utils/charts/chartHighlight";
import {
  errorBarBounds,
  groupedBarWhiskerX,
} from "../../../utils/charts/chartLogScale";
import { symlog } from "../../../utils/charts/symlog";
import { parentRunBrushLabelFromCheckpointPath } from "../../../utils/training/checkpoints";
import type { DayLogEntry, EvalAnalyticsRow } from "../../../types";

export interface RunFile {
  path: string;
  label: string;
  entries: DayLogEntry[];
}

export function mean(arr: number[]) {
  return arr.length ? arr.reduce((a, b) => a + b, 0) / arr.length : 0;
}

export function std(arr: number[]) {
  if (arr.length < 2) return 0;
  const m = mean(arr);
  return Math.sqrt(
    arr.reduce((s, x) => s + (x - m) ** 2, 0) / (arr.length - 1)
  );
}

export function fmt(n: number, d = 2) {
  return Number.isFinite(n) ? n.toFixed(d) : "—";
}

export const SIM_METRICS = [
  { key: "profit", label: "Profit (€)", lowerIsBetter: false },
  { key: "km", label: "Distance (km)", lowerIsBetter: true },
  { key: "overflows", label: "Overflows", lowerIsBetter: true },
  { key: "kg/km", label: "Efficiency (kg/km)", lowerIsBetter: false },
  { key: "cost", label: "Cost (€)", lowerIsBetter: true },
];

export const EVAL_METRICS = [
  { key: "cost", label: "Tour Cost", lowerIsBetter: true },
  { key: "gap", label: "Optimality Gap (%)", lowerIsBetter: true },
  { key: "time", label: "Time (s)", lowerIsBetter: true },
];

export const COLORS = [
  "#6366f1",
  "#34d399",
  "#fbbf24",
  "#f87171",
  "#818cf8",
  "#a3e635",
];

export function EvalResultsPanel({
  rows,
  logScale,
  projectRoot,
  onDismiss,
}: {
  rows: EvalAnalyticsRow[];
  logScale: boolean;
  projectRoot?: string | null;
  onDismiss: () => void;
}) {
  const chartRefs = useRef<Record<string, EChartsReact | null>>({});
  const checkpoints = rows.map((r) => r.checkpoint);

  const makeBarOption = (metricKey: string, metricLabel: string) => ({
    backgroundColor: "transparent",
    grid: { left: 50, right: 10, top: 20, bottom: 55 },
    xAxis: {
      type: "category",
      data: checkpoints,
      axisLabel: { color: "#9090b0", fontSize: 9, rotate: 25 },
    },
    yAxis: {
      type: logScale ? "log" : "value",
      logBase: 10,
      name: metricLabel,
      nameTextStyle: { color: "#9090b0" },
      axisLabel: { color: "#9090b0", fontSize: 10 },
      minorSplitLine: { show: false },
    },
    series: [
      {
        type: "bar",
        data: rows.map((r) => {
          const v = (r[metricKey] as number | undefined) ?? 0;
          return logScale ? Math.max(v, 0.001) : v;
        }),
        itemStyle: {
          color: (params: { dataIndex: number }) =>
            COLORS[params.dataIndex % COLORS.length],
        },
      },
    ],
    tooltip: { trigger: "axis" },
  });

  return (
    <div className="space-y-4">
      <div className="card flex items-center justify-between">
        <div>
          <p className="text-sm font-semibold text-gray-200">
            Eval Results — Checkpoint Comparison
          </p>
          <p className="text-xs text-canvas-muted mt-0.5">
            {rows.length} checkpoint(s) loaded from Evaluation Runner
            {logScale ? " · log-scale bars" : " · linear bars"}
          </p>
        </div>
        <button
          onClick={onDismiss}
          className="btn-ghost text-xs flex items-center gap-1"
        >
          <X size={12} /> Dismiss
        </button>
      </div>

      <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-4">
        {EVAL_METRICS.map(({ key, label }) => (
          <div key={key} className="card">
            <div className="flex items-center justify-between mb-2">
              <p className="text-xs text-canvas-muted">{label}</p>
              <ChartExportButtons
                chartRef={{ current: chartRefs.current[key] }}
                filenameStem={`eval-${key}`}
              />
            </div>
            <ReactECharts
              ref={(el) => {
                chartRefs.current[key] = el;
              }}
              option={makeBarOption(key, label)}
              style={{ height: 220 }}
            />
          </div>
        ))}
      </div>

      <div className="card overflow-x-auto">
        <table className="w-full text-xs">
          <thead>
            <tr className="border-b border-canvas-border">
              <th className="text-left py-2 px-3 text-canvas-muted font-medium">
                Checkpoint
              </th>
              {EVAL_METRICS.map(({ key, label }) => (
                <th
                  key={key}
                  className="text-right py-2 px-3 text-canvas-muted font-medium"
                >
                  {label}
                </th>
              ))}
            </tr>
          </thead>
          <tbody className="divide-y divide-canvas-border/30">
            {rows.map((r) => (
              <tr key={r.checkpoint} className="hover:bg-canvas-hover/40">
                <td className="py-1.5 px-3 max-w-[240px]">
                  {r.checkpointPath ? (
                    <OpenPathToolbar
                      path={r.checkpointPath}
                      projectRoot={projectRoot}
                      kind="checkpoint"
                      label={r.checkpoint}
                      brushLabel={parentRunBrushLabelFromCheckpointPath(
                        r.checkpointPath,
                        projectRoot
                      )}
                      chipClassName="max-w-full"
                      className="max-w-full min-w-0"
                    />
                  ) : (
                    <span className="font-mono text-gray-300">
                      {r.checkpoint}
                    </span>
                  )}
                </td>
                {EVAL_METRICS.map(({ key }) => (
                  <td
                    key={key}
                    className="py-1.5 px-3 text-right font-mono text-gray-400 tabular-nums"
                  >
                    {typeof r[key] === "number"
                      ? (r[key] as number).toFixed(4)
                      : "—"}
                  </td>
                ))}
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
}

export function buildEfficiencyRankOption(
  efficiencyRanking: Array<{ policy: string; mean: number; std: number }>,
  brushedPolicies: string[] | null,
  logScale: boolean,
  showErrorBars: boolean
) {
  return {
    backgroundColor: "transparent",
    grid: { left: 110, right: 24, top: 12, bottom: 12 },
    xAxis: {
      type: (logScale ? "log" : "value") as "log" | "value",
      logBase: 10,
      name: "kg/km",
      nameTextStyle: { color: "#9090b0", fontSize: 9 },
      axisLabel: { color: "#9090b0", fontSize: 9 },
      minorSplitLine: { show: false },
    },
    yAxis: {
      type: "category",
      data: efficiencyRanking.map((r) => r.policy),
      inverse: true,
      axisLabel: { color: "#9090b0", fontSize: 9 },
    },
    tooltip: {
      trigger: "axis",
      formatter: (params: unknown[]) => {
        const p = (params as Array<{ dataIndex: number; name: string }>)[0];
        const r = efficiencyRanking[p.dataIndex];
        return `${p.name}<br/>${fmt(r.mean)} ± ${fmt(r.std)} kg/km`;
      },
    },
    series: [
      {
        type: "bar",
        data: efficiencyRanking.map((r, i) => ({
          value: logScale ? Math.max(r.mean, 0.001) : r.mean,
          itemStyle: {
            color: COLORS[i % COLORS.length],
            opacity: barOpacity(r.policy, brushedPolicies),
          },
        })),
      },
      ...(showErrorBars
        ? [
            {
              type: "custom" as const,
              renderItem: (
                params: { dataIndex: number },
                api: {
                  coord: (v: [number, number]) => [number, number];
                  style: (s: object) => object;
                }
              ) => {
                const i = params.dataIndex;
                const r = efficiencyRanking[i];
                const bounds = errorBarBounds(r.mean, r.std, "kg/km", logScale);
                const y = api.coord([bounds.center, i])[1];
                const xLeft = api.coord([bounds.low, i])[0];
                const xRight = api.coord([bounds.high, i])[0];
                const cap = 4;
                const st = api.style({ stroke: "#9090b0", lineWidth: 1.5 });
                return {
                  type: "group",
                  children: [
                    { type: "line", shape: { x1: xLeft, y1: y, x2: xRight, y2: y }, style: st },
                    { type: "line", shape: { x1: xLeft, y1: y - cap, x2: xLeft, y2: y + cap }, style: st },
                    { type: "line", shape: { x1: xRight, y1: y - cap, x2: xRight, y2: y + cap }, style: st },
                  ],
                };
              },
              data: efficiencyRanking.map((_, i) => i),
              z: 10,
            },
          ]
        : []),
    ],
  };
}

export function buildBenchmarkMetricBarOption(
  filteredRuns: RunFile[],
  metricKey: string,
  metricLabel: string,
  brushedPolicies: string[] | null,
  logScale: boolean,
  showErrorBars: boolean
) {
  const symlogOverflows = logScale && metricKey === "overflows";
  const runLabels = filteredRuns.map((r) => r.label);
  const policies = [
    ...new Set(filteredRuns.flatMap((r) => r.entries.map((e) => e.policy))),
  ];

  const statsGrid = policies.map((p) =>
    filteredRuns.map((r) => {
      const vals = r.entries
        .filter((e) => e.policy === p)
        .map((e) => (e.data as Record<string, number>)[metricKey] ?? null)
        .filter((v): v is number => v !== null);
      return { mean: mean(vals), std: std(vals) };
    })
  );

  const barSeries = policies.map((p, i) => ({
    name: p,
    type: "bar" as const,
    data: statsGrid[i].map((s) => {
      if (!logScale) return s.mean;
      return symlogOverflows ? symlog(s.mean) : Math.max(s.mean, 0.001);
    }),
    itemStyle: {
      color: COLORS[i % COLORS.length],
      opacity: barOpacity(p, brushedPolicies),
    },
  }));

  const errorBarPoints = policies.flatMap((_, polIdx) =>
    filteredRuns.map((_, runIdx) => ({
      runIdx,
      polIdx,
      ...statsGrid[polIdx][runIdx],
    }))
  );

  const errorBarSeries = showErrorBars
    ? [
        {
          type: "custom" as const,
          renderItem: (
            params: { dataIndex: number },
            api: {
              coord: (v: [number, number]) => [number, number];
              size: (v: [number, number]) => [number, number];
              style: (s: object) => object;
            }
          ) => {
            const point = errorBarPoints[params.dataIndex];
            const bounds = errorBarBounds(
              point.mean,
              point.std,
              metricKey,
              logScale,
              symlogOverflows
            );
            const x = groupedBarWhiskerX(
              api,
              point.runIdx,
              point.polIdx,
              policies.length,
              bounds.center
            );
            const yTop = api.coord([point.runIdx, bounds.high])[1];
            const yBot = api.coord([point.runIdx, bounds.low])[1];
            const cap = 4;
            const st = api.style({ stroke: "#9090b0", lineWidth: 1.5 });
            return {
              type: "group",
              children: [
                { type: "line", shape: { x1: x, y1: yTop, x2: x, y2: yBot }, style: st },
                { type: "line", shape: { x1: x - cap, y1: yTop, x2: x + cap, y2: yTop }, style: st },
                { type: "line", shape: { x1: x - cap, y1: yBot, x2: x + cap, y2: yBot }, style: st },
              ],
            };
          },
          data: errorBarPoints.map((_, i) => i),
          z: 10,
        },
      ]
    : [];

  return {
    backgroundColor: "transparent",
    legend: { textStyle: { color: "#9090b0", fontSize: 11 } },
    grid: { left: 50, right: 10, top: 35, bottom: 55 },
    xAxis: {
      type: "category",
      data: runLabels,
      axisLabel: { color: "#9090b0", fontSize: 9, rotate: 20 },
    },
    yAxis: {
      type: (logScale && !symlogOverflows ? "log" : "value") as "log" | "value",
      logBase: 10,
      name: metricLabel,
      nameTextStyle: { color: "#9090b0" },
      axisLabel: { color: "#9090b0", fontSize: 10 },
      minorSplitLine: { show: false },
    },
    series: [...barSeries, ...errorBarSeries],
    tooltip: {
      trigger: "axis",
      formatter: (params: unknown[]) => {
        const items = params as Array<{ seriesName: string; dataIndex: number }>;
        const run = runLabels[items[0]?.dataIndex ?? 0];
        const lines = items
          .filter((p) => p.seriesName)
          .map((p) => {
            const polIdx = policies.indexOf(p.seriesName);
            const s = statsGrid[polIdx]?.[p.dataIndex];
            return s
              ? `${p.seriesName}: ${fmt(s.mean)} ± ${fmt(s.std)}`
              : `${p.seriesName}: —`;
          });
        return `${run}<br/>${lines.join("<br/>")}`;
      },
    },
  };
}
