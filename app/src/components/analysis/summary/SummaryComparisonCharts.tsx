import { useMemo, useRef, useState } from "react";
import ReactECharts from "echarts-for-react";
import type EChartsReact from "echarts-for-react";
import { ChartExportButtons } from "../../common/ChartExportButtons";
import {
  chartMetricDisplay,
  chartMetricYAxisType,
  isLogScaleMetric,
} from "../../../utils/charts/chartLogScale";
import { barOpacity, isHighlighted } from "../../../utils/charts/chartHighlight";
import {
  formatPolicyMeta,
  type PolicyMeta,
} from "../../../utils/sim/simMetadata";
import {
  buildNormalizedHeatmapCells,
  type HeatmapMode,
} from "../../../utils/benchmark/heatmapMetrics";
import { SIM_METRICS, normalizeCohortMetric } from "../../ui/telemetry/metrics";
import {
  fmt,
  mean,
  POLICY_COLORS,
  std,
  trajectoryByPolicyAndDay,
  type MetricKey,
  type PolicyStats,
} from "./SummaryCoreWidgets";
import type { DayLogEntry } from "../../../types";

type TrajectoryMetric = "overflows" | "profit" | "km" | "kg";

export function TrajectoryChart({
  entries,
  policies,
  brushed,
  logScale = false,
}: {
  entries: DayLogEntry[];
  policies: string[];
  brushed?: string[] | null;
  logScale?: boolean;
}) {
  const chartRef = useRef<EChartsReact | null>(null);
  const [metric, setMetric] = useState<TrajectoryMetric>("overflows");
  const metricLog = logScale && isLogScaleMetric(metric);

  const traj = useMemo(
    () => trajectoryByPolicyAndDay(entries, metric),
    [entries, metric]
  );

  const allDays = useMemo(() => {
    const days = new Set<number>();
    for (const pts of Object.values(traj)) pts.forEach(([d]) => days.add(d));
    return [...days].sort((a, b) => a - b);
  }, [traj]);

  const option = useMemo(
    () => ({
      backgroundColor: "transparent",
      tooltip: { trigger: "axis" as const },
      legend: {
        data: policies,
        textStyle: { color: "#9090b0", fontSize: 10 },
        bottom: 0,
      },
      grid: { left: 50, right: 10, top: 20, bottom: 60 },
      xAxis: {
        type: "category" as const,
        data: allDays,
        name: "Day",
        nameTextStyle: { color: "#9090b0" },
        axisLabel: { color: "#9090b0", fontSize: 9 },
      },
      yAxis: {
        type: chartMetricYAxisType(metric, logScale),
        logBase: 10,
        axisLabel: { color: "#9090b0", fontSize: 10 },
        minorSplitLine: { show: false },
      },
      series: policies.map((policy, i) => ({
        name: policy,
        type: "line" as const,
        smooth: true,
        symbol: "none",
        lineStyle: { width: 1.5, opacity: barOpacity(policy, brushed ?? null) },
        color: POLICY_COLORS[i % POLICY_COLORS.length],
        data: allDays.map((day) => {
          const pt = traj[policy]?.find(([d]) => d === day);
          return pt ? chartMetricDisplay(pt[1], metric, logScale) : null;
        }),
      })),
    }),
    [traj, allDays, policies, brushed, metric, logScale]
  );

  const METRIC_OPTS: Array<{ key: TrajectoryMetric; label: string }> = [
    { key: "overflows", label: "Overflows" },
    { key: "profit", label: "Profit (€)" },
    { key: "km", label: "Distance (km)" },
    { key: "kg", label: "Waste (kg)" },
  ];

  return (
    <div className="card space-y-3">
      <div className="flex items-center justify-between flex-wrap gap-2">
        <div>
          <p className="text-xs font-semibold text-gray-300">
            Per-Day Trajectory
          </p>
          {metricLog && (
            <p className="text-[10px] text-canvas-muted">
              Log-scale y-axis · symlog overflows when selected
            </p>
          )}
        </div>
        <div className="flex items-center gap-2">
          {allDays.length > 0 && (
            <ChartExportButtons
              chartRef={{ current: chartRef.current }}
              filenameStem={`trajectory-${metric}`}
            />
          )}
          <div className="flex items-center gap-1 bg-canvas-elevated rounded-lg p-0.5">
            {METRIC_OPTS.map((o) => (
              <button
                key={o.key}
                onClick={() => setMetric(o.key)}
                className={`text-xs px-2.5 py-1 rounded-md transition-colors ${
                  metric === o.key
                    ? "bg-accent-primary text-white"
                    : "text-canvas-muted hover:text-gray-200"
                }`}
              >
                {o.label}
              </button>
            ))}
          </div>
        </div>
      </div>
      {allDays.length === 0 ? (
        <p className="text-xs text-canvas-muted">No day data available.</p>
      ) : (
        <ReactECharts ref={chartRef} option={option} style={{ height: 260 }} />
      )}
    </div>
  );
}

const RADAR_METRICS: Array<{ key: MetricKey; label: string }> = [
  { key: "profit", label: "Profit (↑)" },
  { key: "km", label: "Distance (↓)" },
  { key: "overflows", label: "Overflows (↓)" },
  { key: "kg", label: "Waste (↑)" },
];

/**
 * Direction-Aware Policy Radar Chart (Owner Ruling #4, Slice V6).
 *
 * Uses `normalizeCohortMetric` so `1.0` strictly represents the best in cohort
 * on every axis (inverting `km` and `overflows`), with raw mean ± std in tooltips.
 */
export function PolicyRadarChart({
  stats,
  policies,
  brushed,
  policyMeta,
  logScale = false,
}: {
  stats: Record<string, PolicyStats>;
  policies: string[];
  brushed?: string[] | null;
  policyMeta?: Record<string, PolicyMeta>;
  logScale?: boolean;
}) {
  const chartRef = useRef<EChartsReact | null>(null);

  const option = useMemo(() => {
    const rawMeansByAxis: Record<MetricKey, number[]> = {
      profit: policies.map((p) => mean(stats[p].profit)),
      km: policies.map((p) => mean(stats[p].km)),
      overflows: policies.map((p) => mean(stats[p].overflows)),
      kg: policies.map((p) => mean(stats[p].kg)),
    };

    const normByAxis: Record<MetricKey, number[]> = {
      profit: normalizeCohortMetric(rawMeansByAxis.profit, SIM_METRICS.profit.higherIsBetter),
      km: normalizeCohortMetric(rawMeansByAxis.km, SIM_METRICS.km.higherIsBetter),
      overflows: normalizeCohortMetric(
        rawMeansByAxis.overflows,
        SIM_METRICS.overflows.higherIsBetter
      ),
      kg: normalizeCohortMetric(rawMeansByAxis.kg, SIM_METRICS.kg.higherIsBetter),
    };

    return {
      backgroundColor: "transparent",
      legend: { data: policies, textStyle: { color: "#9090b0", fontSize: 10 } },
      radar: {
        indicator: RADAR_METRICS.map(({ label }) => ({
          name: label,
          max: 1.0,
          min: 0.0,
        })),
        axisLine: { lineStyle: { color: "#2d2d50" } },
        splitLine: { lineStyle: { color: "#2d2d50" } },
        name: { textStyle: { color: "#9090b0", fontSize: 9 } },
      },
      series: [
        {
          type: "radar" as const,
          data: policies.map((p, i) => ({
            name: p,
            value: RADAR_METRICS.map(({ key }) =>
              Number((normByAxis[key][i] ?? 0.15).toFixed(3))
            ),
            lineStyle: {
              color: POLICY_COLORS[i % POLICY_COLORS.length],
              opacity: barOpacity(p, brushed ?? null),
            },
            areaStyle: {
              color: `${POLICY_COLORS[i % POLICY_COLORS.length]}20`,
              opacity: barOpacity(p, brushed ?? null),
            },
          })),
        },
      ],
      tooltip: {
        formatter: (p: { name: string }) => {
          const meta = policyMeta?.[p.name];
          const s = stats[p.name];
          const kpiLines = s
            ? RADAR_METRICS.map(({ key }) => {
                const c = SIM_METRICS[key];
                return `${c.shortLabel}: ${fmt(mean(s[key]), 1)} ± ${fmt(
                  std(s[key]),
                  1
                )} ${c.unit}`;
              })
            : [];
          return [p.name, meta ? formatPolicyMeta(meta) : "", ...kpiLines]
            .filter(Boolean)
            .join("<br/>");
        },
      },
    };
  }, [stats, policies, brushed, policyMeta]);

  return (
    <div className="card space-y-2">
      <div className="flex items-center justify-between">
        <p className="text-xs font-semibold text-gray-300">
          Policy Radar (1.0 = Best in Cohort)
          {logScale ? " · direction-normalised" : ""}
        </p>
        <ChartExportButtons
          chartRef={{ current: chartRef.current }}
          filenameStem="summary-radar"
        />
      </div>
      <ReactECharts ref={chartRef} option={option} style={{ height: 280 }} />
    </div>
  );
}

const HEATMAP_METRICS: Array<{
  key: MetricKey | "kg/km";
  label: string;
  higherBetter: boolean;
}> = [
  { key: "profit", label: "Profit", higherBetter: true },
  { key: "kg/km", label: "kg/km", higherBetter: true },
  { key: "overflows", label: "Overflows", higherBetter: false },
  { key: "km", label: "km", higherBetter: false },
];

export function PolicyHeatmapChart({
  stats,
  policies,
  mode,
  onModeChange,
  brushed,
  policyMeta,
  logScale = false,
}: {
  stats: Record<string, PolicyStats>;
  policies: string[];
  mode: HeatmapMode;
  onModeChange?: (mode: HeatmapMode) => void;
  brushed?: string[] | null;
  policyMeta?: Record<string, PolicyMeta>;
  logScale?: boolean;
}) {
  const chartRef = useRef<EChartsReact | null>(null);

  const activeMetrics = useMemo(
    () =>
      mode === "all"
        ? HEATMAP_METRICS
        : HEATMAP_METRICS.filter((m) => m.key === mode),
    [mode]
  );

  const option = useMemo(() => {
    const getRaw = (policy: string, metricKey: string) =>
      metricKey === "kg/km"
        ? mean(stats[policy]["kg/km"])
        : mean(stats[policy][metricKey as MetricKey]);

    const { cells, raw: rawValues } = buildNormalizedHeatmapCells(
      policies,
      activeMetrics,
      getRaw,
      (p) => (isHighlighted(p, brushed ?? null) ? 1 : 0.15),
      logScale
    );

    return {
      backgroundColor: "transparent",
      grid: { left: 72, right: 24, top: 8, bottom: 48 },
      xAxis: {
        type: "category" as const,
        data: policies,
        axisLabel: { color: "#9090b0", fontSize: 8, rotate: 30 },
      },
      yAxis: {
        type: "category" as const,
        data: activeMetrics.map((m) => m.label),
        axisLabel: { color: "#9090b0", fontSize: 9 },
      },
      visualMap: {
        min: 0,
        max: 1,
        calculable: false,
        orient: "horizontal" as const,
        left: "center",
        bottom: 0,
        inRange: { color: ["#1e1b4b", "#6366f1", "#34d399"] },
        textStyle: { color: "#9090b0", fontSize: 9 },
        show: false,
      },
      series: [
        {
          type: "heatmap" as const,
          data: cells,
          label: { show: false },
          emphasis: {
            itemStyle: { shadowBlur: 6, shadowColor: "rgba(0,0,0,0.4)" },
          },
        },
      ],
      tooltip: {
        formatter: (p: { value: [number, number, number] }) => {
          const [pi, mi, norm] = p.value;
          const { label } = activeMetrics[mi];
          const policy = policies[pi];
          const meta = policyMeta?.[policy];
          return [
            policy,
            meta ? formatPolicyMeta(meta) : "",
            `${label}: ${fmt(rawValues[mi][pi], 2)}`,
            `Score: ${fmt(norm * 100, 0)}%`,
          ]
            .filter(Boolean)
            .join("<br/>");
        },
      },
    };
  }, [stats, policies, activeMetrics, brushed, policyMeta, logScale]);

  const MODE_OPTS: Array<{ key: HeatmapMode; label: string }> = [
    { key: "all", label: "All metrics" },
    { key: "overflows", label: "Overflows" },
    { key: "kg/km", label: "kg/km" },
  ];

  return (
    <div className="card space-y-2">
      <div className="flex items-center justify-between flex-wrap gap-2">
        <p className="text-xs font-semibold text-gray-300">
          Policy Metric Heatmap{logScale ? " · log-normalised" : ""}
        </p>
        <div className="flex items-center gap-2">
          {onModeChange && (
            <div className="flex items-center gap-1 bg-canvas-elevated rounded-lg p-0.5">
              {MODE_OPTS.map((o) => (
                <button
                  key={o.key}
                  onClick={() => onModeChange(o.key)}
                  className={`text-xs px-2.5 py-1 rounded-md transition-colors ${
                    mode === o.key
                      ? "bg-accent-primary text-white"
                      : "text-canvas-muted hover:text-gray-200"
                  }`}
                >
                  {o.label}
                </button>
              ))}
            </div>
          )}
          <ChartExportButtons
            chartRef={{ current: chartRef.current }}
            filenameStem="summary-heatmap"
          />
        </div>
      </div>
      <ReactECharts
        ref={chartRef}
        option={option}
        style={{ height: Math.max(120, activeMetrics.length * 36 + 60) }}
      />
    </div>
  );
}
