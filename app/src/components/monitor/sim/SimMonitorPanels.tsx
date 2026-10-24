import { useMemo, useRef } from "react";
import ReactECharts from "echarts-for-react";
import { ChartExportButtons } from "../../common/ChartExportButtons";
import {
  chartMetricDisplay,
  chartMetricYAxisType,
  isLogScaleMetric,
} from "../../../utils/charts/chartLogScale";
import type { DayLogEntry, SimDayData } from "../../../types";

export const PRIMARY_KPIS = [
  { key: "profit", label: "Profit (€)", lowerIsBetter: false },
  { key: "km", label: "Distance (km)", lowerIsBetter: true },
  { key: "kg", label: "Waste (kg)", lowerIsBetter: false },
  { key: "overflows", label: "Overflows", lowerIsBetter: true },
] as const;

export const SECONDARY_KPIS = [
  { key: "ncol", label: "Collections", lowerIsBetter: false },
  { key: "kg_lost", label: "Waste Lost (kg)", lowerIsBetter: true },
  { key: "kg/km", label: "Efficiency (kg/km)", lowerIsBetter: false },
  { key: "cost", label: "Cost (€)", lowerIsBetter: true },
] as const;

export const POLICY_COLORS = [
  "#6366f1",
  "#34d399",
  "#f87171",
  "#fbbf24",
  "#a78bfa",
  "#fb923c",
  "#38bdf8",
  "#f472b6",
];

export function computeDelta(
  entries: DayLogEntry[],
  day: number,
  key: string
): number | null {
  const prev = entries.find((e) => e.day === day - 1);
  const curr = entries.find((e) => e.day === day);
  if (!prev || !curr) return null;
  const a = (curr.data as Record<string, number>)[key];
  const b = (prev.data as Record<string, number>)[key];
  if (typeof a !== "number" || typeof b !== "number") return null;
  return a - b;
}

export function MetricTimeseries({
  policySeries,
  metricKey,
  label,
  logScale = false,
}: {
  policySeries: { policy: string; entries: DayLogEntry[]; color: string }[];
  metricKey: string;
  label: string;
  logScale?: boolean;
}) {
  const chartRef = useRef<ReactECharts>(null);
  const allDays = [
    ...new Set(policySeries.flatMap((s) => s.entries.map((e) => e.day))),
  ].sort((a, b) => a - b);
  const metricLog = logScale && isLogScaleMetric(metricKey);

  const series = policySeries.map(({ policy, entries, color }) => ({
    name: policy,
    type: "line" as const,
    smooth: true,
    symbol: "circle",
    symbolSize: 3,
    lineStyle: { color, width: 1.8 },
    itemStyle: { color },
    areaStyle: policySeries.length === 1 ? { color: `${color}1e` } : undefined,
    data: allDays.map((day) => {
      const e = entries.find((en) => en.day === day);
      if (!e) return null;
      const raw = (e.data as Record<string, number>)[metricKey];
      return chartMetricDisplay(raw, metricKey, logScale);
    }),
  }));

  return (
    <div className="card">
      <div className="flex items-center justify-between mb-2">
        <p className="text-xs text-canvas-muted">
          {label}
          {metricLog ? " · log" : ""}
        </p>
        <ChartExportButtons
          chartRef={chartRef}
          filenameStem={`${metricKey}-timeseries`}
          size={11}
        />
      </div>
      <ReactECharts
        ref={chartRef}
        option={{
          backgroundColor: "transparent",
          grid: {
            left: 40,
            right: 10,
            top: policySeries.length > 1 ? 20 : 10,
            bottom: 30,
          },
          xAxis: {
            type: "category",
            data: allDays,
            axisLabel: { color: "#9090b0", fontSize: 10 },
          },
          yAxis: {
            type: chartMetricYAxisType(metricKey, logScale),
            logBase: 10,
            axisLabel: { color: "#9090b0", fontSize: 10 },
            minorSplitLine: { show: false },
          },
          legend:
            policySeries.length > 1
              ? {
                  data: policySeries.map((s) => s.policy),
                  textStyle: { color: "#9090b0", fontSize: 9 },
                  top: 0,
                }
              : undefined,
          series,
          tooltip: { trigger: "axis" },
        }}
        style={{ height: 140 }}
      />
    </div>
  );
}

export function BinFillStrip({ data }: { data: SimDayData }) {
  const { bin_state_c, bin_state_collected, mandatory } = data;
  if (!bin_state_c?.length) return null;

  const mandatorySet = new Set(mandatory ?? []);
  const sorted = [...bin_state_c.entries()]
    .sort((a, b) => b[1] - a[1])
    .slice(0, 25);

  return (
    <div className="card space-y-2">
      <p className="text-xs text-canvas-muted">
        Bin Fill Levels — top 25 of {bin_state_c.length} bins
      </p>
      <div className="space-y-1">
        {sorted.map(([idx, fill]) => {
          const pct = Math.min(100, fill * 100);
          const barColor =
            pct >= 100
              ? "bg-accent-danger"
              : pct >= 80
              ? "bg-accent-warning"
              : "bg-accent-success";
          const isMandatory = mandatorySet.has(idx);
          const isCollected = bin_state_collected?.[idx] ?? false;

          return (
            <div key={idx} className="flex items-center gap-2 text-xs tabular-nums">
              <span className="font-mono w-10 text-right text-canvas-muted">
                #{idx}
              </span>
              <div className="flex-1 h-2.5 bg-canvas-elevated rounded-full overflow-hidden">
                <div
                  className={`h-full ${barColor} rounded-full`}
                  style={{ width: `${pct}%` }}
                />
              </div>
              <span className="font-mono w-10 text-right text-gray-400">
                {pct.toFixed(0)}%
              </span>
              {isMandatory && (
                <span
                  className="text-accent-danger font-bold w-4 text-center"
                  title="Mandatory"
                >
                  !
                </span>
              )}
              {isCollected && (
                <span
                  className="text-accent-success w-4 text-center"
                  title="Collected"
                >
                  ✓
                </span>
              )}
              {!isMandatory && !isCollected && <span className="w-4" />}
            </div>
          );
        })}
      </div>
    </div>
  );
}

export function TourTable({
  data,
  day,
  policy,
}: {
  data: SimDayData;
  day: number;
  policy: string;
}) {
  const { tour, tour_indices, bin_state_c, bin_state_collected, mandatory } = data;

  const indices: number[] = useMemo(() => {
    if (tour_indices?.length) return tour_indices;
    if (tour?.length) {
      return tour.map((stop) => (typeof stop === "number" ? stop : stop.id));
    }
    return [];
  }, [tour, tour_indices]);

  if (indices.length === 0) return null;

  const mandatorySet = new Set(mandatory ?? []);
  const LIMIT = 60;

  return (
    <div className="card space-y-2">
      <p className="text-xs text-canvas-muted">
        Tour — Day {day} · {policy} · {indices.length} stops
      </p>
      <div className="overflow-x-auto">
        <table className="w-full text-xs tabular-nums">
          <thead>
            <tr className="border-b border-canvas-border text-canvas-muted text-left">
              <th className="py-1 pr-3 font-normal">#</th>
              <th className="py-1 pr-3 font-normal">Bin</th>
              <th className="py-1 pr-3 text-right font-normal">Fill</th>
              <th className="py-1 pr-2 text-center font-normal">Collected</th>
              <th className="py-1 text-center font-normal">Mandatory</th>
            </tr>
          </thead>
          <tbody>
            {indices.slice(0, LIMIT).map((binIdx, stop) => {
              const fill = bin_state_c ? bin_state_c[binIdx] : undefined;
              const collected = bin_state_collected?.[binIdx];
              const isMandatory = mandatorySet.has(binIdx);
              return (
                <tr
                  key={stop}
                  className={`border-b border-canvas-border/20 hover:bg-canvas-hover ${
                    isMandatory ? "text-accent-warning" : "text-gray-300"
                  }`}
                >
                  <td className="py-0.5 pr-3 text-canvas-muted">{stop + 1}</td>
                  <td className="py-0.5 pr-3 font-mono">#{binIdx}</td>
                  <td className="py-0.5 pr-3 text-right font-mono">
                    {fill != null ? `${(fill * 100).toFixed(1)}%` : "—"}
                  </td>
                  <td className="py-0.5 pr-2 text-center">
                    {collected ? (
                      <span className="text-accent-success">✓</span>
                    ) : (
                      <span className="text-canvas-muted">—</span>
                    )}
                  </td>
                  <td className="py-0.5 text-center">
                    {isMandatory ? (
                      <span className="text-accent-danger font-bold">!</span>
                    ) : (
                      <span className="text-canvas-muted">—</span>
                    )}
                  </td>
                </tr>
              );
            })}
          </tbody>
        </table>
        {indices.length > LIMIT && (
          <p className="text-xs text-canvas-muted mt-1">
            Showing {LIMIT} of {indices.length} stops.
          </p>
        )}
      </div>
    </div>
  );
}
