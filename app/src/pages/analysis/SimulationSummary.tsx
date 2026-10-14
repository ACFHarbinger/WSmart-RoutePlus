/**
 * Simulation Summary — post-run aggregate analytics (§G.1 / §G.16).
 *
 * Displays:
 *  • Per-policy mean ± std KPI cards
 *  • Policy ranking table (sortable by any metric)
 *  • Per-day trajectory overlay chart (overflows / profit over simulation days)
 *  • Four bar charts: profit, km, overflows, kg/km
 */
import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import ReactECharts from "echarts-for-react";
import type EChartsReact from "echarts-for-react";
import { invoke } from "@tauri-apps/api/core";
import { open } from "@tauri-apps/plugin-dialog";
import { FolderOpen, ChevronUp, ChevronDown, ChevronLeft, ChevronRight, Download } from "lucide-react";
import { useAppStore } from "../../store/app";
import { useRecentHandoff } from "../../hooks/files/useRecentHandoff";
import { GlobalFilterBar } from "../../components/layout/GlobalFilterBar";
import { useLogPathRunLabelBrush } from "../../hooks/brush/useLogPathRunLabelBrush";
import { usePortfolioRunBrush } from "../../hooks/brush/usePortfolioRunBrush";
import { useGlobalFiltersStore } from "../../store/filters";
import { filterEntries } from "../../store/sim";
import { ChartExportButtons } from "../../components/common/ChartExportButtons";
import { OpenPathToolbar } from "../../components/common/OpenPathToolbar";
import { LoadedRunRow } from "../../components/common/LoadedRunRow";
import { paretoFront, paretoStepLine } from "../../utils/benchmark/pareto";
import {
  chartMetricDisplay,
  chartMetricUsesSymlog,
  chartMetricYAxisType,
  errorBarBounds,
  invertParallelAxisValue,
  isLogScaleMetric,
  parallelAxisValue,
  radarAxisValue,
} from "../../utils/charts/chartLogScale";
import { symlog } from "../../utils/charts/symlog";
import { barOpacity, isHighlighted, toggleBrush } from "../../utils/charts/chartHighlight";
import {
  citySymbol,
  formatLogMeta,
  formatPolicyMeta,
  parseLogPath,
  parsePolicyLabel,
  strategyColor,
  type LogPathMeta,
  type PolicyMeta,
} from "../../utils/sim/simMetadata";
import {
  buildPolicyHierarchy,
  buildPortfolioHierarchy,
  childrenAtPath,
  enrichDrillChildren,
  policiesAtPath,
  resolveDrillBarColor,
  type HierarchyColorMode,
  type PortfolioHierarchyRun,
} from "../../utils/benchmark/policyHierarchy";
import { BenchmarkParetoPanel } from "../../components/analysis/benchmark/BenchmarkParetoPanel";
import { BenchmarkGraphHeatmap } from "../../components/analysis/benchmark/BenchmarkGraphHeatmap";
import { BenchmarkDistributionHeatmap } from "../../components/analysis/benchmark/BenchmarkDistributionHeatmap";
import { BenchmarkPortfolioHeatmap } from "../../components/analysis/benchmark/BenchmarkPortfolioHeatmap";
import { BenchmarkPortfolioParallel } from "../../components/analysis/benchmark/BenchmarkPortfolioParallel";
import { StrategyLegend } from "../../components/analysis/telemetry/StrategyLegend";
import {
  buildNormalizedHeatmapCells,
  type HeatmapMode,
} from "../../utils/benchmark/heatmapMetrics";
import { PortfolioEfficiencyRanking } from "../../components/analysis/benchmark/PortfolioEfficiencyRanking";
import { buildParetoByPanel, type PortfolioRunSlice } from "../../utils/benchmark/paretoPortfolio";
import { groupRunsByDistribution } from "../../utils/benchmark/portfolioDistribution";
import { PARETO_PANELS } from "../../utils/benchmark/paretoPanels";
import {
  groupRunsByCity,
  buildCityComparisonSeries,
  cityComparisonChartOption,
} from "../../utils/benchmark/cityComparison";
import {
  loadPortfolioLogs,
  PORTFOLIO_SCAN_DEFAULT,
  scanOutputPortfolio,
} from "../../utils/runs/outputRunLogs";
import { downloadCsv, downloadParquetTable } from "../../utils/charts/tableExport";
import { buildPolicyParallelAxes } from "../../utils/benchmark/parallelPolicyAxes";
import {
  formatPipelineTimingBadge,
  portfolioRunLabel,
  runPortfolioSimulationArrowPipeline,
} from "../../utils/duckdb/arrowPipeline";
import { RouteViz } from "../../components/analysis/routes/RouteViz";
import { PolicyTelemetryTrendsPanel } from "../../components/analysis/telemetry/PolicyTelemetryTrendsPanel";

import { SqlQueryPanel } from "../../components/analysis/explorer/SqlQueryPanel";
import { useDuckDbStore } from "../../store/duckdb";
import { toast } from "sonner";
import type { DayLogEntry } from "../../types";

const SUMMARY_SIM_TABLE = "summary_sim";

type HierarchyView = "sunburst" | "treemap";

// ── Stat helpers ──────────────────────────────────────────────────────────────

function mean(arr: number[]) {
  return arr.length ? arr.reduce((a, b) => a + b, 0) / arr.length : 0;
}

function std(arr: number[]) {
  if (arr.length < 2) return 0;
  const m = mean(arr);
  return Math.sqrt(arr.reduce((s, x) => s + (x - m) ** 2, 0) / (arr.length - 1));
}

function fmt(n: number, decimals = 2): string {
  return isFinite(n) ? n.toFixed(decimals) : "—";
}

// ── Aggregation helpers ───────────────────────────────────────────────────────

interface PolicyStats {
  profit: number[];
  km: number[];
  overflows: number[];
  kg: number[];
  "kg/km": number[];
  cost: number[];
  days: number;
}

type MetricKey = "profit" | "km" | "overflows" | "kg";

function aggregateByPolicy(entries: DayLogEntry[]): Record<string, PolicyStats> {
  const map: Record<string, PolicyStats> = {};
  for (const e of entries) {
    if (!map[e.policy]) {
      map[e.policy] = { profit: [], km: [], overflows: [], kg: [], "kg/km": [], cost: [], days: 0 };
    }
    const d = e.data;
    if (d.profit != null) map[e.policy].profit.push(d.profit);
    if (d.km != null) map[e.policy].km.push(d.km);
    if (d.overflows != null) map[e.policy].overflows.push(d.overflows);
    if (d.kg != null) map[e.policy].kg.push(d.kg);
    if (d["kg/km"] != null) map[e.policy]["kg/km"].push(d["kg/km"]);
    if (d.cost != null) map[e.policy].cost.push(d.cost);
    map[e.policy].days++;
  }
  return map;
}

/** Group entries by (policy, day) and average across samples for trajectory charts. */
function trajectoryByPolicyAndDay(
  entries: DayLogEntry[],
  metric: keyof DayLogEntry["data"]
): Record<string, Array<[number, number]>> {
  const acc: Record<string, Record<number, number[]>> = {};
  for (const e of entries) {
    const val = e.data[metric] as number | undefined;
    if (val == null) continue;
    if (!acc[e.policy]) acc[e.policy] = {};
    if (!acc[e.policy][e.day]) acc[e.policy][e.day] = [];
    acc[e.policy][e.day].push(val);
  }
  const result: Record<string, Array<[number, number]>> = {};
  for (const policy of Object.keys(acc)) {
    result[policy] = Object.entries(acc[policy])
      .map(([day, vals]) => [Number(day), mean(vals)] as [number, number])
      .sort((a, b) => a[0] - b[0]);
  }
  return result;
}

// ── Palette ───────────────────────────────────────────────────────────────────

const POLICY_COLORS = [
  "#6366f1", "#34d399", "#f87171", "#fbbf24",
  "#a78bfa", "#fb923c", "#38bdf8", "#f472b6",
];

function policyTooltipFooter(
  policy: string,
  policyMeta: Record<string, PolicyMeta> | undefined,
  days: number
): string {
  const meta = policyMeta?.[policy];
  const lines = meta ? [formatPolicyMeta(meta)] : [];
  lines.push(`Days: ${days}`);
  return lines.join("<br/>");
}

function ConfigMetaBanner({
  logPath,
  logMeta,
  projectRoot,
}: {
  logPath: string;
  logMeta: LogPathMeta;
  projectRoot: string | null;
}) {
  return (
    <div className="card flex flex-wrap items-center gap-x-4 gap-y-1 text-xs text-canvas-muted">
      <span className="font-semibold text-gray-300">Run config</span>
      <span>{formatLogMeta(logMeta)}</span>
      <OpenPathToolbar
        path={logPath}
        projectRoot={projectRoot}
        kind="log"
        labeled
        labeledTargets={["monitor"]}
        labeledIconSize={12}
        chipClassName="opacity-70"
      />
    </div>
  );
}

function PolicyBrushBar({
  policies,
  brushed,
  onToggle,
  onClear,
}: {
  policies: string[];
  brushed: string[] | null;
  onToggle: (policy: string) => void;
  onClear: () => void;
}) {
  const active = brushed ?? [];
  return (
    <div className="card space-y-2">
      <div className="flex items-center justify-between">
        <p className="text-xs font-semibold text-gray-300">Cross-filter (§G.1)</p>
        {active.length > 0 && (
          <button onClick={onClear} className="btn-ghost text-xs">
            Clear filter
          </button>
        )}
      </div>
      <div className="flex flex-wrap gap-1.5">
        {policies.map((p, i) => {
          const on = active.length === 0 || active.includes(p);
          return (
            <button
              key={p}
              onClick={() => onToggle(p)}
              className={`text-xs px-2 py-0.5 rounded-full border transition-opacity ${
                on ? "opacity-100" : "opacity-35"
              }`}
              style={{
                borderColor: POLICY_COLORS[i % POLICY_COLORS.length],
                color: on ? POLICY_COLORS[i % POLICY_COLORS.length] : undefined,
              }}
            >
              {parsePolicyLabel(p).selectionStrategy}
            </button>
          );
        })}
      </div>
      <p className="text-[10px] text-canvas-muted">
        Click a policy chip or any bar chart to cross-filter all panels.
      </p>
    </div>
  );
}

interface GroupRow {
  label: string;
  mean: number;
  std: number;
  policies: string[];
}

function buildGroupedStats(
  policies: string[],
  stats: Record<string, PolicyStats>,
  groupFn: (p: string) => string,
  metric: "overflows" | "kg/km"
): GroupRow[] {
  const map = new Map<string, string[]>();
  for (const p of policies) {
    const g = groupFn(p);
    const list = map.get(g) ?? [];
    list.push(p);
    map.set(g, list);
  }
  return [...map.entries()].map(([label, ps]) => {
    const vals = ps.flatMap((p) => stats[p][metric]);
    return { label, policies: ps, mean: mean(vals), std: std(vals) };
  });
}

function GroupedMetricBarChart({
  title,
  subtitle,
  groups,
  color,
  showErrorBars,
  logScale = false,
  useSymlog = false,
  metricKey = "profit",
  exportName,
}: {
  title: string;
  subtitle?: string;
  groups: GroupRow[];
  color: string;
  showErrorBars?: boolean;
  logScale?: boolean;
  useSymlog?: boolean;
  metricKey?: string;
  exportName: string;
}) {
  const chartRef = useRef<EChartsReact | null>(null);
  const labels = groups.map((g) => g.label);
  const symlogMode = logScale && useSymlog;
  const displayValue = (v: number) =>
    !logScale ? v : symlogMode ? symlog(v) : Math.max(v, 0.001);

  const option = useMemo(
    () => ({
      backgroundColor: "transparent",
      tooltip: {
        trigger: "axis" as const,
        formatter: (params: unknown[]) => {
          const p = (params as Array<{ dataIndex: number }>)[0];
          const g = groups[p.dataIndex];
          return `${g.label}<br/>${fmt(g.mean, 2)} ± ${fmt(g.std, 2)}<br/>${g.policies.length} policy variant(s)`;
        },
      },
      grid: { left: 50, right: 10, top: 24, bottom: 40 },
      xAxis: {
        type: "category" as const,
        data: labels,
        axisLabel: { color: "#9090b0", fontSize: 9 },
      },
      yAxis: {
        type: (logScale && !symlogMode ? "log" : "value") as "log" | "value",
        logBase: 10,
        axisLabel: { color: "#9090b0", fontSize: 9 },
        minorSplitLine: { show: false },
      },
      series: [
        {
          type: "bar" as const,
          data: groups.map((g) => displayValue(g.mean)),
          itemStyle: { color },
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
                  const g = groups[params.dataIndex];
                  const errKey = symlogMode ? "overflows" : metricKey;
                  const bounds = errorBarBounds(g.mean, g.std, errKey, logScale, symlogMode);
                  const x = api.coord([params.dataIndex, bounds.center])[0];
                  const yTop = api.coord([params.dataIndex, bounds.high])[1];
                  const yBot = api.coord([params.dataIndex, bounds.low])[1];
                  const cap = 5;
                  return {
                    type: "group",
                    children: [
                      {
                        type: "line",
                        shape: { x1: x, y1: yTop, x2: x, y2: yBot },
                        style: api.style({ stroke: "#9090b0", lineWidth: 1.5 }),
                      },
                      {
                        type: "line",
                        shape: { x1: x - cap, y1: yTop, x2: x + cap, y2: yTop },
                        style: api.style({ stroke: "#9090b0", lineWidth: 1.5 }),
                      },
                      {
                        type: "line",
                        shape: { x1: x - cap, y1: yBot, x2: x + cap, y2: yBot },
                        style: api.style({ stroke: "#9090b0", lineWidth: 1.5 }),
                      },
                    ],
                  };
                },
                data: groups.map((_, i) => i),
                z: 10,
              },
            ]
          : []),
      ],
    }),
    [groups, labels, color, showErrorBars, logScale, symlogMode, metricKey]
  );

  return (
    <div className="card">
      <div className="flex items-center justify-between mb-1">
        <div>
          <p className="text-xs text-canvas-muted">{title}</p>
          {subtitle && <p className="text-[10px] text-canvas-muted">{subtitle}</p>}
        </div>
        <ChartExportButtons
          chartRef={{ current: chartRef.current }}
          filenameStem={exportName}
        />
      </div>
      <ReactECharts ref={chartRef} option={option} style={{ height: 200 }} />
    </div>
  );
}

// ── Subcomponents ─────────────────────────────────────────────────────────────

type SortKey = MetricKey;
type SortDir = "asc" | "desc";

function RankingTable({
  stats,
  policies,
  onExport,
  onExportParquet,
  parquetExporting,
}: {
  stats: Record<string, PolicyStats>;
  policies: string[];
  onExport: () => void;
  onExportParquet?: () => void;
  parquetExporting?: boolean;
}) {
  const [sortKey, setSortKey] = useState<SortKey>("profit");
  const [sortDir, setSortDir] = useState<SortDir>("desc");

  const sorted = useMemo(() => {
    return [...policies].sort((a, b) => {
      const va = mean(stats[a][sortKey]);
      const vb = mean(stats[b][sortKey]);
      return sortDir === "desc" ? vb - va : va - vb;
    });
  }, [policies, stats, sortKey, sortDir]);

  const toggleSort = (key: SortKey) => {
    if (key === sortKey) setDir((d) => (d === "desc" ? "asc" : "desc"));
    else { setSortKey(key); setSortDir("desc"); }
  };

  function setDir(fn: (d: SortDir) => SortDir) {
    setSortDir(fn);
  }

  const SortIcon = ({ k }: { k: SortKey }) => {
    if (k !== sortKey) return <span className="w-3" />;
    return sortDir === "desc" ? <ChevronDown size={11} /> : <ChevronUp size={11} />;
  };

  const cols: Array<{ key: SortKey; label: string; unit: string }> = [
    { key: "profit", label: "Profit", unit: "€" },
    { key: "km", label: "Distance", unit: "km" },
    { key: "overflows", label: "Overflows", unit: "" },
    { key: "kg", label: "Waste", unit: "kg" },
  ];

  return (
    <div className="card overflow-x-auto">
      <div className="flex items-center justify-between mb-3">
        <p className="text-xs font-semibold text-gray-300">Policy Ranking</p>
        <div className="flex items-center gap-2">
          <button onClick={onExport} className="btn-ghost text-xs flex items-center gap-1">
            <Download size={12} />
            Export CSV
          </button>
          {onExportParquet && (
            <button
              onClick={onExportParquet}
              disabled={parquetExporting}
              className="btn-ghost text-xs flex items-center gap-1"
            >
              <Download size={12} />
              Export Parquet
            </button>
          )}
        </div>
      </div>
      <table className="w-full text-xs min-w-[520px]">
        <thead>
          <tr className="border-b border-canvas-border text-left">
            <th className="py-2 px-3 text-canvas-muted font-medium w-8">#</th>
            <th className="py-2 px-3 text-canvas-muted font-medium">Policy</th>
            {cols.map((c) => (
              <th
                key={c.key}
                className="py-2 px-3 text-canvas-muted font-medium cursor-pointer hover:text-gray-300 select-none"
                onClick={() => toggleSort(c.key)}
              >
                <span className="flex items-center gap-1">
                  Mean {c.label} {c.unit ? `(${c.unit})` : ""}
                  <SortIcon k={c.key} />
                </span>
              </th>
            ))}
            <th className="py-2 px-3 text-canvas-muted font-medium">Days</th>
          </tr>
        </thead>
        <tbody className="divide-y divide-canvas-border">
          {sorted.map((policy, rank) => {
            const s = stats[policy];
            const color = POLICY_COLORS[policies.indexOf(policy) % POLICY_COLORS.length];
            return (
              <tr key={policy} className="hover:bg-canvas-hover">
                <td className="py-1.5 px-3 text-canvas-muted">{rank + 1}</td>
                <td className="py-1.5 px-3">
                  <span className="flex items-center gap-1.5">
                    <span className="w-2 h-2 rounded-full shrink-0" style={{ background: color }} />
                    <span className="font-mono text-accent-secondary">{policy}</span>
                  </span>
                </td>
                {cols.map((c) => {
                  const vals = s[c.key];
                  const m = mean(vals);
                  const sd = std(vals);
                  return (
                    <td key={c.key} className="py-1.5 px-3 font-mono text-gray-300">
                      {fmt(m, 1)}
                      {sd > 0 && (
                        <span className="text-canvas-muted text-[10px] ml-1">±{fmt(sd, 1)}</span>
                      )}
                    </td>
                  );
                })}
                <td className="py-1.5 px-3 text-canvas-muted">{s.days}</td>
              </tr>
            );
          })}
        </tbody>
      </table>
    </div>
  );
}

type TrajectoryMetric = "overflows" | "profit" | "km" | "kg";

function TrajectoryChart({
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

  const option = useMemo(() => ({
    backgroundColor: "transparent",
    tooltip: { trigger: "axis" as const },
    legend: { data: policies, textStyle: { color: "#9090b0", fontSize: 10 }, bottom: 0 },
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
  }), [traj, allDays, policies, brushed, metric, logScale]);

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
          <p className="text-xs font-semibold text-gray-300">Per-Day Trajectory</p>
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
  { key: "profit", label: "Profit (€)" },
  { key: "km", label: "Distance (km)" },
  { key: "overflows", label: "Overflows" },
  { key: "kg", label: "Waste (kg)" },
];

function PolicyRadarChart({
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
    const metricMeans: Record<string, Record<string, number>> = {};
    for (const p of policies) {
      metricMeans[p] = {};
      for (const { key } of RADAR_METRICS) {
        metricMeans[p][key] = mean(stats[p][key]);
      }
    }
    const displayValue = (key: string, raw: number) => radarAxisValue(raw, key, logScale);
    const maxes = RADAR_METRICS.map(({ key }) =>
      Math.max(...policies.map((p) => displayValue(key, metricMeans[p][key] ?? 0)), 1)
    );

    return {
      backgroundColor: "transparent",
      legend: { data: policies, textStyle: { color: "#9090b0", fontSize: 10 } },
      radar: {
        indicator: RADAR_METRICS.map(({ label }, i) => ({ name: label, max: maxes[i] * 1.1 })),
        axisLine: { lineStyle: { color: "#2d2d50" } },
        splitLine: { lineStyle: { color: "#2d2d50" } },
        name: { textStyle: { color: "#9090b0", fontSize: 9 } },
      },
      series: [
        {
          type: "radar" as const,
          data: policies.map((p, i) => ({
            name: p,
            value: RADAR_METRICS.map(({ key }) => displayValue(key, metricMeans[p][key] ?? 0)),
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
          return meta
            ? `${p.name}<br/>${formatPolicyMeta(meta)}`
            : p.name;
        },
      },
    };
  }, [stats, policies, brushed, policyMeta, logScale]);

  return (
    <div className="card space-y-2">
      <div className="flex items-center justify-between">
        <p className="text-xs font-semibold text-gray-300">
          Policy Radar{logScale ? " · log-normalised axes" : ""}
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

const HEATMAP_METRICS: Array<{ key: MetricKey | "kg/km"; label: string; higherBetter: boolean }> = [
  { key: "profit", label: "Profit", higherBetter: true },
  { key: "kg/km", label: "kg/km", higherBetter: true },
  { key: "overflows", label: "Overflows", higherBetter: false },
  { key: "km", label: "km", higherBetter: false },
];

function PolicyHeatmapChart({
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

function PolicyParallelChart({
  stats,
  policies,
  policyMeta,
  logMeta,
  brushed,
  logScale = false,
  onPolicyClick,
  onAxisBrush,
  onOverflowCorridorBrush,
}: {
  stats: Record<string, PolicyStats>;
  policies: string[];
  policyMeta: Record<string, PolicyMeta>;
  logMeta: LogPathMeta;
  brushed?: string[] | null;
  logScale?: boolean;
  onPolicyClick?: (policy: string) => void;
  onAxisBrush?: (policies: string[] | null) => void;
  /** Sync zero-overflow corridor slider when brushing the overflows axis (§G.1.4). */
  onOverflowCorridorBrush?: (maxOverflow: number | null) => void;
}) {
  const chartRef = useRef<EChartsReact | null>(null);

  const parallel = useMemo(
    () => buildPolicyParallelAxes(policies, stats, policyMeta, logMeta),
    [policies, stats, policyMeta, logMeta]
  );

  const transformedRows = useMemo(
    () =>
      parallel.rows.map((row) => ({
        ...row,
        value: row.value.map((v, dim) => {
          if (typeof v !== "number") return v;
          const axis = parallel.axes.find((a) => a.dim === dim);
          return axis ? parallelAxisValue(v, axis.name, logScale) : v;
        }),
      })),
    [parallel, logScale]
  );

  const option = useMemo(() => {
    const axisLabel = (name: string) => {
      if (!logScale) return name;
      if (name === "Overflows") return "Overflows (symlog)";
      if (["Profit", "kg/km", "km"].includes(name)) return `${name} (log-norm)`;
      return name;
    };

    return {
      backgroundColor: "transparent",
      brush: {
        toolbox: ["parallelAxis", "clear"],
        parallelAxisIndex: "all",
        throttleType: "debounce",
        throttleDelay: 300,
      },
      toolbox: {
        right: 8,
        top: 0,
        itemSize: 12,
        iconStyle: { borderColor: "#9090b0" },
        feature: {
          brush: { type: ["parallelAxis", "clear"] },
        },
      },
      parallelAxis: parallel.axes.map((s) => ({
        dim: s.dim,
        name: axisLabel(s.name),
        type: s.type,
        ...(s.data ? { data: s.data } : {}),
        ...(s.max != null
          ? {
              max:
                s.type === "value" && logScale
                  ? parallelAxisValue(s.max, s.name, true) * 1.1
                  : s.max,
            }
          : {}),
        nameTextStyle: { color: "#9090b0", fontSize: 9 },
        axisLine: { lineStyle: { color: "#2d2d50" } },
        axisLabel: { color: "#9090b0", fontSize: 8 },
      })),
      parallel: {
        left: 50,
        right: 24,
        top: 28,
        bottom: 36,
      },
      series: [
        {
          type: "parallel" as const,
          lineStyle: { width: 2 },
          data: transformedRows.map((row) => ({
            name: row.name,
            value: row.value,
            lineStyle: {
              color: strategyColor(row.name, policyMeta),
              opacity: barOpacity(row.name, brushed ?? null),
            },
          })),
        },
      ],
      tooltip: {
        formatter: (p: { name: string }) => {
          const meta = policyMeta[p.name];
          return meta ? `${p.name}<br/>${formatPolicyMeta(meta)}` : p.name;
        },
      },
    };
  }, [parallel, transformedRows, policyMeta, brushed, logScale]);

  const events = useMemo(() => {
    const handlers: Record<string, (params: unknown) => void> = {};
    if (onPolicyClick) {
      handlers.click = (params: unknown) => {
        const p = params as { name?: string };
        if (p.name) onPolicyClick(p.name);
      };
    }
    if (onAxisBrush) {
      handlers.brushselected = (params: unknown) => {
        const batch = (params as { batch?: Array<{ selected?: Array<{ dataIndex?: number[] }> }> })
          .batch;
        const indices = batch?.[0]?.selected?.[0]?.dataIndex;
        if (!indices?.length) {
          onAxisBrush(null);
          return;
        }
        onAxisBrush(indices.map((i) => policies[i]).filter(Boolean));
      };
    }
    if (onOverflowCorridorBrush) {
      handlers.brushEnd = (params: unknown) => {
        const areas = (params as { areas?: Array<{ parallelAxisIndex?: number; coordRange?: number[] }> })
          .areas;
        if (!areas?.length) {
          onOverflowCorridorBrush(null);
          return;
        }
        for (const area of areas) {
          if (area.parallelAxisIndex !== parallel.overflowDim || !area.coordRange?.length) continue;
          const range = area.coordRange;
          const hi = Math.max(range[0], range[range.length - 1]);
          onOverflowCorridorBrush(invertParallelAxisValue(hi, "Overflows", logScale));
          return;
        }
      };
    }
    return Object.keys(handlers).length ? handlers : undefined;
  }, [onPolicyClick, onAxisBrush, onOverflowCorridorBrush, policies, parallel.overflowDim, logScale]);

  return (
    <div className="card space-y-2">
      <div className="flex items-center justify-between">
        <div>
          <p className="text-xs font-semibold text-gray-300">
            Policy Parallel Coordinates{logScale ? " · log-normalised axes" : ""}
          </p>
          <p className="text-[10px] text-canvas-muted">
            One polyline per policy · coloured by packing algorithm
          </p>
          {(onAxisBrush || onOverflowCorridorBrush) && (
            <p className="text-[10px] text-canvas-muted">
              Drag on any axis to brush · overflows axis syncs corridor slider · toolbox clear resets
            </p>
          )}
        </div>
        <ChartExportButtons
          chartRef={{ current: chartRef.current }}
          filenameStem="summary-parallel"
        />
      </div>
      <StrategyLegend />
      <ReactECharts
        ref={chartRef}
        option={option}
        style={{ height: 220 }}
        onEvents={events}
      />
    </div>
  );
}

function HierarchyBreadcrumb({
  path,
  onNavigate,
}: {
  path: string[];
  onNavigate: (depth: number) => void;
}) {
  if (path.length === 0) return null;
  return (
    <div className="flex flex-wrap items-center gap-1 text-xs">
      <button
        onClick={() => onNavigate(0)}
        className="text-accent-primary hover:underline font-medium"
      >
        All
      </button>
      {path.map((seg, i) => (
        <span key={`${seg}-${i}`} className="flex items-center gap-1">
          <span className="text-canvas-muted">›</span>
          <button
            onClick={() => onNavigate(i + 1)}
            className="text-accent-secondary hover:underline font-mono"
          >
            {seg}
          </button>
        </span>
      ))}
    </div>
  );
}

function PolicyHierarchyPanel({
  stats,
  policies,
  policyMeta,
  logMeta,
  portfolioRuns,
  brushed,
  onBrushPolicies,
  showErrorBars = false,
  logScale = false,
}: {
  stats: Record<string, PolicyStats>;
  policies: string[];
  policyMeta: Record<string, PolicyMeta>;
  logMeta: LogPathMeta;
  portfolioRuns?: PortfolioHierarchyRun[];
  brushed?: string[] | null;
  onBrushPolicies: (ps: string[]) => void;
  showErrorBars?: boolean;
  logScale?: boolean;
}) {
  const chartRef = useRef<EChartsReact | null>(null);
  const [view, setView] = useState<HierarchyView>("sunburst");
  const [colorMode, setColorMode] = useState<HierarchyColorMode>("kgkm");
  const [drillPath, setDrillPath] = useState<string[]>([]);

  const tree = useMemo(() => {
    if (portfolioRuns && portfolioRuns.length >= 2) {
      return buildPortfolioHierarchy(portfolioRuns, colorMode);
    }
    return buildPolicyHierarchy(policies, stats, policyMeta, logMeta, colorMode);
  }, [portfolioRuns, colorMode, policies, stats, policyMeta, logMeta]);

  const drillChildren = useMemo(
    () => enrichDrillChildren(childrenAtPath(tree, drillPath), stats, policyMeta),
    [tree, drillPath, stats, policyMeta]
  );

  const showDrillMorph = drillPath.length > 0 && drillChildren.length > 0;

  const hierarchyOption = useMemo(
    () => ({
      backgroundColor: "transparent",
      animationDurationUpdate: 750,
      animationEasingUpdate: "cubicOut" as const,
      tooltip: {
        formatter: (p: { name: string; value: number }) =>
          `${p.name}<br/>Profit: ${fmt(p.value, 1)} €`,
      },
      series: showDrillMorph
        ? [
            {
              id: "hierarchy-drill",
              type: "bar" as const,
              data: drillChildren.map((c) => ({
                value: logScale ? Math.max(c.profit, 0.001) : c.profit,
                name: c.name,
                itemStyle: {
                  color: resolveDrillBarColor(c.name, c.policies, stats, colorMode),
                  opacity: c.policies.some((p) => isHighlighted(p, brushed ?? null)) ? 1 : 0.25,
                },
              })),
              universalTransition: { enabled: true },
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
                      const c = drillChildren[params.dataIndex];
                      const err = Math.max(c.profitStd, c.distSpread);
                      const bounds = errorBarBounds(c.profit, err, "profit", logScale);
                      const y = api.coord([bounds.center, params.dataIndex])[1];
                      const xLeft = api.coord([bounds.low, params.dataIndex])[0];
                      const xRight = api.coord([bounds.high, params.dataIndex])[0];
                      const cap = 4;
                      return {
                        type: "group",
                        children: [
                          {
                            type: "line",
                            shape: { x1: xLeft, y1: y, x2: xRight, y2: y },
                            style: api.style({ stroke: "#9090b0", lineWidth: 1.5 }),
                          },
                          {
                            type: "line",
                            shape: { x1: xLeft, y1: y - cap, x2: xLeft, y2: y + cap },
                            style: api.style({ stroke: "#9090b0", lineWidth: 1.5 }),
                          },
                          {
                            type: "line",
                            shape: { x1: xRight, y1: y - cap, x2: xRight, y2: y + cap },
                            style: api.style({ stroke: "#9090b0", lineWidth: 1.5 }),
                          },
                        ],
                      };
                    },
                    data: drillChildren.map((_, i) => i),
                    z: 10,
                  },
                ]
              : []),
          ]
        : [
            view === "sunburst"
              ? {
                  id: "hierarchy-sunburst",
                  type: "sunburst" as const,
                  data: tree,
                  radius: ["12%", "90%"],
                  label: { color: "#c0c0d8", fontSize: 9 },
                  itemStyle: { borderWidth: 1, borderColor: "#1a1a2e" },
                  emphasis: { focus: "ancestor" as const },
                  universalTransition: { enabled: true },
                }
              : {
                  id: "hierarchy-treemap",
                  type: "treemap" as const,
                  data: tree,
                  leafDepth: 2,
                  label: { color: "#c0c0d8", fontSize: 9 },
                  upperLabel: { show: true, height: 22, color: "#9090b0" },
                  itemStyle: { borderWidth: 1, borderColor: "#1a1a2e" },
                  universalTransition: { enabled: true },
                },
          ],
      ...(showDrillMorph
        ? {
            grid: { left: 110, right: 24, top: 12, bottom: 12 },
            xAxis: {
              type: (logScale ? "log" : "value") as "log" | "value",
              logBase: 10,
              name: logScale ? "Profit (€, log)" : "Profit (€)",
              nameTextStyle: { color: "#9090b0", fontSize: 9 },
              axisLabel: { color: "#9090b0", fontSize: 9 },
              minorSplitLine: { show: false },
            },
            yAxis: {
              type: "category" as const,
              data: drillChildren.map((c) => c.name),
              inverse: true,
              axisLabel: { color: "#9090b0", fontSize: 9 },
            },
          }
        : {}),
    }),
    [tree, view, showDrillMorph, drillChildren, brushed, showErrorBars, logScale]
  );

  const handleSegmentClick = (path: string[]) => {
    setDrillPath(path);
    onBrushPolicies(policiesAtPath(tree, path));
  };

  return (
    <div className="card space-y-3">
      <div className="flex items-center justify-between flex-wrap gap-2">
        <div>
          <p className="text-xs font-semibold text-gray-300">Policy Hierarchy (§G.2)</p>
          <p className="text-[10px] text-canvas-muted">
            Span = profit · color ={" "}
            {colorMode === "kgkm" ? "kg/km efficiency" : "mean overflows"} · strategy ring
            borders + drill bars use packing palette
          </p>
        </div>
        <div className="flex items-center gap-2">
          <div className="flex items-center gap-1 bg-canvas-elevated rounded-lg p-0.5">
            {(["kgkm", "overflows"] as const).map((m) => (
              <button
                key={m}
                onClick={() => setColorMode(m)}
                className={`text-xs px-2.5 py-1 rounded-md transition-colors ${
                  colorMode === m
                    ? "bg-accent-primary text-white"
                    : "text-canvas-muted hover:text-gray-200"
                }`}
              >
                {m === "kgkm" ? "kg/km" : "Overflows"}
              </button>
            ))}
          </div>
          <div className="flex items-center gap-1 bg-canvas-elevated rounded-lg p-0.5">
            {(["sunburst", "treemap"] as const).map((v) => (
              <button
                key={v}
                onClick={() => setView(v)}
                className={`text-xs px-2.5 py-1 rounded-md capitalize transition-colors ${
                  view === v
                    ? "bg-accent-primary text-white"
                    : "text-canvas-muted hover:text-gray-200"
                }`}
              >
                {v}
              </button>
            ))}
          </div>
          <ChartExportButtons
            chartRef={{ current: chartRef.current }}
            filenameStem={`summary-${view}`}
          />
        </div>
      </div>

      <HierarchyBreadcrumb
        path={drillPath}
        onNavigate={(depth) => {
          const next = drillPath.slice(0, depth);
          setDrillPath(next);
          onBrushPolicies(next.length ? policiesAtPath(tree, next) : policies);
        }}
      />

      <StrategyLegend />

      <ReactECharts
        ref={chartRef}
        option={hierarchyOption}
        notMerge={false}
        style={{
          height: showDrillMorph ? Math.max(180, drillChildren.length * 32) : 300,
        }}
        onEvents={{
          click: (params: { treePathInfo?: Array<{ name: string }>; name?: string }) => {
            if (params.treePathInfo?.length) {
              const path = params.treePathInfo.map((n) => n.name);
              handleSegmentClick(path);
              return;
            }
            if (showDrillMorph && params.name) {
              const child = drillChildren.find((c) => c.name === params.name);
              if (child) onBrushPolicies(child.policies);
            }
          },
        }}
      />
    </div>
  );
}

function DistributionFacetHeatmaps({
  stats,
  policies,
  policyMeta,
  heatmapMode,
  onModeChange,
  brushed,
  logScale = false,
}: {
  stats: Record<string, PolicyStats>;
  policies: string[];
  policyMeta: Record<string, PolicyMeta>;
  heatmapMode: HeatmapMode;
  onModeChange: (mode: HeatmapMode) => void;
  brushed?: string[] | null;
  logScale?: boolean;
}) {
  const facets = useMemo(() => {
    const map = new Map<string, string[]>();
    for (const p of policies) {
      const dist = policyMeta[p]?.distribution ?? "—";
      const list = map.get(dist) ?? [];
      list.push(p);
      map.set(dist, list);
    }
    return [...map.entries()];
  }, [policies, policyMeta]);

  if (facets.length <= 1) {
    return (
      <PolicyHeatmapChart
        stats={stats}
        policies={policies}
        mode={heatmapMode}
        onModeChange={onModeChange}
        brushed={brushed}
        policyMeta={policyMeta}
        logScale={logScale}
      />
    );
  }

  return (
    <div className="space-y-2">
      <p className="text-xs font-semibold text-gray-300">
        Policy Heatmaps by Distribution (§G.1.3)
        {logScale ? " · log-normalised" : ""}
      </p>
      <div className={`grid gap-4 ${facets.length >= 2 ? "grid-cols-1 lg:grid-cols-2" : ""}`}>
        {facets.map(([dist, ps]) => (
          <div key={dist} className="space-y-1">
            <p className="text-[10px] text-canvas-muted font-mono">{dist}</p>
            <PolicyHeatmapChart
              stats={stats}
              policies={ps}
              mode={heatmapMode}
              onModeChange={onModeChange}
              brushed={brushed}
              policyMeta={policyMeta}
              logScale={logScale}
            />
          </div>
        ))}
      </div>
    </div>
  );
}

function EfficiencyRankingChart({
  stats,
  policies,
  showErrorBars = false,
  logScale = false,
  brushed,
  onPolicyClick,
  policyMeta,
}: {
  stats: Record<string, PolicyStats>;
  policies: string[];
  showErrorBars?: boolean;
  logScale?: boolean;
  brushed?: string[] | null;
  onPolicyClick?: (policy: string) => void;
  policyMeta?: Record<string, PolicyMeta>;
}) {
  const chartRef = useRef<EChartsReact | null>(null);

  const ranked = useMemo(
    () =>
      [...policies]
        .map((p) => ({
          policy: p,
          mean: mean(stats[p]["kg/km"]),
          std: std(stats[p]["kg/km"]),
        }))
        .sort((a, b) => b.mean - a.mean),
    [stats, policies]
  );

  const option = useMemo(
    () => ({
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
        type: "category" as const,
        data: ranked.map((r) => r.policy),
        inverse: true,
        axisLabel: { color: "#9090b0", fontSize: 9 },
      },
      tooltip: {
        trigger: "axis" as const,
        formatter: (params: unknown[]) => {
          const p = (params as Array<{ name: string; value: number; dataIndex: number }>)[0];
          const r = ranked[p.dataIndex];
          const days = stats[r.policy]?.days ?? 0;
          const footer = policyTooltipFooter(r.policy, policyMeta, days);
          return `${r.policy}<br/>${fmt(r.mean, 2)} ± ${fmt(r.std, 2)} kg/km<br/>${footer}`;
        },
      },
      series: [
        {
          type: "bar" as const,
          data: ranked.map((r) => ({
            value: logScale ? Math.max(r.mean, 0.001) : r.mean,
            name: r.policy,
            itemStyle: {
              color: strategyColor(r.policy, policyMeta),
              opacity: barOpacity(r.policy, brushed ?? null),
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
                  const r = ranked[i];
                  const bounds = errorBarBounds(r.mean, r.std, "kg/km", logScale);
                  const y = api.coord([bounds.center, i])[1];
                  const xLeft = api.coord([bounds.low, i])[0];
                  const xRight = api.coord([bounds.high, i])[0];
                  const cap = 4;
                  return {
                    type: "group",
                    children: [
                      {
                        type: "line",
                        shape: { x1: xLeft, y1: y, x2: xRight, y2: y },
                        style: api.style({ stroke: "#9090b0", lineWidth: 1.5 }),
                      },
                      {
                        type: "line",
                        shape: { x1: xLeft, y1: y - cap, x2: xLeft, y2: y + cap },
                        style: api.style({ stroke: "#9090b0", lineWidth: 1.5 }),
                      },
                      {
                        type: "line",
                        shape: { x1: xRight, y1: y - cap, x2: xRight, y2: y + cap },
                        style: api.style({ stroke: "#9090b0", lineWidth: 1.5 }),
                      },
                    ],
                  };
                },
                data: ranked.map((_, i) => i),
                z: 10,
              },
            ]
          : []),
      ],
    }),
    [ranked, showErrorBars, logScale, brushed, policyMeta, stats]
  );

  return (
    <div className="card space-y-2">
      <div className="flex items-center justify-between">
        <p className="text-xs font-semibold text-gray-300">Efficiency Ranking (kg/km)</p>
        <ChartExportButtons
          chartRef={{ current: chartRef.current }}
          filenameStem="summary-efficiency-rank"
        />
      </div>
      <ReactECharts
        ref={chartRef}
        option={option}
        style={{ height: Math.max(160, ranked.length * 28) }}
        onEvents={
          onPolicyClick
            ? {
                click: (params: { name?: string }) => {
                  if (params.name) onPolicyClick(params.name);
                },
              }
            : undefined
        }
      />
    </div>
  );
}

function PolicyParetoChart({
  stats,
  policies,
  logScale = false,
  brushed,
  onPolicyClick,
  policyMeta,
  logMeta,
}: {
  stats: Record<string, PolicyStats>;
  policies: string[];
  logScale?: boolean;
  brushed?: string[] | null;
  onPolicyClick?: (policy: string) => void;
  policyMeta?: Record<string, PolicyMeta>;
  logMeta?: LogPathMeta | null;
}) {
  const chartRef = useRef<EChartsReact | null>(null);

  const option = useMemo(() => {
    const points = policies.map((p) => ({
      id: p,
      x: mean(stats[p].profit),
      y: mean(stats[p].overflows),
    }));
    const frontIds = new Set(paretoFront(points).map((p) => p.id));
    const step = paretoStepLine(paretoFront(points));
    const overflowSymlog = chartMetricUsesSymlog("overflows", logScale);
    const displayX = (x: number) => chartMetricDisplay(x, "profit", logScale) ?? x;
    const displayY = (y: number) => chartMetricDisplay(y, "overflows", logScale) ?? y;
    const displayStep = step.map(([x, y]) => [displayX(x), displayY(y)] as [number, number]);

    return {
      backgroundColor: "transparent",
      grid: { left: 50, right: 16, top: 24, bottom: 40 },
      xAxis: {
        type: (logScale ? "log" : "value") as "log" | "value",
        logBase: 10,
        name: logScale ? "Profit (€, log)" : "Profit (€)",
        nameTextStyle: { color: "#9090b0", fontSize: 9 },
        axisLabel: { color: "#9090b0", fontSize: 9 },
        minorSplitLine: { show: false },
      },
      yAxis: {
        type: (logScale && !overflowSymlog ? "log" : "value") as "log" | "value",
        logBase: 10,
        name: overflowSymlog ? "Overflows (symlog)" : logScale ? "Overflows (log)" : "Overflows",
        nameTextStyle: { color: "#9090b0", fontSize: 9 },
        axisLabel: { color: "#9090b0", fontSize: 9 },
        minorSplitLine: { show: false },
      },
      series: [
        {
          name: "Policies",
          type: "scatter" as const,
          data: points.map((pt) => {
            const onFront = frontIds.has(pt.id);
            const highlighted = isHighlighted(pt.id, brushed ?? null);
            return {
              name: pt.id,
              value: [displayX(pt.x), displayY(pt.y)],
              itemStyle: {
                color: onFront ? strategyColor(pt.id, policyMeta) : "#6b7280",
                opacity: highlighted ? 1 : 0.2,
              },
              symbol: citySymbol(logMeta ?? null),
              symbolSize: onFront ? 10 : 7,
            };
          }),
          tooltip: {
            formatter: (p: { name: string; value: [number, number] }) => {
              const pt = points.find((x) => x.id === p.name);
              const meta = policyMeta?.[p.name];
              const lines = [
                p.name,
                meta ? formatPolicyMeta(meta) : "",
                `Profit: ${fmt(pt?.x ?? p.value[0], 1)} €`,
                `Overflows: ${fmt(pt?.y ?? p.value[1], 1)}`,
                frontIds.has(p.name) ? "Pareto-optimal" : "",
              ].filter(Boolean);
              return lines.join("<br/>");
            },
          },
        },
        ...(displayStep.length > 1
          ? [
              {
                name: "Pareto front",
                type: "line" as const,
                data: displayStep,
                lineStyle: { color: "#f3f4f6", type: "dashed" as const, width: 1.5 },
                symbol: "none",
                tooltip: { show: false },
                z: 1,
              },
            ]
          : []),
      ],
      legend: {
        data: displayStep.length > 1 ? ["Policies", "Pareto front"] : ["Policies"],
        textStyle: { color: "#9090b0", fontSize: 9 },
        top: 0,
      },
    };
  }, [stats, policies, logScale, brushed, policyMeta, logMeta]);

  return (
    <div className="card space-y-2">
      <div className="flex items-center justify-between">
        <p className="text-xs font-semibold text-gray-300">
          Profit vs Overflows (Pareto)
          {logScale ? " · symlog overflows + log profit" : ""}
        </p>
        <ChartExportButtons
          chartRef={{ current: chartRef.current }}
          filenameStem="summary-pareto"
        />
      </div>
      <ReactECharts
        ref={chartRef}
        option={option}
        style={{ height: 260 }}
        onEvents={
          onPolicyClick
            ? {
                click: (params: { name?: string }) => {
                  if (params.name && params.name !== "Pareto front") onPolicyClick(params.name);
                },
              }
            : undefined
        }
      />
    </div>
  );
}

function MetricBarChart({
  title,
  policies,
  values,
  color,
  logScale = false,
  useSymlog = false,
  showErrorBars = false,
  exportName,
  brushed,
  onPolicyClick,
  policyMeta,
  stats,
}: {
  title: string;
  policies: string[];
  values: Array<{ mean: number; std: number }>;
  color: string;
  logScale?: boolean;
  useSymlog?: boolean;
  showErrorBars?: boolean;
  exportName: string;
  brushed?: string[] | null;
  onPolicyClick?: (policy: string) => void;
  policyMeta?: Record<string, PolicyMeta>;
  stats?: Record<string, PolicyStats>;
}) {
  const chartRef = useRef<EChartsReact | null>(null);
  const symlogMode = logScale && useSymlog;
  const option = useMemo(() => ({
    backgroundColor: "transparent",
    tooltip: {
      trigger: "axis" as const,
      formatter: (params: unknown[]) => {
        const p = (params as Array<{ name: string; value: number; dataIndex: number }>)[0];
        const raw = values[p.dataIndex];
        const days = stats?.[p.name]?.days ?? 0;
        const footer = policyTooltipFooter(p.name, policyMeta, days);
        return `${p.name}<br/>${fmt(raw?.mean ?? p.value, 2)} ± ${fmt(raw?.std ?? 0, 2)}<br/>${footer}`;
      },
    },
    xAxis: {
      type: "category" as const,
      data: policies,
      axisLabel: { color: "#9090b0", fontSize: 9, rotate: 30 },
    },
    yAxis: {
      type: (logScale && !symlogMode ? "log" : "value") as "log" | "value",
      logBase: 10,
      axisLabel: { color: "#9090b0", fontSize: 9 },
      minorSplitLine: { show: false },
    },
    grid: { left: 45, right: 10, top: 16, bottom: 55 },
    series: [
      {
        type: "bar" as const,
        data: policies.map((policy, i) => {
          const v = values[i];
          const display =
            !logScale ? v.mean : symlogMode ? symlog(v.mean) : Math.max(v.mean, 0.001);
          return {
            value: display,
            name: policy,
            itemStyle: { color, opacity: barOpacity(policy, brushed ?? null) },
          };
        }),
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
                const v = values[i];
                const metricKey = symlogMode ? "overflows" : "profit";
                const bounds = errorBarBounds(v.mean, v.std, metricKey, logScale, symlogMode);
                const x = api.coord([i, bounds.center])[0];
                const yTop = api.coord([i, bounds.high])[1];
                const yBot = api.coord([i, bounds.low])[1];
                const cap = 5;
                return {
                  type: "group",
                  children: [
                    {
                      type: "line",
                      shape: { x1: x, y1: yTop, x2: x, y2: yBot },
                      style: api.style({ stroke: "#9090b0", lineWidth: 1.5 }),
                    },
                    {
                      type: "line",
                      shape: { x1: x - cap, y1: yTop, x2: x + cap, y2: yTop },
                      style: api.style({ stroke: "#9090b0", lineWidth: 1.5 }),
                    },
                    {
                      type: "line",
                      shape: { x1: x - cap, y1: yBot, x2: x + cap, y2: yBot },
                      style: api.style({ stroke: "#9090b0", lineWidth: 1.5 }),
                    },
                  ],
                };
              },
              data: policies.map((_, i) => i),
              z: 10,
            },
          ]
        : []),
    ],
  }), [policies, values, color, logScale, symlogMode, showErrorBars, brushed, policyMeta, stats]);

  return (
    <div className="card">
      <div className="flex items-center justify-between mb-2">
        <p className="text-xs text-canvas-muted">{title}</p>
        <ChartExportButtons
          chartRef={{ current: chartRef.current }}
          filenameStem={exportName}
        />
      </div>
      <ReactECharts
        ref={chartRef}
        option={option}
        style={{ height: 190 }}
        onEvents={
          onPolicyClick
            ? {
                click: (params: { name?: string }) => {
                  if (params.name) onPolicyClick(params.name);
                },
              }
            : undefined
        }
      />
    </div>
  );
}

// ── Main page ─────────────────────────────────────────────────────────────────

interface ComparisonRun extends PortfolioRunSlice {}

export function SimulationSummary() {
  const { pendingLogPath, setPendingLogPath, effectiveTheme: theme } = useAppStore();
  const { projectRoot, handoff } = useRecentHandoff();
  const {
    ready: duckdbReady,
    loading: duckdbLoading,
    lastPipeline,
    setLastPipeline,
    setLoading: setDuckdbLoading,
  } = useDuckDbStore();
  const [parquetExporting, setParquetExporting] = useState(false);
  const logScale = useGlobalFiltersStore((s) => s.logScale);
  const [showErrorBars, setShowErrorBars] = useState(false);
  const [brushed, setBrushed] = useState<string[] | null>(null);
  const [heatmapMode, setHeatmapMode] = useState<HeatmapMode>("all");
  const [overflowMax, setOverflowMax] = useState<number | null>(null);
  const [portfolioLoading, setPortfolioLoading] = useState(false);
  const { policy, sampleId } = useGlobalFiltersStore(); // used by filteredEntries
  const [entries, setEntries] = useState<DayLogEntry[]>([]);
  const [logPath, setLogPath] = useState<string | null>(null);
  const derivedRunLabel = useLogPathRunLabelBrush(logPath);
  const [comparisonRuns, setComparisonRuns] = useState<ComparisonRun[]>([]);
  const [routeVizDay, setRouteVizDay] = useState(1);
  const [showFailureOverlay, setShowFailureOverlay] = useState(true);
  const [showRouteDiff, setShowRouteDiff] = useState(false);
  const cityCompareChartRef = useRef<EChartsReact | null>(null);

  const loadLog = useCallback(async (path: string) => {
    const loaded = await invoke<DayLogEntry[]>("load_simulation_log", { path });
    setEntries(loaded);
    setLogPath(path);
    handoff(path, "log", { navigate: false });
  }, [handoff]);

  const allDuckDbLogs = useMemo(() => {
    const logs: { path: string; label: string }[] = [];
    if (logPath) {
      logs.push({
        path: logPath,
        label: portfolioRunLabel(logPath, undefined, projectRoot),
      });
    }
    for (const r of comparisonRuns) {
      if (!logs.some((l) => l.path === r.path)) {
        logs.push({
          path: r.path,
          label: portfolioRunLabel(r.path, r.label, projectRoot),
        });
      }
    }
    return logs;
  }, [logPath, comparisonRuns, projectRoot]);

  useEffect(() => {
    if (!duckdbReady || allDuckDbLogs.length === 0) return;
    setDuckdbLoading(true);
    runPortfolioSimulationArrowPipeline(allDuckDbLogs, SUMMARY_SIM_TABLE, projectRoot)
      .then(setLastPipeline)
      .catch((err) => console.warn("Summary Arrow pipeline:", err))
      .finally(() => setDuckdbLoading(false));
  }, [allDuckDbLogs, duckdbReady, projectRoot, setLastPipeline, setDuckdbLoading]);

  // Auto-load when another page hands off a log path (e.g. OutputBrowser)
  useEffect(() => {
    if (pendingLogPath) {
      loadLog(pendingLogPath);
      setPendingLogPath(null);
    }
  }, [pendingLogPath, setPendingLogPath, loadLog]);

  const openLog = useCallback(async () => {
    const path = (await open({
      filters: [{ name: "Logs", extensions: ["jsonl", "log", "txt"] }],
    })) as string | null;
    if (!path) return;
    loadLog(path);
  }, [loadLog]);

  const addComparisonRun = useCallback(async () => {
    const path = (await open({
      filters: [{ name: "Logs", extensions: ["jsonl", "log", "txt"] }],
    })) as string | null;
    if (!path) return;
    try {
      const loaded = await invoke<DayLogEntry[]>("load_simulation_log", { path });
      const label = portfolioRunLabel(path, undefined, projectRoot);
      handoff(path, "log", { storedLabel: label, navigate: false });
      setComparisonRuns((prev) => [
        ...prev.filter((r) => r.path !== path),
        { path, label, entries: loaded },
      ]);
    } catch (err) {
      toast.error("Failed to load comparison log", { description: String(err) });
    }
  }, [projectRoot, handoff]);

  const loadOutputPortfolio = useCallback(async () => {
    if (!projectRoot) {
      toast.error("Set project root in Settings to scan output portfolio");
      return;
    }
    setPortfolioLoading(true);
    try {
      const refs = await scanOutputPortfolio(
        `${projectRoot}/assets/output`,
        PORTFOLIO_SCAN_DEFAULT
      );
      if (!refs.length) {
        toast.info("No simulation logs found under assets/output");
        return;
      }
      const progressId = toast.loading(`Loading portfolio… 0 / ${refs.length}`);
      const loaded = await loadPortfolioLogs(refs, {
        batchSize: 24,
        onProgress: (n, total) => {
          toast.loading(`Loading portfolio… ${n} / ${total}`, { id: progressId });
        },
      });
      if (loaded.length > 0) {
        const [primary, ...rest] = loaded;
        for (const r of loaded) {
          handoff(r.path, "log", { storedLabel: r.label, navigate: false });
        }
        if (!logPath) {
          await loadLog(primary.path);
        }
        setComparisonRuns((prev) => {
          const seen = new Set(prev.map((r) => r.path));
          if (logPath) seen.add(logPath);
          const normalize = (r: ComparisonRun): ComparisonRun => ({
            ...r,
            label: portfolioRunLabel(r.path, r.label, projectRoot),
          });
          return [
            ...prev,
            ...rest.filter((r) => !seen.has(r.path)).map(normalize),
            ...(logPath && primary.path !== logPath && !seen.has(primary.path)
              ? [normalize(primary)]
              : []),
          ];
        });
      }
      toast.success(`Loaded ${loaded.length} simulation log(s) from output portfolio`, {
        id: progressId,
      });
    } catch (err) {
      toast.error("Portfolio load failed", { description: String(err) });
    } finally {
      setPortfolioLoading(false);
    }
  }, [projectRoot, logPath, loadLog, handoff]);

  const removeComparisonRun = useCallback((path: string) => {
    setComparisonRuns((prev) => prev.filter((r) => r.path !== path));
  }, []);

  const filteredEntries = useMemo(
    () => filterEntries(entries, policy, sampleId),
    [entries, policy, sampleId]
  );

  const routeVizDays = useMemo(
    () => [...new Set(filteredEntries.map((e) => e.day))].sort((a, b) => a - b),
    [filteredEntries]
  );

  useEffect(() => {
    if (routeVizDays.length === 0) return;
    setRouteVizDay((d) => {
      if (routeVizDays.includes(d)) return d;
      return routeVizDays[routeVizDays.length - 1];
    });
  }, [routeVizDays]);

  const stats = useMemo(() => aggregateByPolicy(filteredEntries), [filteredEntries]);
  const policies = useMemo(() => Object.keys(stats), [stats]);

  const logMeta = useMemo(() => parseLogPath(logPath), [logPath]);
  const policyMeta = useMemo(() => {
    const map: Record<string, PolicyMeta> = {};
    for (const p of policies) map[p] = parsePolicyLabel(p);
    return map;
  }, [policies]);

  const handlePolicyClick = useCallback((p: string) => {
    setBrushed((prev) => toggleBrush(prev, p));
  }, []);

  const handleBrushPolicies = useCallback(
    (ps: string[] | null) => {
      if (!ps?.length) {
        setBrushed(null);
        return;
      }
      setBrushed(ps.length >= policies.length ? null : ps);
    },
    [policies.length]
  );

  const effectiveBrushed = useMemo(() => {
    if (overflowMax == null) return brushed;
    const corridor = policies.filter((p) => mean(stats[p].overflows) <= overflowMax);
    if (!brushed || brushed.length === 0) return corridor;
    return corridor.filter((p) => brushed.includes(p));
  }, [brushed, overflowMax, policies, stats]);

  const routeVizPolicies = useMemo(() => {
    if (effectiveBrushed?.length) return effectiveBrushed;
    return policies;
  }, [effectiveBrushed, policies]);

  const routeVizEntries = useMemo(
    () =>
      routeVizPolicies
        .map((p) => filteredEntries.find((e) => e.policy === p && e.day === routeVizDay))
        .filter((e): e is DayLogEntry => e != null && (e.data.all_bin_coords?.length ?? 0) > 0),
    [filteredEntries, routeVizPolicies, routeVizDay]
  );

  const routeVizHasFailureData = useMemo(
    () =>
      routeVizEntries.some(
        (e) =>
          e.data.failure_analysis?.has_failure &&
          ((e.data.failure_analysis.overflow_bins?.length ?? 0) > 0 ||
            (e.data.failure_analysis.skipped_high_fill_bins?.length ?? 0) > 0)
      ),
    [routeVizEntries]
  );

  const maxOverflow = useMemo(
    () => Math.max(...policies.map((p) => mean(stats[p].overflows)), 0),
    [policies, stats]
  );

  const overflowGroups = useMemo(
    () =>
      buildGroupedStats(
        policies,
        stats,
        (p) => policyMeta[p]?.selectionStrategy ?? "—",
        "overflows"
      ),
    [policies, stats, policyMeta]
  );

  const kgkmGroups = useMemo(
    () =>
      buildGroupedStats(
        policies,
        stats,
        (p) => policyMeta[p]?.constructor ?? "—",
        "kg/km"
      ),
    [policies, stats, policyMeta]
  );

  const allRuns = useMemo((): ComparisonRun[] => {
    const runs: ComparisonRun[] = [];
    if (logPath && entries.length > 0) {
      runs.push({
        path: logPath,
        label: portfolioRunLabel(logPath, undefined, projectRoot),
        entries: filteredEntries,
      });
    }
    for (const r of comparisonRuns) {
      if (r.path === logPath) continue;
      runs.push({
        ...r,
        label: portfolioRunLabel(r.path, r.label, projectRoot),
        entries: filterEntries(r.entries, policy, sampleId),
      });
    }
    return runs;
  }, [logPath, entries, filteredEntries, comparisonRuns, policy, sampleId, projectRoot]);

  const portfolioMode = allRuns.length >= 2;

  const portfolioHierarchyRuns = useMemo((): PortfolioHierarchyRun[] | undefined => {
    if (!portfolioMode) return undefined;
    return allRuns.map((run) => {
      const runStats = aggregateByPolicy(run.entries);
      const runPolicies = Object.keys(runStats);
      const runMeta: Record<string, PolicyMeta> = {};
      for (const p of runPolicies) runMeta[p] = parsePolicyLabel(p);
      return {
        path: run.path,
        policies: runPolicies,
        stats: runStats,
        policyMeta: runMeta,
      };
    });
  }, [allRuns, portfolioMode]);

  const paretoByPanel = useMemo(() => buildParetoByPanel(allRuns), [allRuns]);

  const cityGroups = useMemo(() => groupRunsByCity(allRuns), [allRuns]);

  const {
    runLabels: portfolioRunLabels,
    runLabel: activeRunLabel,
    brushedRunLabels,
    handleCityClick,
    handleRunLabelClick,
  } = usePortfolioRunBrush(allRuns);

  const cityComparisonOption = useMemo(
    () =>
      cityComparisonChartOption(buildCityComparisonSeries(cityGroups), {
        logScale,
        showErrorBars,
      }),
    [cityGroups, logScale, showErrorBars]
  );

  const handlePortfolioConfigClick = useCallback(
    (policyName: string, label: string) => {
      handlePolicyClick(policyName);
      handleRunLabelClick(label);
    },
    [handlePolicyClick, handleRunLabelClick]
  );

  const onCityChartClick = useCallback(
    (params: { name?: string }) => {
      if (params.name) handleCityClick(params.name);
    },
    [handleCityClick]
  );

  const cityScaleKgkmGroups = useMemo(() => {
    if (!portfolioMode) return null;
    return cityGroups.map(([label, runs]) => {
      const vals = runs.flatMap((r) =>
        r.entries
          .map((e) => e.data["kg/km"])
          .filter((v): v is number => v != null)
      );
      return { label, policies: [] as string[], mean: mean(vals), std: std(vals) };
    });
  }, [cityGroups, portfolioMode]);

  const cityScaleOverflowGroups = useMemo(() => {
    if (!portfolioMode) return null;
    return cityGroups.map(([label, runs]) => {
      const vals = runs.flatMap((r) =>
        r.entries
          .map((e) => e.data.overflows)
          .filter((v): v is number => v != null)
      );
      return { label, policies: [] as string[], mean: mean(vals), std: std(vals) };
    });
  }, [cityGroups, portfolioMode]);

  const distributionGroups = useMemo(() => groupRunsByDistribution(allRuns), [allRuns]);

  const rankingExportData = useCallback(() => {
    const cols: MetricKey[] = ["profit", "km", "overflows", "kg"];
    const headers = ["policy", ...cols.map((c) => `mean_${c}`), ...cols.map((c) => `std_${c}`), "days"];
    const rows = policies.map((p) => [
      p,
      ...cols.map((c) => mean(stats[p][c]).toFixed(4)),
      ...cols.map((c) => std(stats[p][c]).toFixed(4)),
      stats[p].days,
    ]);
    return { headers, rows };
  }, [policies, stats]);

  const exportRankingCsv = useCallback(() => {
    const { headers, rows } = rankingExportData();
    downloadCsv("simulation-ranking.csv", headers, rows);
  }, [rankingExportData]);

  const exportRankingParquet = useCallback(async () => {
    if (!projectRoot) {
      toast.error("Set project root in Settings to export Parquet");
      return;
    }
    const { headers, rows } = rankingExportData();
    setParquetExporting(true);
    try {
      const out = await downloadParquetTable(projectRoot, "simulation-ranking.parquet", headers, rows);
      if (out) toast.success("Parquet export complete", { description: out.split("/").pop() });
    } catch (err) {
      toast.error("Parquet export failed", { description: String(err) });
    } finally {
      setParquetExporting(false);
    }
  }, [projectRoot, rankingExportData]);

  const metricValues = (key: MetricKey) =>
    policies.map((p) => ({
      mean: mean(stats[p][key]),
      std: std(stats[p][key]),
    }));

  return (
    <div className="space-y-4">
      <GlobalFilterBar
        runLabels={
          portfolioMode
            ? portfolioRunLabels
            : derivedRunLabel
              ? [derivedRunLabel]
              : []
        }
        cities={portfolioMode ? cityGroups.map(([city]) => city) : []}
        showLogScale
      />

      <div className="flex items-center gap-3 flex-wrap">
        <button onClick={openLog} className="btn-primary flex items-center gap-2">
          <FolderOpen size={14} />
          Open Log File
        </button>
        {logPath && (
          <>
            <button onClick={() => void addComparisonRun()} className="btn-ghost flex items-center gap-2 text-xs">
              <FolderOpen size={14} />
              Add comparison log
            </button>
            {projectRoot && (
              <button
                onClick={() => void loadOutputPortfolio()}
                disabled={portfolioLoading}
                className="btn-ghost flex items-center gap-2 text-xs"
              >
                <FolderOpen size={14} />
                {portfolioLoading ? "Scanning output…" : "Load output portfolio"}
              </button>
            )}
            <OpenPathToolbar
              path={logPath}
              projectRoot={projectRoot}
              kind="log"
              labeled
              labeledTargets={["monitor"]}
              trailing={
                <>
                  {duckdbLoading && <span className="shrink-0">· DuckDB ingesting…</span>}
                  {!duckdbLoading && lastPipeline?.tableName === SUMMARY_SIM_TABLE && (
                    <span className="shrink-0">· {formatPipelineTimingBadge(lastPipeline)}</span>
                  )}
                </>
              }
            />
          </>
        )}
        {allRuns.length > 1 && (
          <span className="text-xs text-canvas-muted">{allRuns.length} runs loaded</span>
        )}
      </div>

      {comparisonRuns.length > 0 && (
        <div className="card">
          <p className="text-xs font-semibold text-canvas-muted uppercase tracking-wider mb-2">
            Comparison Runs
          </p>
          <div className="space-y-1">
            {comparisonRuns.map((r) => (
              <LoadedRunRow
                key={r.path}
                path={r.path}
                projectRoot={projectRoot}
                label={portfolioRunLabel(r.path, r.label, projectRoot)}
                activeRunLabel={activeRunLabel}
                onRemove={() => removeComparisonRun(r.path)}
                pathHandoffs
                trailing={
                  <span className="ml-auto text-canvas-muted shrink-0">{r.entries.length} days</span>
                }
              />
            ))}
          </div>
        </div>
      )}

      {entries.length === 0 && (
        <div className="flex items-center justify-center h-48 text-canvas-muted text-sm">
          Load a completed simulation log to view summary analytics.
        </div>
      )}

      {policies.length > 0 && (
        <>
          {logPath && (
            <ConfigMetaBanner logPath={logPath} logMeta={logMeta} projectRoot={projectRoot} />
          )}

          <PolicyBrushBar
            policies={policies}
            brushed={brushed}
            onToggle={handlePolicyClick}
            onClear={() => setBrushed(null)}
          />

          {/* Policy ranking table */}
          <RankingTable
            stats={stats}
            policies={policies}
            onExport={exportRankingCsv}
            onExportParquet={projectRoot ? exportRankingParquet : undefined}
            parquetExporting={parquetExporting}
          />

          <PolicyHierarchyPanel
            stats={stats}
            policies={policies}
            policyMeta={policyMeta}
            logMeta={logMeta}
            portfolioRuns={portfolioHierarchyRuns}
            brushed={effectiveBrushed}
            onBrushPolicies={handleBrushPolicies}
            showErrorBars={showErrorBars}
            logScale={logScale}
          />

          {portfolioMode && allRuns.length >= 2 && (
            <BenchmarkPortfolioParallel runs={allRuns} logScale={logScale} />
          )}

          {portfolioMode && (
            <BenchmarkPortfolioHeatmap
              runs={allRuns}
              heatmapMode={heatmapMode}
              onModeChange={setHeatmapMode}
              brushed={effectiveBrushed}
              logScale={logScale}
            />
          )}

          {allRuns.length >= 1 && (
            <div className="space-y-2">
              <p className="text-xs font-semibold text-gray-300">Pareto Panels (§G.1.2)</p>
              <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                {PARETO_PANELS.map((panel) => (
                  <BenchmarkParetoPanel
                    key={panel.id}
                    label={panel.label}
                    points={paretoByPanel[panel.id] ?? []}
                    logScale={logScale}
                  />
                ))}
              </div>
            </div>
          )}

          <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
            {cityScaleOverflowGroups ? (
              <GroupedMetricBarChart
                title={
                  logScale
                    ? "Overflows by City / Scale (symlog)"
                    : "Overflows by City / Scale"
                }
                subtitle="Mean ± std across portfolio runs (§G.1.1 multi-city)"
                groups={cityScaleOverflowGroups}
                color="#f87171"
                showErrorBars={showErrorBars}
                logScale={logScale}
                useSymlog
                exportName="summary-overflows-city"
              />
            ) : (
              <GroupedMetricBarChart
                title={
                  logScale
                    ? "Overflows by Selection Strategy (symlog)"
                    : "Overflows by Selection Strategy"
                }
                subtitle="Mean ± std across constructor variants (§G.1.1)"
                groups={overflowGroups}
                color="#f87171"
                showErrorBars={showErrorBars}
                logScale={logScale}
                useSymlog
                exportName="summary-overflows-grouped"
              />
            )}
            {cityScaleKgkmGroups ? (
              <GroupedMetricBarChart
                title="kg/km by City / Scale"
                subtitle="Mean ± std across portfolio runs (§G.1.1 multi-city)"
                groups={cityScaleKgkmGroups}
                color="#34d399"
                showErrorBars={showErrorBars}
                logScale={logScale}
                metricKey="kg/km"
                exportName="summary-kgkm-city"
              />
            ) : (
              <GroupedMetricBarChart
                title="kg/km by Constructor"
                subtitle="Mean ± std across selection strategies (§G.1.1)"
                groups={kgkmGroups}
                color="#34d399"
                showErrorBars={showErrorBars}
                logScale={logScale}
                metricKey="kg/km"
                exportName="summary-kgkm-grouped"
              />
            )}
          </div>

          {portfolioMode && distributionGroups.length > 1 && (
            <div className="space-y-2">
              <p className="text-xs font-semibold text-gray-300">Heatmaps by Distribution (§G.1.3)</p>
              <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
                {distributionGroups.map(([dist, distRuns]) => (
                  <BenchmarkDistributionHeatmap
                    key={dist}
                    distributionLabel={dist}
                    runs={distRuns}
                    heatmapMode={heatmapMode}
                    logScale={logScale}
                  />
                ))}
              </div>
            </div>
          )}

          {portfolioMode && cityGroups.length > 1 && (
            <div className="space-y-2">
              <p className="text-xs font-semibold text-gray-300">Heatmaps by Graph (§G.1.3)</p>
              <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
                {cityGroups.map(([graph, graphRuns]) => (
                  <BenchmarkGraphHeatmap
                    key={graph}
                    graphLabel={graph}
                    runs={graphRuns}
                    heatmapMode={heatmapMode}
                    logScale={logScale}
                  />
                ))}
              </div>
            </div>
          )}

          {portfolioMode && cityGroups.length >= 1 && (
            <div className="card space-y-2">
              <div className="flex items-center justify-between">
                <div>
                  <p className="text-xs font-semibold text-gray-300">City Comparison (§G.1.6)</p>
                  <p className="text-[10px] text-canvas-muted">
                    {logScale
                      ? `Log-scale bars — profit · symlog-overflows · kg/km${showErrorBars ? " · error bars on" : ""}`
                      : `Linear bars — profit · overflows · kg/km${showErrorBars ? " · error bars on" : ""}`}
                  </p>
                </div>
                <ChartExportButtons
                  chartRef={{ current: cityCompareChartRef.current }}
                  filenameStem="summary-city-compare"
                />
              </div>
              <ReactECharts
                ref={cityCompareChartRef}
                option={cityComparisonOption}
                style={{ height: 240 }}
                onEvents={{ click: onCityChartClick }}
              />
            </div>
          )}

          {routeVizDays.length > 0 && (
            <div className="space-y-2">
              <div className="flex items-center justify-between gap-2 flex-wrap">
                <div>
                  <p className="text-xs font-semibold text-gray-300">Route Solution (§A.1 / §A.6)</p>
                  <p className="text-[10px] text-canvas-muted">
                    ECharts spatial overlay — depot, demand-sized nodes, per-vehicle edges;
                    failure + route-diff toggles when comparing two policies (§A.6)
                  </p>
                </div>
                <div className="flex items-center gap-2 flex-wrap justify-end">
                  {routeVizHasFailureData && (
                    <button
                      className={`btn-ghost text-xs ${
                        showFailureOverlay ? "text-accent-danger" : ""
                      }`}
                      onClick={() => setShowFailureOverlay((v) => !v)}
                    >
                      {showFailureOverlay ? "Hide" : "Show"} failure overlay
                    </button>
                  )}
                  {routeVizEntries.length === 2 && (
                    <button
                      className={`btn-ghost text-xs ${showRouteDiff ? "text-accent-secondary" : ""}`}
                      onClick={() => setShowRouteDiff((v) => !v)}
                    >
                      {showRouteDiff ? "Hide" : "Show"} route diff
                    </button>
                  )}
                  <button
                    onClick={() => {
                      const idx = routeVizDays.indexOf(routeVizDay);
                      if (idx > 0) setRouteVizDay(routeVizDays[idx - 1]);
                    }}
                    disabled={routeVizDays.indexOf(routeVizDay) <= 0}
                    className="btn-ghost p-1"
                    title="Previous day"
                  >
                    <ChevronLeft size={14} />
                  </button>
                  <span className="text-xs font-mono text-gray-300 min-w-[4rem] text-center">
                    Day {routeVizDay}
                  </span>
                  <button
                    onClick={() => {
                      const idx = routeVizDays.indexOf(routeVizDay);
                      if (idx >= 0 && idx < routeVizDays.length - 1) {
                        setRouteVizDay(routeVizDays[idx + 1]);
                      }
                    }}
                    disabled={routeVizDays.indexOf(routeVizDay) >= routeVizDays.length - 1}
                    className="btn-ghost p-1"
                    title="Next day"
                  >
                    <ChevronRight size={14} />
                  </button>
                  <input
                    type="range"
                    min={routeVizDays[0]}
                    max={routeVizDays[routeVizDays.length - 1]}
                    value={routeVizDay}
                    onChange={(e) => setRouteVizDay(Number(e.target.value))}
                    className="w-28 accent-accent-primary"
                  />
                </div>
              </div>
              {routeVizEntries.length > 0 ? (
                showRouteDiff && routeVizEntries.length === 2 ? (
                  <RouteViz
                    data={routeVizEntries[0].data}
                    compareData={routeVizEntries[1].data}
                    primaryLabel={routeVizEntries[0].policy}
                    compareLabel={routeVizEntries[1].policy}
                    title="Route Solution"
                    subtitle={`Day ${routeVizDay} · overlay compare`}
                    filenameStem={`route-viz-overlay-day${routeVizDay}`}
                    showFailureOverlay={showFailureOverlay}
                    showTourDiff
                  />
                ) : (
                  <div
                    className={`grid gap-3 ${
                      routeVizEntries.length > 1 ? "grid-cols-1 lg:grid-cols-2" : "grid-cols-1"
                    }`}
                  >
                    {routeVizEntries.map((entry) => (
                      <RouteViz
                        key={`${entry.policy}-${entry.day}`}
                        data={entry.data}
                        title="Route Solution"
                        subtitle={`${parsePolicyLabel(entry.policy).selectionStrategy} · Day ${entry.day}`}
                        filenameStem={`route-viz-day${entry.day}-${entry.policy}`}
                        showFailureOverlay={showFailureOverlay}
                      />
                    ))}
                  </div>
                )
              ) : (
                <p className="text-xs text-canvas-muted py-4 text-center card">
                  No bin coordinates for day {routeVizDay} — run simulation with coordinate logging enabled.
                </p>
              )}
            </div>
          )}

          {/* Per-day trajectory */}
          <TrajectoryChart
            entries={filteredEntries}
            policies={policies}
            brushed={effectiveBrushed}
            logScale={logScale}
          />

          <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
            <PolicyRadarChart
              stats={stats}
              policies={policies}
              brushed={effectiveBrushed}
              policyMeta={policyMeta}
              logScale={logScale}
            />
            <div className="space-y-2">
              <div className="card space-y-2">
                <p className="text-xs font-semibold text-gray-300">Zero-overflow corridor (§G.1.4)</p>
                <div className="flex items-center gap-3">
                  <input
                    type="range"
                    min={0}
                    max={Math.max(maxOverflow, 1)}
                    step={0.5}
                    value={overflowMax ?? maxOverflow}
                    onChange={(e) => {
                      const v = Number(e.target.value);
                      setOverflowMax(v >= maxOverflow ? null : v);
                    }}
                    className="flex-1 accent-accent-primary"
                  />
                  <span className="text-xs font-mono text-canvas-muted w-24 text-right">
                    ≤ {overflowMax == null ? "off" : fmt(overflowMax, 1)}
                  </span>
                  {overflowMax != null && (
                    <button onClick={() => setOverflowMax(null)} className="btn-ghost text-xs">
                      Clear
                    </button>
                  )}
                </div>
              </div>
              <PolicyParallelChart
                stats={stats}
                policies={policies}
                policyMeta={policyMeta}
                logMeta={logMeta}
                brushed={effectiveBrushed}
                logScale={logScale}
                onPolicyClick={handlePolicyClick}
                onAxisBrush={handleBrushPolicies}
                onOverflowCorridorBrush={(max) => setOverflowMax(max)}
              />
            </div>
          </div>

          <DistributionFacetHeatmaps
            stats={stats}
            policies={policies}
            policyMeta={policyMeta}
            heatmapMode={heatmapMode}
            onModeChange={setHeatmapMode}
            brushed={effectiveBrushed}
            logScale={logScale}
          />

          {portfolioMode && (
            <PortfolioEfficiencyRanking
              runs={allRuns}
              showErrorBars={showErrorBars}
              logScale={logScale}
              brushed={effectiveBrushed}
              onConfigClick={handlePortfolioConfigClick}
            />
          )}

          <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
            {!portfolioMode && (
              <EfficiencyRankingChart
                stats={stats}
                policies={policies}
                showErrorBars={showErrorBars}
                logScale={logScale}
                brushed={effectiveBrushed}
                onPolicyClick={handlePolicyClick}
                policyMeta={policyMeta}
              />
            )}
            <div className={portfolioMode ? "lg:col-span-2" : ""}>
              <PolicyParetoChart
                stats={stats}
                policies={policies}
                logScale={logScale}
                brushed={effectiveBrushed}
                onPolicyClick={handlePolicyClick}
                policyMeta={policyMeta}
                logMeta={logMeta}
              />
            </div>
          </div>

          <div className="flex items-center justify-end gap-2">
            <button
              onClick={() => setShowErrorBars((v) => !v)}
              className={`btn-ghost text-xs ${showErrorBars ? "text-accent-secondary" : ""}`}
            >
              {showErrorBars ? "Error bars (on)" : "Error bars (off)"}
            </button>
          </div>

          {/* 4-metric bar charts */}
          <div className="grid grid-cols-1 sm:grid-cols-2 gap-4">
            <MetricBarChart
              title="Avg Profit by Policy (€) — hover for ±std"
              policies={policies}
              values={metricValues("profit")}
              color="#6366f1"
              logScale={logScale}
              useSymlog={logScale}
              showErrorBars={showErrorBars}
              exportName="summary-profit"
              brushed={effectiveBrushed}
              onPolicyClick={handlePolicyClick}
              policyMeta={policyMeta}
              stats={stats}
            />
            <MetricBarChart
              title="Avg Distance by Policy (km)"
              policies={policies}
              values={metricValues("km")}
              color="#38bdf8"
              logScale={logScale}
              useSymlog={logScale}
              showErrorBars={showErrorBars}
              exportName="summary-km"
              brushed={effectiveBrushed}
              onPolicyClick={handlePolicyClick}
              policyMeta={policyMeta}
              stats={stats}
            />
            <MetricBarChart
              title="Avg Overflows by Policy"
              policies={policies}
              values={metricValues("overflows")}
              color="#f87171"
              logScale={logScale}
              useSymlog
              showErrorBars={showErrorBars}
              exportName="summary-overflows"
              brushed={effectiveBrushed}
              onPolicyClick={handlePolicyClick}
              policyMeta={policyMeta}
              stats={stats}
            />
            <MetricBarChart
              title="Avg Waste Collected by Policy (kg)"
              policies={policies}
              values={metricValues("kg")}
              color="#34d399"
              logScale={logScale}
              useSymlog={logScale}
              showErrorBars={showErrorBars}
              exportName="summary-kg"
              brushed={effectiveBrushed}
              onPolicyClick={handlePolicyClick}
              policyMeta={policyMeta}
              stats={stats}
            />
          </div>

          {!logScale && (
            <div className="space-y-2">
              <p className="text-xs font-semibold text-gray-300">Log-scale views (§G.1.6)</p>
              <div className="grid grid-cols-2 gap-4">
                <MetricBarChart
                  title="Avg Profit by Policy (€) — symlog scale"
                  policies={policies}
                  values={metricValues("profit")}
                  color="#6366f1"
                  logScale
                  useSymlog
                  exportName="summary-profit-log"
                  brushed={effectiveBrushed}
                  onPolicyClick={handlePolicyClick}
                  policyMeta={policyMeta}
                  stats={stats}
                />
                <MetricBarChart
                  title="Avg Distance by Policy (km) — symlog scale"
                  policies={policies}
                  values={metricValues("km")}
                  color="#38bdf8"
                  logScale
                  useSymlog
                  exportName="summary-km-log"
                  brushed={effectiveBrushed}
                  onPolicyClick={handlePolicyClick}
                  policyMeta={policyMeta}
                  stats={stats}
                />
                <MetricBarChart
                  title="Avg Overflows by Policy — symlog scale"
                  policies={policies}
                  values={metricValues("overflows")}
                  color="#f87171"
                  logScale
                  useSymlog
                  exportName="summary-overflows-log"
                  brushed={effectiveBrushed}
                  onPolicyClick={handlePolicyClick}
                  policyMeta={policyMeta}
                  stats={stats}
                />
                <MetricBarChart
                  title="Avg Waste Collected by Policy (kg) — symlog scale"
                  policies={policies}
                  values={metricValues("kg")}
                  color="#34d399"
                  logScale
                  useSymlog
                  exportName="summary-kg-log"
                  brushed={effectiveBrushed}
                  onPolicyClick={handlePolicyClick}
                  policyMeta={policyMeta}
                  stats={stats}
                />
              </div>
            </div>
          )}

          {logPath && (
            <PolicyTelemetryTrendsPanel
              theme={theme}
              logScale={logScale}
              initialPolicy={effectiveBrushed?.length === 1 ? effectiveBrushed[0]! : null}
              initialRunLabel={
                derivedRunLabel ??
                activeRunLabel ??
                (brushedRunLabels?.length === 1 ? brushedRunLabels[0]! : null)
              }
            />
          )}

          {logPath && duckdbReady && (
            <SqlQueryPanel
              tableName={SUMMARY_SIM_TABLE}
              theme={theme}
              highlightPolicies={effectiveBrushed}
              highlightRunLabels={brushedRunLabels}
              brushSqlSync
              autoRunOnBrushSync
              portfolioMode={allDuckDbLogs.length > 1}
              portfolioRunLabels={portfolioMode ? portfolioRunLabels : []}
            />
          )}
        </>
      )}
    </div>
  );
}
