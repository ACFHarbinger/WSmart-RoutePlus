import { useMemo, useRef, useState } from "react";
import ReactECharts from "echarts-for-react";
import type EChartsReact from "echarts-for-react";
import { ChevronDown, ChevronUp, Download } from "lucide-react";
import { ChartExportButtons } from "../../common/ChartExportButtons";
import { OpenPathToolbar } from "../../common/OpenPathToolbar";
import { errorBarBounds } from "../../../utils/charts/chartLogScale";
import { symlog } from "../../../utils/charts/symlog";
import {
  formatLogMeta,
  formatPolicyMeta,
  parsePolicyLabel,
  type LogPathMeta,
  type PolicyMeta,
} from "../../../utils/sim/simMetadata";
import type { DayLogEntry } from "../../../types";

export type HierarchyView = "sunburst" | "treemap";
export type MetricKey = "profit" | "km" | "overflows" | "kg";
export type SortKey = MetricKey;
export type SortDir = "asc" | "desc";

export interface PolicyStats {
  profit: number[];
  km: number[];
  overflows: number[];
  kg: number[];
  "kg/km": number[];
  cost: number[];
  days: number;
}

export interface GroupRow {
  label: string;
  mean: number;
  std: number;
  policies: string[];
}

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

export function fmt(n: number, decimals = 2): string {
  return isFinite(n) ? n.toFixed(decimals) : "—";
}

export function aggregateByPolicy(
  entries: DayLogEntry[]
): Record<string, PolicyStats> {
  const map: Record<string, PolicyStats> = {};
  for (const e of entries) {
    if (!map[e.policy]) {
      map[e.policy] = {
        profit: [],
        km: [],
        overflows: [],
        kg: [],
        "kg/km": [],
        cost: [],
        days: 0,
      };
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

export function trajectoryByPolicyAndDay(
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

export function policyTooltipFooter(
  policy: string,
  policyMeta: Record<string, PolicyMeta> | undefined,
  days: number
): string {
  const meta = policyMeta?.[policy];
  const lines = meta ? [formatPolicyMeta(meta)] : [];
  lines.push(`Days: ${days}`);
  return lines.join("<br/>");
}

export function buildGroupedStats(
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

export function ConfigMetaBanner({
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

export function PolicyBrushBar({
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
        <p className="text-xs font-semibold text-gray-300">
          Cross-filter (§G.1)
        </p>
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

export function GroupedMetricBarChart({
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
          return `${g.label}<br/>${fmt(g.mean, 2)} ± ${fmt(g.std, 2)}<br/>${
            g.policies.length
          } policy variant(s)`;
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
                  const bounds = errorBarBounds(
                    g.mean,
                    g.std,
                    errKey,
                    logScale,
                    symlogMode
                  );
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
          {subtitle && (
            <p className="text-[10px] text-canvas-muted">{subtitle}</p>
          )}
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

export function RankingTable({
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
    if (key === sortKey) setSortDir((d) => (d === "desc" ? "asc" : "desc"));
    else {
      setSortKey(key);
      setSortDir("desc");
    }
  };

  const SortIcon = ({ k }: { k: SortKey }) => {
    if (k !== sortKey) return <span className="w-3" />;
    return sortDir === "desc" ? (
      <ChevronDown size={11} />
    ) : (
      <ChevronUp size={11} />
    );
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
          <button
            onClick={onExport}
            className="btn-ghost text-xs flex items-center gap-1"
          >
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
      <table className="w-full text-xs min-w-[520px] tabular-nums">
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
            const color =
              POLICY_COLORS[policies.indexOf(policy) % POLICY_COLORS.length];
            return (
              <tr key={policy} className="hover:bg-canvas-hover">
                <td className="py-1.5 px-3 text-canvas-muted">{rank + 1}</td>
                <td className="py-1.5 px-3">
                  <span className="flex items-center gap-1.5">
                    <span
                      className="w-2 h-2 rounded-full shrink-0"
                      style={{ background: color }}
                    />
                    <span className="font-mono text-accent-secondary">
                      {policy}
                    </span>
                  </span>
                </td>
                {cols.map((c) => {
                  const vals = s[c.key];
                  const m = mean(vals);
                  const sd = std(vals);
                  return (
                    <td
                      key={c.key}
                      className="py-1.5 px-3 font-mono text-gray-300"
                    >
                      {fmt(m, 1)}
                      {sd > 0 && (
                        <span className="text-canvas-muted text-[10px] ml-1">
                          ±{fmt(sd, 1)}
                        </span>
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
