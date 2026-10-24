import { useMemo, useRef } from "react";
import ReactECharts from "echarts-for-react";
import type EChartsReact from "echarts-for-react";
import { ChartExportButtons } from "../../common/ChartExportButtons";
import {
  chartMetricDisplay,
  chartMetricUsesSymlog,
  errorBarBounds,
} from "../../../utils/charts/chartLogScale";
import { symlog } from "../../../utils/charts/symlog";
import { barOpacity, isHighlighted } from "../../../utils/charts/chartHighlight";
import {
  citySymbol,
  formatPolicyMeta,
  strategyColor,
  type LogPathMeta,
  type PolicyMeta,
} from "../../../utils/sim/simMetadata";
import { paretoFront, paretoStepLine } from "../../../utils/benchmark/pareto";
import {
  fmt,
  mean,
  policyTooltipFooter,
  std,
  type PolicyStats,
} from "./SummaryCoreWidgets";

export { PolicyHierarchyPanel } from "./SummaryHierarchyPanel";
export {
  DistributionFacetHeatmaps,
  PolicyParallelChart,
} from "./SummaryParallelAndFacets";

export function EfficiencyRankingChart({
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
          const p = (
            params as Array<{ name: string; value: number; dataIndex: number }>
          )[0];
          const r = ranked[p.dataIndex];
          const days = stats[r.policy]?.days ?? 0;
          const footer = policyTooltipFooter(r.policy, policyMeta, days);
          return `${r.policy}<br/>${fmt(r.mean, 2)} ± ${fmt(
            r.std,
            2
          )} kg/km<br/>${footer}`;
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
        <p className="text-xs font-semibold text-gray-300">
          Efficiency Ranking (kg/km)
        </p>
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

export function PolicyParetoChart({
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
    const displayY = (y: number) =>
      chartMetricDisplay(y, "overflows", logScale) ?? y;
    const displayStep = step.map(
      ([x, y]) => [displayX(x), displayY(y)] as [number, number]
    );

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
        name: overflowSymlog
          ? "Overflows (symlog)"
          : logScale
          ? "Overflows (log)"
          : "Overflows",
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
                lineStyle: {
                  color: "#f3f4f6",
                  type: "dashed" as const,
                  width: 1.5,
                },
                symbol: "none",
                tooltip: { show: false },
                z: 1,
              },
            ]
          : []),
      ],
      legend: {
        data:
          displayStep.length > 1
            ? ["Policies", "Pareto front"]
            : ["Policies"],
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
                  if (params.name && params.name !== "Pareto front")
                    onPolicyClick(params.name);
                },
              }
            : undefined
        }
      />
    </div>
  );
}

export function MetricBarChart({
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
  const option = useMemo(
    () => ({
      backgroundColor: "transparent",
      tooltip: {
        trigger: "axis" as const,
        formatter: (params: unknown[]) => {
          const p = (
            params as Array<{ name: string; value: number; dataIndex: number }>
          )[0];
          const raw = values[p.dataIndex];
          const days = stats?.[p.name]?.days ?? 0;
          const footer = policyTooltipFooter(p.name, policyMeta, days);
          return `${p.name}<br/>${fmt(raw?.mean ?? p.value, 2)} ± ${fmt(
            raw?.std ?? 0,
            2
          )}<br/>${footer}`;
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
            const display = !logScale
              ? v.mean
              : symlogMode
              ? symlog(v.mean)
              : Math.max(v.mean, 0.001);
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
                  const bounds = errorBarBounds(
                    v.mean,
                    v.std,
                    metricKey,
                    logScale,
                    symlogMode
                  );
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
    }),
    [
      policies,
      values,
      color,
      logScale,
      symlogMode,
      showErrorBars,
      brushed,
      policyMeta,
      stats,
    ]
  );

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
