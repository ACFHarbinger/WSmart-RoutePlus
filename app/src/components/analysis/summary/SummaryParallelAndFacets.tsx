import { useMemo, useRef } from "react";
import ReactECharts from "echarts-for-react";
import type EChartsReact from "echarts-for-react";
import { ChartExportButtons } from "../../common/ChartExportButtons";
import { StrategyLegend } from "../telemetry/StrategyLegend";
import {
  invertParallelAxisValue,
  parallelAxisValue,
} from "../../../utils/charts/chartLogScale";
import { barOpacity } from "../../../utils/charts/chartHighlight";
import {
  formatPolicyMeta,
  strategyColor,
  type LogPathMeta,
  type PolicyMeta,
} from "../../../utils/sim/simMetadata";
import { buildPolicyParallelAxes } from "../../../utils/benchmark/parallelPolicyAxes";
import type { HeatmapMode } from "../../../utils/benchmark/heatmapMetrics";
import type { PolicyStats } from "./SummaryCoreWidgets";
import { PolicyHeatmapChart } from "./SummaryComparisonCharts";

export function PolicyParallelChart({
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
        const batch = (
          params as {
            batch?: Array<{ selected?: Array<{ dataIndex?: number[] }> }>;
          }
        ).batch;
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
        const areas = (
          params as {
            areas?: Array<{ parallelAxisIndex?: number; coordRange?: number[] }>;
          }
        ).areas;
        if (!areas?.length) {
          onOverflowCorridorBrush(null);
          return;
        }
        for (const area of areas) {
          if (
            area.parallelAxisIndex !== parallel.overflowDim ||
            !area.coordRange?.length
          )
            continue;
          const range = area.coordRange;
          const hi = Math.max(range[0], range[range.length - 1]);
          onOverflowCorridorBrush(
            invertParallelAxisValue(hi, "Overflows", logScale)
          );
          return;
        }
      };
    }
    return Object.keys(handlers).length ? handlers : undefined;
  }, [
    onPolicyClick,
    onAxisBrush,
    onOverflowCorridorBrush,
    policies,
    parallel.overflowDim,
    logScale,
  ]);

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
              Drag on any axis to brush · overflows axis syncs corridor slider ·
              toolbox clear resets
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

export function DistributionFacetHeatmaps({
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
      <div
        className={`grid gap-4 ${
          facets.length >= 2 ? "grid-cols-1 lg:grid-cols-2" : ""
        }`}
      >
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
