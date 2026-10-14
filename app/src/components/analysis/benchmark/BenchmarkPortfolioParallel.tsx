/**
 * Portfolio parallel coordinates — one polyline per loaded simulation log (§G.1.4).
 */
import { useMemo, useRef } from "react";
import ReactECharts from "echarts-for-react";
import type EChartsReact from "echarts-for-react";
import { ChartExportButtons } from "../../common/ChartExportButtons";
import { parallelAxisValue } from "../../../utils/charts/chartLogScale";
import {
  cityScaleLabel,
  parseLogPath,
  resolveRunSelectionStrategy,
  selectionStrategyColor,
} from "../../../utils/sim/simMetadata";
import type { DayLogEntry } from "../../../types";
import { StrategyLegend } from "../telemetry/StrategyLegend";

const AXIS_NAMES = ["N", "Profit", "kg/km", "Overflows", "km"] as const;

export interface PortfolioParallelRun {
  path: string;
  label: string;
  entries: DayLogEntry[];
}

function mean(arr: number[]) {
  return arr.length ? arr.reduce((a, b) => a + b, 0) / arr.length : 0;
}

export function BenchmarkPortfolioParallel({
  runs,
  logScale = false,
}: {
  runs: PortfolioParallelRun[];
  logScale?: boolean;
}) {
  const chartRef = useRef<EChartsReact | null>(null);
  const lines = useMemo(
    () =>
      runs.map((run) => {
        const meta = parseLogPath(run.path);
        const policies = [...new Set(run.entries.map((e) => e.policy))];
        const strategy = resolveRunSelectionStrategy(run.path, policies);
        const vals = run.entries.map((e) => e.data as Record<string, number>);
        const profit = mean(vals.map((d) => d.profit).filter((v): v is number => v != null));
        const kgkm = mean(vals.map((d) => d["kg/km"]).filter((v): v is number => v != null));
        const overflows = mean(vals.map((d) => d.overflows).filter((v): v is number => v != null));
        const km = mean(vals.map((d) => d.km).filter((v): v is number => v != null));
        const raw = [meta.scale ?? 100, profit, kgkm, overflows, km];
        const value = raw.map((v, idx) =>
          idx === 0 ? v : parallelAxisValue(v, AXIS_NAMES[idx], logScale)
        );
        return {
          name: run.label,
          color: selectionStrategyColor(strategy),
          strategy,
          value,
          meta,
        };
      }),
    [runs, logScale]
  );

  const option = useMemo(() => {
    const schema = [
      { dim: 0, name: "N", max: Math.max(...lines.map((l) => l.value[0]), 1) * 1.1 },
      {
        dim: 1,
        name: logScale ? "Profit (log-norm)" : "Profit",
        max: Math.max(...lines.map((l) => l.value[1]), 1) * 1.1,
      },
      {
        dim: 2,
        name: logScale ? "kg/km (log-norm)" : "kg/km",
        max: Math.max(...lines.map((l) => l.value[2]), 0.01) * 1.1,
      },
      {
        dim: 3,
        name: logScale ? "Overflows (symlog)" : "Overflows",
        max: Math.max(...lines.map((l) => l.value[3]), 1) * 1.1,
      },
      {
        dim: 4,
        name: logScale ? "km (log-norm)" : "km",
        max: Math.max(...lines.map((l) => l.value[4]), 1) * 1.1,
      },
    ];

    return {
      backgroundColor: "transparent",
      parallelAxis: schema.map((s) => ({
        dim: s.dim,
        name: s.name,
        nameTextStyle: { color: "#9090b0", fontSize: 9 },
        axisLine: { lineStyle: { color: "#2d2d50" } },
        axisLabel: { color: "#9090b0", fontSize: 8 },
      })),
      parallel: { left: 50, right: 24, top: 28, bottom: 36 },
      series: [
        {
          type: "parallel",
          lineStyle: { width: 1.5, opacity: 0.75 },
          data: lines.map((l) => ({
            name: l.name,
            value: l.value,
            lineStyle: { color: l.color },
          })),
        },
      ],
      tooltip: {
        formatter: (p: { name: string }) => {
          const line = lines.find((l) => l.name === p.name);
          if (!line) return p.name;
          return [
            p.name,
            line.strategy !== "—" ? `Strategy: ${line.strategy}` : null,
            cityScaleLabel(line.meta),
            line.meta.distribution,
            line.meta.improver,
          ]
            .filter(Boolean)
            .join("<br/>");
        },
      },
    };
  }, [lines, logScale]);

  return (
    <div className="card space-y-2">
      <div className="flex items-center justify-between gap-2">
        <p className="text-xs font-semibold text-gray-300">
          Portfolio Parallel Coordinates (§G.1.4)
          {logScale ? " · log-normalised axes" : ""}
        </p>
        <ChartExportButtons
          chartRef={{ current: chartRef.current }}
          filenameStem="portfolio-parallel"
          size={11}
          className="shrink-0"
        />
      </div>
      <p className="text-[10px] text-canvas-muted">
        {runs.length} simulation log(s) — one polyline per loaded run · coloured by packing algorithm
        strategy
      </p>
      <StrategyLegend />
      <ReactECharts
        ref={chartRef}
        option={option}
        style={{ height: Math.min(360, 120 + runs.length * 4) }}
      />
    </div>
  );
}
